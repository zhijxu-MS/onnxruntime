/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/tensor/onehot.h"
#include "core/util/eigen_common_wrapper.h"
#include "core/platform/env.h"

#define EIGEN_USE_THREADS

using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot

// T1: indices, T2: depth, T3: values
#define REG_TYPED_ONE_HOT_OP(types_str, in_type, out_type, depth_type)     \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                          \
      OneHot,                                                              \
      9,                                                                   \
      types_str,                                                           \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<depth_type>()) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<out_type>()),  \
      OneHotOp<in_type, out_type, depth_type>);

#define REG_ONE_HOT_OP(in_type, out_type, depth_type) \
  REG_TYPED_ONE_HOT_OP(in_type##_##out_type##_##depth_type, in_type, out_type, depth_type)

REG_ONE_HOT_OP(int64_t, int64_t, int64_t);
REG_ONE_HOT_OP(float, int64_t, int64_t);
REG_ONE_HOT_OP(int64_t, string, int64_t);
REG_ONE_HOT_OP(float, string, int64_t);
REG_ONE_HOT_OP(float, float, float);      // added this to satisfy onnx model tests
REG_ONE_HOT_OP(int64_t, int32_t, float);  // added this to satisfy onnx model tests

Status ValidateInputs(const Tensor* depth,
                      const Tensor* values) {
  // validation scenarios
  // depth should be scalar and > 0
  if (!depth->Shape().IsScalar()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid argument for depth; it's not a scalar.");
  }

  // values Rank 1 tensor containing exactly two elements
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }

  return Status::OK();
}

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct EigenTensorTypes {
  using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using Scalar = Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstScalar = Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
};

namespace generator {
template <typename in_type, typename out_type>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename EigenTensorTypes<in_type>::ConstMatrix& indices,
               const typename EigenTensorTypes<out_type>::ConstScalar& on_value,
               const typename EigenTensorTypes<out_type>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE out_type
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename EigenTensorTypes<in_type>::ConstMatrix indices_;
  const typename EigenTensorTypes<out_type>::ConstScalar on_value_;
  const typename EigenTensorTypes<out_type>::ConstScalar off_value_;
};
}  // namespace generator

template <typename in_type, typename out_type, typename depth_type>
Status OneHotOp<in_type, out_type, depth_type>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* indices = p_op_kernel_context->Input<Tensor>(0);
  const auto* depth = p_op_kernel_context->Input<Tensor>(1);
  const auto* values = p_op_kernel_context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(ValidateInputs(depth, values));

  // prepare output shape
  const auto* depth_data = depth->Data<depth_type>();
  const int64_t depth_val = static_cast<int64_t>(*depth_data);  // As per spec in case 'depth' is of non-integer type, it will be casted to int64 before use.
  if (depth_val <= 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Depth is negative.");
  }

  const auto& indices_shape = indices->Shape();
  const auto& indices_dims = indices_shape.GetDims();
  const auto indices_num_dims = indices_shape.NumDimensions();
  std::vector<int64_t> output_shape(indices_shape.GetDims());
  output_shape.insert(axis_ == -1 ? output_shape.end() : output_shape.begin() + axis_,
                      depth_val);

  // allocate output
  const auto* values_data = values->Data<out_type>();
  Tensor* output = p_op_kernel_context->Output(0, TensorShape(output_shape));

  const int64_t axis = (axis_ == -1) ? indices_num_dims : axis_;
  int64_t prefix_dim_size = 1;
  for (int64_t i = 0; i < axis; ++i) {
    prefix_dim_size *= indices_dims[i];
  }
  const int64_t suffix_dim_size = indices_shape.Size() / prefix_dim_size;

  // Split indices into matrix of size prefix_dim_size x suffix_dim_size
  Eigen::array<Eigen::DenseIndex, 2> indices_dims_e = {{prefix_dim_size, suffix_dim_size}};
  const auto* indices_data = indices->Data<in_type>();
  typename EigenTensorTypes<in_type, 2>::ConstEigenTensorMap indices_tensor_e(indices_data, indices_dims_e);

  // Split output into 3-Tensor of size:
  //   prefix_dim_size x depth x suffix_dim_size.
  Eigen::array<Eigen::DenseIndex, 3> output_dims_e = {{prefix_dim_size, depth_val, suffix_dim_size}};
  auto* output_data = output->MutableData<out_type>();
  typename EigenTensorTypes<out_type, 3>::EigenTensorMap output_tensor_e(output_data, output_dims_e);

  typename EigenTensorTypes<out_type>::ConstScalar on_value_e(values_data + 1);
  typename EigenTensorTypes<out_type>::ConstScalar off_value_e(values_data);

  generator::OneGenerator<in_type, out_type> generator(indices_tensor_e, on_value_e, off_value_e);

  // TODO potential optimization opportunity
  // TODO figure out the eigen threadpool stuff for use here
  output_tensor_e = output_tensor_e.generate(generator);

  return Status::OK();
}
}  // namespace onnxruntime
