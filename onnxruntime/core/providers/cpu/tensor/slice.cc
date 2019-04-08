// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {

#define ADD_TYPED_SLICE_OP(data_type)                                                   \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                       \
      Slice,                                                                            \
      1,                                                                                \
      data_type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Slice<data_type, false>);

ADD_TYPED_SLICE_OP(uint8_t);
ADD_TYPED_SLICE_OP(uint16_t);
ADD_TYPED_SLICE_OP(uint32_t);
ADD_TYPED_SLICE_OP(uint64_t);
ADD_TYPED_SLICE_OP(int8_t);
ADD_TYPED_SLICE_OP(int16_t);
ADD_TYPED_SLICE_OP(int32_t);
ADD_TYPED_SLICE_OP(int64_t);
ADD_TYPED_SLICE_OP(float);
ADD_TYPED_SLICE_OP(double);
ADD_TYPED_SLICE_OP(MLFloat16);
ADD_TYPED_SLICE_OP(bool);
ADD_TYPED_SLICE_OP(string);

#define ADD_TYPED_DYNAMIC_SLICE_OP(data_type)                                              \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                          \
      DynamicSlice,                                                                        \
      1,                                                                                   \
      data_type,                                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())     \
                        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(),   \
                                                 DataTypeImpl::GetTensorType<int64_t>()}), \
      Slice<data_type, true>);

ADD_TYPED_DYNAMIC_SLICE_OP(uint8_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint16_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int8_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int16_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(float);
ADD_TYPED_DYNAMIC_SLICE_OP(double);
ADD_TYPED_DYNAMIC_SLICE_OP(MLFloat16);
ADD_TYPED_DYNAMIC_SLICE_OP(bool);
ADD_TYPED_DYNAMIC_SLICE_OP(string);

namespace {
// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
}  // namespace

Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    const std::vector<int64_t>& input_dimensions,
                                    std::vector<int64_t>& starts,
                                    std::vector<int64_t>& output_dims) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);
  if (axes.size() == 0) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end ranges
  const auto& dimension_count = input_dimensions.size();
  for (size_t axesIndex = 0; axesIndex < axes.size(); axesIndex++) {
    auto axis = axes[axesIndex] < 0 ? axes[axesIndex] + static_cast<int64_t>(dimension_count) : axes[axesIndex];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    auto start = raw_starts[axesIndex];
    if (start < 0)
      start += input_dimensions[axis];
    starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    auto end = raw_ends[axesIndex];
    if (end < 0)
      end += input_dimensions[axis];
    output_dims[axis] = clamp(end, int64_t{0}, input_dimensions[axis]) - starts[axis];
    if (output_dims[axis] < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'starts' and 'ends' values resulted in a negative dimension");
  }

  return Status::OK();
}

void SliceBase::FillVectorsFromInput(const OpKernelContext* context,
                                     std::vector<int64_t>& input_starts,
                                     std::vector<int64_t>& input_ends,
                                     std::vector<int64_t>& input_axes) const {
  auto start_tensor = context->Input<Tensor>(1);
  auto ends_tensor = context->Input<Tensor>(2);
  auto axes_tensor = context->Input<Tensor>(3);

  ORT_ENFORCE(nullptr != start_tensor && start_tensor->Shape().NumDimensions() == 1, "Starts must be a 1-D array");
  ORT_ENFORCE(nullptr != ends_tensor && ends_tensor->Shape().NumDimensions() == 1, "Ends must be a 1-D array");
  ORT_ENFORCE(start_tensor->Shape() == ends_tensor->Shape(), "Starts and ends shape mismatch");
  ORT_ENFORCE(nullptr == axes_tensor || start_tensor->Shape() == axes_tensor->Shape(), "Starts and axes shape mismatch");

  const auto& dtype = start_tensor->DataType();
  const auto& size = start_tensor->Shape().Size();
  input_starts.resize(size);
  input_ends.resize(size);
  if (nullptr != axes_tensor)
    input_axes.resize(size);

  if (dtype == DataTypeImpl::GetType<int32_t>()) {
    std::copy(start_tensor->Data<int32_t>(), start_tensor->Data<int32_t>() + size, input_starts.begin());
    std::copy(ends_tensor->Data<int32_t>(), ends_tensor->Data<int32_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int32_t>(), axes_tensor->Data<int32_t>() + size, input_axes.begin());
  }

  else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    std::copy(start_tensor->Data<int64_t>(), start_tensor->Data<int64_t>() + size, input_starts.begin());
    std::copy(ends_tensor->Data<int64_t>(), ends_tensor->Data<int64_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int64_t>(), axes_tensor->Data<int64_t>() + size, input_axes.begin());
  }

  // should not reach this as no kernel is registered for this condition to be triggered - just an additional safety check
  else {
    ORT_THROW("Data type for starts and ends inputs' need to be int32_t or int64_t, but instead got ", dtype);
  }
}

template <typename T>
Status SliceImpl(OpKernelContext* ctx,
	             const Tensor& input_tensor,
                 std::vector<int64_t>& output_dims,
                 const std::vector<int64_t>& starts) {
  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<T>();
  const auto* output_end = output + output_tensor.Shape().Size();

  SliceIterator<T> input_iterator(input_tensor, starts, output_tensor.Shape().GetDims());
  while (output != output_end)
    *output++ = *input_iterator++;

  return Status::OK();
}

template <typename T, bool dynamic>
Status Slice<T, dynamic>::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr, "Missing input tensor to be processed");
  const auto& input_tensor = *input_tensor_ptr;
  const auto& input_dimensions = input_tensor.Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  std::vector<int64_t> starts(input_dimensions.size(), 0);
  std::vector<int64_t> output_dims(input_dimensions);

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes;
    FillVectorsFromInput(ctx, input_starts, input_ends, input_axes);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                                          input_dimensions, starts, output_dims));
  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                                          input_dimensions, starts, output_dims));
  }

  return SliceImpl<T>(ctx, input_tensor, output_dims, starts);
}
}  // namespace onnxruntime
