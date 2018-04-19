#include "core/providers/cpu/ml/scaler.h"

/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(Scaler)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
  Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
  )DOC")
.Input(0, "X", "Data to be scaled", "T")
.Output(0, "Y", "Scaled output data", "tensor(float)")
.TypeConstraint(
  "T",
  { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
  " allowed types.")
.Attr(
  "scale",
  "second, multiply by this, can be length of features or length 1",
  AttributeProto::FLOATS,
  OPTIONAL)
.Attr(
  "offset",
  "first, offset by this, must be same length as scale",
  AttributeProto::FLOATS,
  OPTIONAL);
*/

namespace Lotus {
namespace ML {

REGISTER_KERNEL(KernelDefBuilder("Scaler")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .MayInplace(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ScalerOp<float>);

template <typename T>
ScalerOp<T>::ScalerOp(const OpKernelInfo& info) : OpKernel(info) {
  op_kernel_info_.GetAttrs<float>("scale", scale_);    // optional
  op_kernel_info_.GetAttrs<float>("offset", offset_);  // optional
  LOTUS_ENFORCE(!scale_.empty(), "Empty scale in attributes");
  LOTUS_ENFORCE(scale_.size() == offset_.size(),
                "Scale size: (" + std::to_string(scale_.size()) + ") != (" + std::to_string(offset_.size()) + ")");
}

template <typename T>
Common::Status ScalerOp<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  Tensor* Y = context->Output(0, x_shape);
  const T* x_data = X.Data<T>();
  float* y_data = Y->MutableData<float>();
  const vector<int64_t>& x_dims = x_shape.GetDims();
  if (x_dims.empty()) {
    return Status(LOTUS, INVALID_ARGUMENT, "Invalid argument: input has empty dimensions.");
  }

  size_t x_size = x_shape.Size();
  int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];
  if (static_cast<int64_t>(offset_.size()) == stride &&
      static_cast<int64_t>(scale_.size()) == stride) {
    for (size_t i = 0; i < x_size; i++) {
      y_data[i] = static_cast<float>((x_data[i] - offset_[i % stride]) * scale_[i % stride]);
    }
  } else if (offset_.size() == 1 && scale_.size() == 1) {
    for (size_t i = 0; i < x_size; i++) {
      y_data[i] = static_cast<float>((x_data[i] - offset_[0]) * scale_[0]);
    }
  } else {
    std::ostringstream err_msg;
    err_msg << "Either both scale and offset can be of feature size (" << stride << ") or 1";
    return Status(LOTUS, INVALID_ARGUMENT, err_msg.str());
  }
  return Status::OK();
}
}  // namespace ML
}  // namespace Lotus
