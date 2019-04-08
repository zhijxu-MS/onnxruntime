// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
template <int OpSet>
class Scan final : public OpKernel {
 public:
  Scan(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t num_scan_inputs_;
  std::vector<int64_t> input_directions_;
  std::vector<int64_t> output_directions_;
  std::vector<int64_t> input_axes_;
  std::vector<int64_t> output_axes_;

  mutable std::unique_ptr<FeedsFetchesManager> cached_feeds_fetches_manager_;
};
}  // namespace onnxruntime
