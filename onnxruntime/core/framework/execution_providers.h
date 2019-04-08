// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

/**
Class for managing lookup of the execution providers in a session.
*/
class ExecutionProviders {
 public:
  ExecutionProviders() = default;

  common::Status Add(const std::string& provider_id, std::unique_ptr<IExecutionProvider> p_exec_provider) {
    // make sure there are no issues before we change any internal data structures
    if (provider_idx_map_.find(provider_id) != provider_idx_map_.end()) {
      auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " has already been registered.");
      LOGS_DEFAULT(ERROR) << status.ErrorMessage();
      return status;
    }

    for (const auto& allocator : p_exec_provider->GetAllocators()) {
      if (allocator_idx_map_.find(allocator->Info()) != allocator_idx_map_.end()) {
        auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, allocator->Info(), " allocator already registered.");
        LOGS_DEFAULT(ERROR) << status.ErrorMessage();
        return status;
      }
    }

    // index that provider will have after insertion
    auto new_provider_idx = exec_providers_.size();

    ORT_IGNORE_RETURN_VALUE(provider_idx_map_.insert({provider_id, new_provider_idx}));

    for (const auto& allocator : p_exec_provider->GetAllocators()) {
      ORT_IGNORE_RETURN_VALUE(allocator_idx_map_.insert({allocator->Info(), new_provider_idx}));
    }

    exec_providers_.push_back(std::move(p_exec_provider));

    return Status::OK();
  }

  const IExecutionProvider* Get(const onnxruntime::Node& node) const {
    return Get(node.GetExecutionProviderType());
  }

  const IExecutionProvider* Get(onnxruntime::ProviderType provider_id) const {
    auto it = provider_idx_map_.find(provider_id);
    if (it == provider_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  const IExecutionProvider* Get(const OrtAllocatorInfo& allocator_info) const {
    auto it = allocator_idx_map_.find(allocator_info);
    if (it == allocator_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  AllocatorPtr GetAllocator(const OrtAllocatorInfo& allocator_info) const {
    auto exec_provider = Get(allocator_info);
    if (exec_provider == nullptr) {
      return nullptr;
    }

    return exec_provider->GetAllocator(allocator_info.id, allocator_info.mem_type);
  }

  bool Empty() const { return exec_providers_.empty(); }

  size_t NumProviders() const { return exec_providers_.size(); }

  using const_iterator = typename std::vector<std::unique_ptr<IExecutionProvider>>::const_iterator;
  const_iterator begin() const noexcept { return exec_providers_.cbegin(); }
  const_iterator end() const noexcept { return exec_providers_.cend(); }

 private:
  std::vector<std::unique_ptr<IExecutionProvider>> exec_providers_;

  // maps for fast lookup of an index into exec_providers_
  std::unordered_map<std::string, size_t> provider_idx_map_;
  // using std::map as OrtAllocatorInfo would need a custom hash function to be used with unordered_map,
  // and as this isn't performance critical it's not worth the maintenance overhead of adding one.
  std::map<OrtAllocatorInfo, size_t> allocator_idx_map_;
};
}  // namespace onnxruntime
