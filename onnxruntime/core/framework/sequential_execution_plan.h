// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
// Every ml-value has a unique name and is assigned a unique integral number.
// While we use names at static-planning time, the goal is that at runtime
// (that is, at inference time), there is no need to refer to names, and only
// the integer index is used (e.g., to index into appropriate vectors in
// the ExecutionFrame).
using MLValueIndex = int;
using MLValueName = std::string;

class SessionState;

// AllocPlanPerValue: (a simplified form of AllocationPlanPerValue above)
// Captures information required to allocate/reuse buffer for a ml-value
struct AllocPlanPerValue {
  AllocKind alloc_kind{AllocKind::kAllocate};
  MLDataType value_type{nullptr};
  OrtAllocatorInfo location;
  // reused_buffer is valid only if alloc_kind == kReuse. It indicates
  // which MLValue's buffer must be reused for this MLValue.
  MLValueIndex reused_buffer{0};
  // if the value is used in async kernel, a fence object would be created
  // note the fence object would be shared between MLValues reusing the same buffer
  bool create_fence_if_async{false};

 public:
  AllocPlanPerValue() : location(CPU, OrtArenaAllocator) {}
};

// SequentialExecutionPlan: This is the data that is produced by a static
// planner for a sequential execution, to be used by a SequentialExecutor.
struct SequentialExecutionPlan {
  // Allocation plan:
  // ExecutionFrame::GetOrCreateTensor() should use the following information
  // to decide whether to allocate a new buffer or reuse an existing buffer

  // The following vector is indexed by MLValueIndex
  std::vector<AllocPlanPerValue> allocation_plan;

  // The following indicates the order in which nodes should be executed and the
  // ml-values to be free after each node's execution:

  // NodeExecutionPlan: represents execution data for a single node
  struct NodeExecutionPlan {
    // node to be executed;
    onnxruntime::NodeIndex node_index;

    // ml-values to be freed after node execution:
    // for (auto i = free_from_index; i <= free_to_index; i++)
    //    free ml-value corresponding to ml-value-index to_be_freed[i]
    int free_from_index;
    int free_to_index;

    explicit NodeExecutionPlan(onnxruntime::NodeIndex index) : node_index(index), free_from_index(1), free_to_index(0) {}
  };

  // Execution_plan: represents the nodes in the sequential order to be executed
  std::vector<NodeExecutionPlan> execution_plan;

  // to_be_freed: vector elements represent indices of ml-values to be freed (as described above)
  std::vector<MLValueIndex> to_be_freed;
};

// Output details of an execution plan:
std::ostream& operator<<(std::ostream& out, std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo);
}  // namespace onnxruntime
