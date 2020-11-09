// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sequential_executor.h"

#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include <iomanip>
#include <fstream>
#include "core/util/math.h"
#include <iostream>

#if defined DEBUG_NODE_INPUTS_OUTPUTS
#include "core/framework/debug_node_inputs_outputs_utils.h"
#endif

#ifdef ENABLE_NVTX_PROFILE
// This header is for profile using Nvidia's visual profilier.
#include "core/profile/profile.h"
#include "core/profile/context.h"
#endif

// #define TRACE_EXECUTION

// Define this symbol to create Concurrency Visualizer markers.
// See https://docs.microsoft.com/en-us/visualstudio/profiling/concurrency-visualizer-sdk
// You will need to install Concurrency Visualizer and add the SDK to the project that compiles this file
// via Analyze->Concurrency Visualizer->Add SDK to Project...
// #define CONCURRENCY_VISUALIZER
#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
using namespace Concurrency;
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include <Windows.h>
#include "core/platform/tracing.h"
namespace {
LARGE_INTEGER OrtGetPerformanceFrequency() {
  LARGE_INTEGER v;
  // On systems that run Windows XP or later, the QueryPerformanceFrequency function will always succeed
  // and will thus never return zero.
  (void)QueryPerformanceFrequency(&v);
  return v;
}

LARGE_INTEGER perf_freq = OrtGetPerformanceFrequency();
}  // namespace
#endif

namespace onnxruntime {


class InputDebugger {
 public:
 static int curr_iter;
 static int start_iter;
 static int end_iter;
 static unsigned int element_counts;

  void handle_node(OpKernelContext& context, const Node& node, const SessionState& session_state) {
    if (curr_iter < start_iter || curr_iter >= end_iter) {
      return;
    }

    if (print_node_infos) {
      std::cout << "Node: " << node.Name() << ", Op Type: " << node.OpType() << ", Inputs: ";
    }

    if (!all_nodes && enabled_nodes.find(node.Name()) == enabled_nodes.end()) {
      return;
    }

    const auto& input_defs = node.InputDefs();
    for (int i = 0; i < (int)input_defs.size(); i++) {
      std::string arg_name = input_defs[i]->Name();
      if (print_node_infos) {
        std::cout << arg_name << ", ";
      }

      if (!input_defs[i]->Exists() || (!all_nodes && enabled_nodes[node.Name()].find(i) == enabled_nodes[node.Name()].end())) {
        continue;
      }

      if (processed_inputs.find(arg_name) != processed_inputs.end()) {
        continue;
      }

      processed_inputs.insert(arg_name);
      const auto* type = context.InputType(i);
      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Input<Tensor>(i);
          const auto& shape = tensor.Shape();
          auto& tensor_location = tensor.Location();
          const auto data_type = tensor.DataType();
          if (tensor_location.device.Type() == OrtDevice::GPU &&
              (data_type->AsPrimitiveDataType()->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
              data_type->AsPrimitiveDataType()->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
              data_type->AsPrimitiveDataType()->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)) {
            const auto& execution_providers = session_state.GetExecutionProviders();
            const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
            auto cpu_allocator = cpu_execution_provider->GetAllocator(0, OrtMemTypeDefault);
            std::unique_ptr<Tensor> cpu_tensor = onnxruntime::make_unique<Tensor>(data_type, shape, cpu_allocator);
            const auto& data_transfer_mgr = session_state.GetDataTransferMgr();
            auto status = data_transfer_mgr.CopyTensor(tensor, *cpu_tensor.get(), 0);
            if (status == common::Status::OK()) {
              size_t num_items = shape.Size();
              if (num_items > element_counts) {
                num_items = element_counts;
              }

              input_tensors[arg_name] = std::vector<float>();
              if (data_type->AsPrimitiveDataType()->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
                auto float_data = cpu_tensor.get()->DataAsSpan<float>();
                for (size_t j = 0; j < num_items; j++) {
                  input_tensors[arg_name].push_back(float_data[j]);
                }
              } else if (data_type->AsPrimitiveDataType()->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
                auto fp16_data = cpu_tensor.get()->DataAsSpan<MLFloat16>();
                for (size_t j = 0; j < num_items; j++) {
                  input_tensors[arg_name].push_back(math::halfToFloat(fp16_data[j].val));
                }
              } else {
                auto bf16_data = cpu_tensor.get()->DataAsSpan<BFloat16>();
                for (size_t j = 0; j < num_items; j++) {
                  input_tensors[arg_name].push_back(bf16_data[j].ToFloat());
                }
              }
            }
          }
        }
      }
    }

    if (print_node_infos) {
      std::cout << std::endl;
    }
  }

  void print() {
    if (curr_iter < start_iter || curr_iter >= end_iter) {
      return;
    }

    std::map<std::string, std::vector<float>>::iterator it = input_tensors.begin();
    while (it != input_tensors.end()) {
      std::cout << "Arg Name: " << it->first << std::endl << "    ";
      if (it->second.size() == 0) {
        std::cout << "no data" << std::endl;
      } else {
        for (size_t i = 0; i < it->second.size() - 1; i++) {
          std::cout << std::setprecision(8) << it->second[i] << ", ";
        }

        std::cout << std::setprecision(8) << it->second[it->second.size() - 1] << std::endl;
      }

      it++;
    }
  }

  void write(std::string path) {
    if (curr_iter < start_iter || curr_iter >= end_iter) {
      return;
    }

    std::ofstream out_file;
    out_file.open(path + "/iter_" + std::to_string(curr_iter) + ".out");
    std::map<std::string, std::vector<float>>::iterator it = input_tensors.begin();
    while (it != input_tensors.end()) {
      out_file << "Arg Name: " << it->first << std::endl << "    ";
      if (it->second.size() == 0) {
        out_file << "no data" << std::endl;
      } else {
        for (size_t i = 0; i < it->second.size() - 1; i++) {
          out_file << std::setprecision(8) << it->second[i] << ", ";
        }

        out_file << std::setprecision(8) << it->second[it->second.size() - 1] << std::endl;
      }

      it++;
    }

    out_file.close();
  }

 private:
  bool all_nodes = true;
  bool print_node_infos = false;
  std::map<std::string, std::set<int>> enabled_nodes = {
      //{"Add_276_Grad/ReduceSum_1", {0}}
  };
  std::map<std::string, std::vector<float>> input_tensors = {};
  std::set<std::string> processed_inputs = {};
};

int InputDebugger::curr_iter = 0;
int InputDebugger::start_iter = 0;
int InputDebugger::end_iter = 1;
unsigned int InputDebugger::element_counts = 128;


static void CalculateTotalOutputSizes(OpKernelContextInternal* op_kernel_context,
                                      size_t& total_output_sizes, const std::string& node_name) {
  // Calculate total output sizes for this operation.
  total_output_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  for (auto i = 0; i < op_kernel_context->OutputCount(); i++) {
    const OrtValue* p_output = op_kernel_context->GetOutputMLValue(i);
    if (p_output != nullptr && p_output->IsTensor()) {
      const auto& tensor = p_output->Get<Tensor>();
      size_t tensor_size = tensor.SizeInBytes();
#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = tensor.Shape();
      std::cout << node_name << " output[" << i << "]"
                << " size=" << tensor_size
                << " shape=" << tensor_shape.ToString()
                << " element_size=" << tensor.DataType()->Size()
                << "\n";
#endif
      total_output_sizes += tensor_size;
    }
  }
}

static void CalculateTotalInputSizes(const OpKernelContextInternal* op_kernel_context,
                                     const onnxruntime::OpKernel* p_op_kernel,
                                     size_t& input_activation_sizes, size_t& input_parameter_sizes,
                                     const std::string& node_name) {
  // Calculate total input sizes for this operation.
  input_activation_sizes = 0;
  input_parameter_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  const int input_count = op_kernel_context->InputCount();
  for (auto i = 0; i < input_count; i++) {
    const OrtValue* p_input = op_kernel_context->GetInputMLValue(i);
    if (p_input != nullptr && p_input->IsTensor()) {
      const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
      const Tensor* p_tensor = nullptr;
      bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
      if (!is_param) {
        p_tensor = &(p_input->Get<Tensor>());
      }
      size_t tensor_size = p_tensor->SizeInBytes();

#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = p_tensor->Shape();
      size_t element_size = p_tensor->DataType()->Size();
      std::cout << node_name << " input[" << i << "]"
                << " is_param=" << is_param
                << " size=" << tensor_size
                << " shape=" << tensor_shape.ToString()
                << " element_size=" << element_size
                << "\n";
#endif
      if (is_param) {
        input_parameter_sizes += tensor_size;
      } else {
        input_activation_sizes += tensor_size;
      }
    }
  }
}

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger);

Status SequentialExecutor::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger) {
  const bool is_profiler_enabled = session_state.Profiler().IsEnabled();
  TimePoint tp;
  TimePoint sync_time_begin;
  TimePoint kernel_begin_time;
  size_t input_activation_sizes = 0;
  size_t input_parameter_sizes = 0;
  size_t total_output_sizes = 0;

  if (is_profiler_enabled) {
    tp = session_state.Profiler().StartTime();
  }

  ExecutionFrame frame{feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state};
  const std::unordered_set<NodeIndex>* to_be_executed_nodes = nullptr;

#if !defined(ORT_MINIMAL_BUILD)
  to_be_executed_nodes = session_state.GetToBeExecutedNodes(fetch_mlvalue_idxs);
  const bool only_execute_path_to_fetches = only_execute_path_to_fetches_ && (to_be_executed_nodes != nullptr);

  if (only_execute_path_to_fetches) {
    VLOGS(logger, 1) << to_be_executed_nodes->size() << " nodes to be executed\n";
  }
#else
  ORT_UNUSED_PARAMETER(only_execute_path_to_fetches_);
  const bool only_execute_path_to_fetches = false;
#endif

  LOGS(logger, INFO) << "Begin execution";
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

// Enable TRACE_EXECUTION compile flag to dump execution plan
#if defined(TRACE_EXECUTION)
  std::cout << std::make_pair(&seq_exec_plan, &session_state) << std::endl;
#endif

  const auto& graph_viewer = session_state.GetGraphViewer();

#ifdef CONCURRENCY_VISUALIZER
  // need unique name for the series. number of nodes should be good enough for a subgraph
  char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
  if (graph_viewer->IsSubgraph()) {
    auto s = graph_viewer->ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
    std::copy(s.cbegin(), s.cend(), series_name);
  }

  diagnostic::marker_series series(series_name);
#endif

#ifdef ENABLE_NVTX_PROFILE
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
  profile::NvtxRangeCreator forward_range(
      "Batch-" + tag + " Forward",
      profile::Color::White);
  profile::NvtxRangeCreator backward_range(
      "Batch-" + tag + " Backward",
      profile::Color::Black);
#endif
    if(auto env_p = std::getenv("END_ITER")){
        InputDebugger::end_iter = atoi(env_p);
        printf("setting end_iter to %d\n", InputDebugger::end_iter);
    }

    if(auto env_p = std::getenv("ELEMENT_CNT")){
        InputDebugger::element_counts = atoi(env_p);
        printf("setting element_counts to %d\n", InputDebugger::element_counts);
    }


  InputDebugger debugger = InputDebugger();
  for (const auto& node_exec_plan : exec_plan_vec) {
    if (terminate_flag_) {
      LOGS(logger, WARNING) << "Exiting due to terminate flag being set to true.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
    }

    auto node_index = node_exec_plan.node_index;

    // If it is not necessary to execute the node.
    if (only_execute_path_to_fetches && to_be_executed_nodes->count(node_index) == 0) {
      continue;
    }

    const auto& node = *graph_viewer.GetNode(node_exec_plan.node_index);

#ifdef CONCURRENCY_VISUALIZER
    series.write_flag(node.Name().c_str());
#endif

#ifdef ENABLE_NVTX_PROFILE
    if (node.Description() != "Backward pass" && !forward_range.IsBeginCalled()) {
      // Start timing forward pass when encountering the first forward node.
      forward_range.Begin();
    } else if (node.Description() == "Backward pass" && !backward_range.IsBeginCalled() && forward_range.IsBeginCalled()) {
      // Start timing backward pass when encountering the first backward node.
      // In the meanwhile, forward range ends.
      forward_range.End();
      backward_range.Begin();
    }
#endif

    auto p_op_kernel = session_state.GetKernel(node_index);

    // if a kernel has been added in the session state, it better be NON-null.
    if (p_op_kernel == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ",
                             node.Name());

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_start;
    QueryPerformanceCounter(&kernel_start);
#endif
    // construct OpKernelContext
    // TODO: log kernel inputs?
    OpKernelContextInternal op_kernel_context(session_state, frame, *p_op_kernel, logger, terminate_flag_);
    // TODO: log kernel outputs?
    if (is_profiler_enabled) {
      sync_time_begin = session_state.Profiler().StartTime();
    }

    // sync before compute
    int queue_id = p_op_kernel->KernelDef().ExecQueueId();
    if (seq_exec_plan.NodeHasFence(node_index)) {
      for (int input_index = 0; input_index < op_kernel_context.InputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.InputFence(input_index);
        if (fence) {
          auto execution_provider_type = p_op_kernel->Node().GetExecutionProviderType();
          if (OrtMemTypeCPUInput == p_op_kernel->KernelDef().InputMemoryType(input_index)) {
            execution_provider_type = kCpuExecutionProvider;
          }
          fence->BeforeUsingAsInput(execution_provider_type, queue_id);
        }
      }

      for (int input_index = 0; input_index < op_kernel_context.ImplicitInputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.ImplicitInputFence(input_index);
        if (fence) {
          auto execution_provider_type = p_op_kernel->Node().GetExecutionProviderType();
          if (OrtMemTypeCPUInput == p_op_kernel->KernelDef().InputMemoryType(input_index)) {
            execution_provider_type = kCpuExecutionProvider;
          }
          fence->BeforeUsingAsInput(execution_provider_type, queue_id);
        }
      }

      for (int output_index = 0; output_index < op_kernel_context.OutputCount(); ++output_index) {
        Fence_t fence = op_kernel_context.OutputFence(output_index);
        if (fence) {
          fence->BeforeUsingAsOutput(p_op_kernel->Node().GetExecutionProviderType(), queue_id);
        }
      }
    }
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeInputs(op_kernel_context, p_op_kernel->Node(), session_state);
#endif

    const std::string node_name_for_profiling = [&]() -> std::string {
      if (!is_profiler_enabled) return {};
      // Derive something meaningful for profile traces and logs if node name field is blank in execution graph
      return node.Name().empty() ? MakeString(node.OpType(), "_", node_index) : node.Name();
    }();

    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name_for_profiling + "_fence_before",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});

      // call compute on the kernel
      VLOGS(logger, 1) << "Computing kernel: " << node_name_for_profiling;

      kernel_begin_time = session_state.Profiler().StartTime();

      // Calculate total input sizes for this operation.
      CalculateTotalInputSizes(&op_kernel_context, p_op_kernel,
                               input_activation_sizes, input_parameter_sizes, node_name_for_profiling);
    }

#ifdef CONCURRENCY_VISUALIZER
    {
      diagnostic::span span(series, "%s.%d", node.OpType().c_str(), node.Index());
#endif
      Status compute_status;

      ORT_TRY {
        if(std::getenv("DUMP_GRAD")){
            debugger.handle_node(op_kernel_context, node, session_state);
        }
        compute_status = p_op_kernel->Compute(&op_kernel_context);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          compute_status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
        });
      }

      if (!compute_status.IsOK()) {
        std::ostringstream ss;
        ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
           << "' Status Message: " << compute_status.ErrorMessage();
        const auto msg_string = ss.str();
        LOGS(logger, ERROR) << msg_string;
        return Status(compute_status.Category(), compute_status.Code(), msg_string);
      }

#ifdef CONCURRENCY_VISUALIZER
    }
#endif

    if (is_profiler_enabled) {
      // Calculate total output sizes for this operation.
      CalculateTotalOutputSizes(&op_kernel_context, total_output_sizes, node_name_for_profiling);

#if defined(TRACE_EXECUTION)
      // Trace execution step.
      const Node& node = p_op_kernel->Node();
      std::cout << "Executed op kernel node " << node_name_for_profiling
                << " Index=" << node.Index()
                << " OpType=" << node.OpType()
                << " Name=" << node.Name()
                << " Activation_Size=" << input_activation_sizes
                << " Parameter_Size=" << input_parameter_sizes
                << " Output_Size=" << total_output_sizes
                << "\n";
#endif

      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name_for_profiling + "_kernel_time",
                                                     kernel_begin_time,
                                                     // Log additional operation args / info.
                                                     {
                                                         {"op_name", p_op_kernel->KernelDef().OpName()},
                                                         {"provider", p_op_kernel->KernelDef().Provider()},
                                                         {"graph_index", std::to_string(p_op_kernel->Node().Index())},
                                                         {"exec_plan_index", std::to_string(node_index)},
                                                         {"activation_size", std::to_string(input_activation_sizes)},
                                                         {"parameter_size", std::to_string(input_parameter_sizes)},
                                                         {"output_size", std::to_string(total_output_sizes)},
                                                     });

      sync_time_begin = session_state.Profiler().StartTime();
    }

    // sync after compute for outputs
    if (seq_exec_plan.NodeHasFence(node_index)) {
      for (int input_index = 0; input_index < op_kernel_context.InputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.InputFence(input_index);
        if (fence) {
          fence->AfterUsedAsInput(queue_id);
        }
      }

      for (int input_index = 0; input_index < op_kernel_context.ImplicitInputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.ImplicitInputFence(input_index);
        if (fence) {
          fence->AfterUsedAsInput(queue_id);
        }
      }

      for (int output_index = 0; output_index < op_kernel_context.OutputCount(); ++output_index) {
        Fence_t fence = op_kernel_context.OutputFence(output_index);
        if (fence) {
          fence->AfterUsedAsOutput(queue_id);
        }
      }
    }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_stop;
    QueryPerformanceCounter(&kernel_stop);
    LARGE_INTEGER elapsed;
    elapsed.QuadPart = kernel_stop.QuadPart - kernel_start.QuadPart;
    elapsed.QuadPart *= 1000000;
    elapsed.QuadPart /= perf_freq.QuadPart;
    // Log an event
    TraceLoggingWrite(telemetry_provider_handle,  // handle to my provider
                      "OpEnd",                    // Event Name that should uniquely identify your event.
                      TraceLoggingValue(p_op_kernel->KernelDef().OpName().c_str(), "op_name"),
                      TraceLoggingValue(elapsed.QuadPart, "time"));
#endif
    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name_for_profiling + "_fence_after",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
    }

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeOutputs(op_kernel_context, p_op_kernel->Node(), session_state);
#endif

    // free ml-values corresponding to this node
    VLOGS(logger, 1) << "Releasing node ML values.";
    ORT_RETURN_IF_ERROR(ReleaseNodeMLValues(frame, seq_exec_plan, node_exec_plan, logger));
  }
  if(std::getenv("DUMP_GRAD")){
    printf("\n\nwrite gradient to file /bert/logs\n\n");
    debugger.write("/bert/logs");

    InputDebugger::curr_iter++;
  }

#ifdef ENABLE_NVTX_PROFILE
  // Make sure forward Range object call Begin and End.
  if (!forward_range.IsBeginCalled()) {
    forward_range.Begin();
  }
  if (!forward_range.IsEndCalled()) {
    forward_range.End();
  }
  // Make sure backward Range object call Begin and End.
  if (!backward_range.IsBeginCalled()) {
    backward_range.Begin();
  }
  if (!backward_range.IsEndCalled()) {
    backward_range.End();
  }
#endif

  VLOGS(logger, 1) << "Fetching output.";
  // ExecutionFrame::Finalize will update 'fetches' with the final output
  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));
  VLOGS(logger, 1) << "Done with execution.";

  if (frame.HasMemoryPatternPlanner()) {
    std::vector<std::reference_wrapper<const TensorShape>> input_shapes;
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.Get<Tensor>();
      input_shapes.push_back(std::cref(tensor.Shape()));
    }

    if (all_tensors) {
      auto mem_patterns = onnxruntime::make_unique<MemoryPatternGroup>();
      ORT_RETURN_IF_ERROR(frame.GeneratePatterns(mem_patterns.get()));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(input_shapes, std::move(mem_patterns)));
    }
  }

  if (is_profiler_enabled) {
    session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", tp);
  }

  for (auto i : frame.GetStaticMemorySizeInfo()) {
    LOGS(logger, INFO) << "[Memory] ExecutionFrame statically allocates "
                       << i.second << " bytes for " << i.first << std::endl;
  }

  for (auto i : frame.GetDynamicMemorySizeInfo()) {
    LOGS(logger, INFO) << "[Memory] ExecutionFrame dynamically allocates "
                       << i.second << " bytes for " << i.first << std::endl;
  }

  return Status::OK();
}

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger) {
  for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
    auto ort_value_idx = seq_exec_plan.to_be_freed[i];
    VLOGS(logger, 1) << "Releasing ort_value with index: " << ort_value_idx;
    ORT_RETURN_IF_ERROR(frame.ReleaseMLValue(ort_value_idx));
  }

  return Status::OK();
}
}  // namespace onnxruntime
