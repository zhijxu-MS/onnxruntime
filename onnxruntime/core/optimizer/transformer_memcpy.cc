// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transformer_memcpy.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries);
  void BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(onnxruntime::NodeArg* arg, bool is_input);
  void ProcessInitializers(const KernelRegistryManager& kernel_registries);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransformerMemcpyImpl);

  // use value-based compare to make sure transformer output order is consistent
  struct NodeCompare {
    bool operator()(const onnxruntime::Node* lhs, const onnxruntime::Node* rhs) const {
      return lhs->Index() < rhs->Index();
    }
  };

  // use value-based compare to make sure transformer output order is consistent
  struct NodeArgCompare {
    bool operator()(const onnxruntime::NodeArg* lhs, const onnxruntime::NodeArg* rhs) const {
      return lhs->Name() < rhs->Name();
    }
  };

  std::set<onnxruntime::Node*, NodeCompare> provider_nodes_;
  std::set<const onnxruntime::NodeArg*, NodeArgCompare> non_provider_input_defs_;  // all input defs of non-provider nodes
  std::set<onnxruntime::NodeArg*, NodeArgCompare> non_provider_output_defs_;       // all output defs of non-provider nodes
  std::set<const onnxruntime::NodeArg*, NodeArgCompare> provider_input_defs_;      // all input defs of provider nodes that should be in provider allocator
  std::set<onnxruntime::NodeArg*, NodeArgCompare> provider_output_defs_;           // all output defs of provider nodes that should be in provider allocator
  std::map<const onnxruntime::NodeArg*, std::set<onnxruntime::Node*, NodeCompare>> provider_input_nodes_;
  std::map<const onnxruntime::NodeArg*, std::set<onnxruntime::Node*, NodeCompare>> provider_output_nodes_;

  onnxruntime::Graph& graph_;
  std::string provider_;
};

// very simple GraphTransformer that uses TransformerMemcpyImpl for each graph
// and mainly provides the subgraph recursion functionality
common::Status MemcpyTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  for (auto& provider : provider_types_) {
    if (provider != onnxruntime::kCpuExecutionProvider &&
        provider != onnxruntime::kMklDnnExecutionProvider &&
        provider != onnxruntime::kNupharExecutionProvider &&
        provider != onnxruntime::kTensorrtExecutionProvider) {
      TransformerMemcpyImpl copy_impl(graph, provider);
      modified = copy_impl.ModifyGraph(registry_manager_);
    }
  }

  // TODO: We probably need to do the recursion inline when processing the main graph in order to maximize efficiency.
  // e.g. data on GPU prior to an 'If' node. The 'If' must run on CPU, but if the subgraph is GPU based it could
  // consume the data from GPU and we shouldn't insert a memcpy from GPU to CPU prior to the If node, and one from
  // CPU back to GPU when beginning execution of the subgraph. To do that requires inspecting the subgraph (and any
  // nested subgraphs) when deciding whether to insert a memcpy in the parent graph, and may need to be fully done
  // within TransformerMemcpyImpl instead of via this simple GraphTransformer.

  // handle any subgraphs in nodes
  for (auto& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));
  }

  return Status::OK();
}

/*

Overview: The transformer transforms the input graph as follows:

(1) For every initializer W that is referenced by both provider and non-provider nodes,
we create a duplicate initializer W2 and change all provider nodes to reference this
duplicate copy.

(2) For every ml-value X that is computed by a provider node and referenced by a
non-provider node, we introduce a new ml-value X2. We replace all references to X
in provider nodes by X2, and introduce a copy from X2 to X. (All graph outputs
are considered as non-provider references here.)

(3 For every ml-value X that is computed by a non-provider node and referenced by
a provider node, we introduce a new ml-value X2. We replace all references to X in
provider nodes by X2, and introduce a copy from X to X2. (All graph inputs are
considered to be non-provider here.)

Note that every ml-value is computed at a unique point (either provider or non-provider),
but it may be referenced and used at multiple points (by both provider and non-provider).

This transformer does not currently optimize copies between, e.g., two different GPU devices, etc.

*/

bool TransformerMemcpyImpl::ModifyGraph(const KernelRegistryManager& kernel_registries) {
  bool modified = false;
  // find defs that require copy
  for (auto& node : graph_.Nodes()) {
    //don't need to do node placement here now, onnxruntime will do it according to registered kernels.
    ProcessDefs(node, kernel_registries);
  }

  // for initializers shared by different providers, create dups
  ProcessInitializers(kernel_registries);

  for (auto arg : graph_.GetInputs())
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : non_provider_input_defs_)
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : non_provider_output_defs_)
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : graph_.GetInputs())
    // For inputs we need to create a copy node only when the input is connected to both provider
    // and non-provider nodes. Otherwise utils::CopyInputsAcrossDevices() will do the job.
    if (provider_input_defs_.count(arg) && non_provider_input_defs_.count(arg)) {
      AddCopyNode(const_cast<onnxruntime::NodeArg*>(arg), true);
      modified = true;
    }

  for (auto arg : non_provider_output_defs_)
    if (provider_input_defs_.count(arg)) {
      AddCopyNode(arg, true);
      modified = true;
    }

  for (auto arg : provider_output_defs_)
    if (non_provider_input_defs_.count(arg)) {
      AddCopyNode(arg, false);
      modified = true;
    }

  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries) {
  if (node.GetExecutionProviderType() == provider_) {
    provider_nodes_.insert(&node);
    // note KernelCreateInfo might be nullptr for custom kernel
    const KernelCreateInfo* kci = nullptr;
    kernel_registries.SearchKernelRegistry(node, &kci);

    ORT_ENFORCE(onnxruntime::Node::ForEachWithIndex(
                    node.InputDefs(),
                    [this, &kci](const onnxruntime::NodeArg& arg, size_t index) {
                      if (kci && MemTypeOnCpuExplicitly(kci->kernel_def->InputMemoryType(index)))
                        non_provider_input_defs_.insert(&arg);
                      else
                        provider_input_defs_.insert(&arg);
                      return Status::OK();
                    })
                    .IsOK());

    // we don't need to handle implicit input here as provider_ is never kCpuExecutionProvider, all control flow
    // nodes are CPU based, and only control flow nodes have implicit inputs.

    auto& output_defs = node.MutableOutputDefs();
    for (size_t i = 0; i < output_defs.size(); ++i) {
      auto arg = output_defs[i];
      if (!arg->Exists())
        continue;

      if (kci && MemTypeOnCpuExplicitly(kci->kernel_def->OutputMemoryType(i)))
        non_provider_output_defs_.insert(arg);
      else
        provider_output_defs_.insert(arg);
    }
  } else {
    // TODO: copy between devices? i.e. multiple GPUs
    if (node.GetExecutionProviderType() != onnxruntime::kCpuExecutionProvider && node.GetExecutionProviderType() != onnxruntime::kTensorrtExecutionProvider &&
        !node.GetExecutionProviderType().empty()) {
      ORT_THROW("Execution type '", node.GetExecutionProviderType(), "' doesn't support memcpy ");
    }

    for (const auto* arg : node.InputDefs()) {
      if (arg->Exists())
        non_provider_input_defs_.insert(arg);
    }

    for (const auto* arg : node.ImplicitInputDefs()) {
      if (arg->Exists())
        non_provider_input_defs_.insert(arg);
    }

    for (auto* arg : node.MutableOutputDefs()) {
      if (arg->Exists())
        non_provider_output_defs_.insert(arg);
    }
  }
}

//for non_provider defs, collect the nodes that expect it is provider tensor as input/output.
void TransformerMemcpyImpl::BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries) {
  for (auto it = graph_.Nodes().begin(); it != graph_.Nodes().end(); it++) {
    if (it->OpType() == "MemcpyFromHost" || it->OpType() == "MemcpyToHost")
      continue;
    auto input_it = std::find(it->MutableInputDefs().begin(), it->MutableInputDefs().end(), const_cast<onnxruntime::NodeArg*>(arg));
    auto output_it = std::find(it->MutableOutputDefs().begin(), it->MutableOutputDefs().end(), const_cast<onnxruntime::NodeArg*>(arg));
    int arg_input_index = input_it != it->MutableInputDefs().end() ? static_cast<int>(input_it - it->MutableInputDefs().begin()) : -1;
    int arg_output_index = output_it != it->MutableOutputDefs().end() ? static_cast<int>(output_it - it->MutableOutputDefs().begin()) : -1;
    if (arg_input_index == -1 && arg_output_index == -1)
      continue;
    if (it->GetExecutionProviderType() == provider_) {
      const KernelCreateInfo* kci = nullptr;
      kernel_registries.SearchKernelRegistry(*it, &kci);
      if (arg_input_index != -1) {
        if (!kci || !MemTypeOnCpuExplicitly(kci->kernel_def->InputMemoryType(arg_input_index)))
          provider_input_nodes_[arg].insert(&*it);
      }
      if (arg_output_index != -1) {
        if (!kci || !MemTypeOnCpuExplicitly(kci->kernel_def->OutputMemoryType(arg_output_index)))
          provider_output_nodes_[arg].insert(&*it);
      }
    }
  }
}

void TransformerMemcpyImpl::AddCopyNode(onnxruntime::NodeArg* arg, bool is_input) {
  // create unique name for new def
  std::string new_def_name = graph_.GenerateNodeArgName(arg->Name() + "_" + provider_);

  auto* new_arg = &graph_.GetOrCreateNodeArg(new_def_name, arg->TypeAsProto());
  auto* src_arg = is_input ? arg : new_arg;
  auto* dst_arg = is_input ? new_arg : arg;

  // create unique name for copy node
  std::string new_node_name = graph_.GenerateNodeName("Memcpy");

  const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
  auto& new_node = graph_.AddNode(new_node_name, op_name, "Copy from/to host memory",
                                  std::vector<onnxruntime::NodeArg*>{src_arg},
                                  std::vector<onnxruntime::NodeArg*>{dst_arg});
  new_node.SetExecutionProviderType(provider_);
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> map = {{arg, new_arg}};
  auto it = provider_input_nodes_.find(arg);
  if (it != provider_input_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }
  it = provider_output_nodes_.find(arg);
  if (it != provider_output_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }
}

template <typename NodeArgSetType>
static const onnxruntime::NodeArg* FindNodeArg(const NodeArgSetType& def_set, const std::string& name) {
  NodeArg def(name, nullptr);
  auto it = def_set.find(&def);  // this works since we use name to compare NodeArg
  if (it != def_set.end())
    return *it;
  return nullptr;
}

// We duplicate any initializer that is used by both provider nodes and non-provider nodes
// to ensure that provider nodes and non-provider nodes don't share initializers, as they
// need to stay in different memory locations.
void TransformerMemcpyImpl::ProcessInitializers(const KernelRegistryManager& kernel_registries) {
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacements;
  for (const auto& pair : graph_.GetAllInitializedTensors()) {
    const auto& name = pair.first;
    const onnxruntime::NodeArg* provider_def = FindNodeArg(provider_input_defs_, name);
    const onnxruntime::NodeArg* non_provider_def = FindNodeArg(non_provider_input_defs_, name);
    if (provider_def != nullptr && non_provider_def != nullptr) {
      std::string new_def_name = graph_.GenerateNodeArgName(name);
      auto& new_def = graph_.GetOrCreateNodeArg(new_def_name, provider_def->TypeAsProto());

      const TensorProto* tensor_proto = nullptr;
      bool found = graph_.GetInitializedTensor(name, tensor_proto);
      ORT_ENFORCE(found, "Failed to get initialized tensor ", name);

      TensorProto new_tensor_proto = *tensor_proto;
      *(new_tensor_proto.mutable_name()) = new_def_name;
      graph_.AddInitializedTensor(new_tensor_proto);

      replacements.insert(std::make_pair(provider_def, &new_def));
    }
  }

  for (auto p_node : provider_nodes_) {
    // make a copy of replacement map as the node may exclude mapping for InputDefs with MemTypeOnCpuExplicitly
    auto dup_replacements = replacements;

    const KernelCreateInfo* kci = nullptr;
    kernel_registries.SearchKernelRegistry(*p_node, &kci);
    p_node->ForEachWithIndex(
        p_node->InputDefs(),
        [kci, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
          if (kci && MemTypeOnCpuExplicitly(kci->kernel_def->InputMemoryType(index)))
            dup_replacements.erase(&arg);
          return Status::OK();
        });

    // normally initializers are only inputs, but things may change with ops like assign
    p_node->ForEachWithIndex(
        p_node->OutputDefs(),
        [kci, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
          if (kci && MemTypeOnCpuExplicitly(kci->kernel_def->OutputMemoryType(index))) {
            ORT_ENFORCE(dup_replacements.find(&arg) == dup_replacements.end());
          }
          return Status::OK();
        });

    p_node->ReplaceDefs(dup_replacements);
  }
}

}  // namespace onnxruntime
