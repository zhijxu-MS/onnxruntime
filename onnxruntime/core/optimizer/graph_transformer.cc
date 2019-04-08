// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformer::Apply(Graph& graph, bool& modified) const {
  // the Graph should be in a good state prior this being called, so there should be no need to call Resolve here
  // ORT_RETURN_IF_ERROR(graph.Resolve());

  auto status = ApplyImpl(graph, modified, 0);
  ORT_RETURN_IF_ERROR(status);

  // At least currently, some transformers (InsertCastTransformer and MemcpyTransformer) need this to be called
  // after they complete to put the graph back into a valid state for the next transformer.
  if (modified) {
    status = graph.Resolve();
  }

  return status;
}

}  // namespace onnxruntime
