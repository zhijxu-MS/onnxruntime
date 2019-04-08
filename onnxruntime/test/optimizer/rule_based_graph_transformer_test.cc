// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "dummy_graph_transformer.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(RuleBasedGraphTransformerTest, TestCompatibleProviders) {
  string model_uri = "testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  
  // Create rule based transformer with a dummy rewrite rule and register it with Cuda as compatible provider
  std::unordered_set<std::string> compatible_provider{onnxruntime::kCudaExecutionProvider};
  auto dummy_rule = std::make_unique<DummyRewriteRule>("DummyRule");
  const auto* dummy_rule_ptr = dummy_rule.get();

  auto graph_transformer = std::make_unique<RuleBasedGraphTransformer>("CUDATopDownTransformer", "Registered for CUDA",
                                                                       compatible_provider);
  graph_transformer->Register(std::move(dummy_rule));  

  // Create rule based transformer with a dummy rewrite rule and register it with CPU as compatible provider
  auto dummy_rule1 = std::make_unique<DummyRewriteRule>("DummyRule1");
  const auto* dummy_rule1_ptr = dummy_rule1.get();

  auto graph_transformer1 = std::make_unique<RuleBasedGraphTransformer>("CPUTopDownTransformer", "Registered for CPU");

  graph_transformer1->Register(std::move(dummy_rule1));  

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(graph_transformer), TransformerLevel::Level2);
  graph_transformation_mgr.Register(std::move(graph_transformer1), TransformerLevel::Level2);

  graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2);

  // Validate transformer registered with CUDA as compatible provider is not called.
  ASSERT_FALSE(dummy_rule_ptr->IsRewriteRuleInvoked());

  // Validate transformer registered as global is called.
  ASSERT_TRUE(dummy_rule1_ptr->IsRewriteRuleInvoked());
}

}  // namespace test
}  // namespace onnxruntime
