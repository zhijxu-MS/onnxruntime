// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

namespace transformer_utils {

/** Generates all predefined rules for this level.
   If rules_to_enable is not empty, it returns the intersection of predefined rules and rules_to_enable.
   TODO: This is visible for testing at the moment, but we should rather make it private. */
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(TransformerLevel level,
                                                               const std::vector<std::string>* rules_to_enable = nullptr);

/** Generates all predefined (both rule-based and non-rule-based) transformers for this level.
    If transformers_and_rules_to_enable is not empty, it returns the intersection between the predefined transformers/rules 
	and the transformers_and_rules_to_enable. */
std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(TransformerLevel level,
                                                         std::vector<std::string>* rules_and_transformers_to_enable = nullptr);

/** Given a TransformerLevel, this method generates a name for the rule-based graph transformer of that level. */
std::string GenerateRuleBasedTransformerName(TransformerLevel level);

}  // namespace transformer_utils
}  // namespace onnxruntime
