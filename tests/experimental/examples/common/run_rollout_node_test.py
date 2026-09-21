# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for distributed rollout model-family configuration."""

from unittest import mock

from absl.testing import absltest
from tunix.experimental.examples.common import run_rollout_node
from tunix.rl.agentic.parser.chat_template_parser import parser


class RunRolloutNodeTest(absltest.TestCase):

  def test_gemma4_uses_gemma4_chat_parser(self):
    parser_factory = mock.Mock(return_value=mock.sentinel.chat_parser)
    with mock.patch.dict(
        run_rollout_node.CHAT_PARSERS,
        {"gemma-4": parser_factory},
        clear=True,
    ):
      result = run_rollout_node._chat_parser_for(
          "google/gemma-4-E2B-it",
          mock.sentinel.tokenizer,
          enable_thinking=False,
      )

    self.assertIs(result, mock.sentinel.chat_parser)
    parser_factory.assert_called_once_with(
        mock.sentinel.tokenizer, enable_thinking=False
    )

  def test_gemma4_mapping_includes_preprocessing_and_tp_hook(self):
    config = run_rollout_node._mapping_config_for("google/gemma-4-E2B-it")

    self.assertIsNotNone(config.preprocess_src_state)
    self.assertIn(
        "layers.*.mlp.gate_up_proj.kernel", config.to_hf_hook_fns
    )
    self.assertIn("layers.*.attn.v_einsum.w", config.to_hf_mappings)

  def test_gemma4_vllm_overrides_match_text_only_reference(self):
    overrides = run_rollout_node._gemma4_vllm_overrides(
        "google/gemma-4-E2B-it"
    )

    self.assertEqual(overrides["architectures"], ["Gemma4ForCausalLM"])
    self.assertEqual(overrides["final_logit_softcapping"], 30.0)
    self.assertEqual(
        overrides["text_config"]["final_logit_softcapping"], 30.0
    )

  def test_qwen_does_not_get_gemma4_overrides(self):
    self.assertIsNone(
        run_rollout_node._gemma4_vllm_overrides("Qwen/Qwen3-8B")
    )

  def test_registered_parser_is_gemma4_specific(self):
    self.assertIs(
        run_rollout_node.CHAT_PARSERS["gemma-4"],
        parser.Gemma4ChatTemplateParser,
    )


if __name__ == "__main__":
  absltest.main()
