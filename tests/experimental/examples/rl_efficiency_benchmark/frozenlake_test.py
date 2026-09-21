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

"""CPU-only benchmark regression tests; runnable without TPU/JAX/pytest.

Run directly:
python3 tests/experimental/examples/rl_efficiency_benchmark/frozenlake_test.py
"""

import argparse
import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


def load_module(name):
  path = (
      Path(__file__).resolve().parents[4]
      / "tunix/experimental/examples/rl_efficiency_benchmark"
      / (name + ".py")
  )
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


benchmark = load_module("frozenlake")
runtime = load_module("runtime")


class BenchmarkTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.path = Path(self.temp.name)
    self.w = dict(
        batch=8,
        generations=8,
        steps=3,
        warmup=1,
        micro_groups=2,
        prompt=2048,
        response=2048,
        turns=8,
        seed=42,
        dataset_size=10000,
        concurrency=64,
        max_num_seqs=32,
        batched_tokens=8192,
    )

  def test_microbatches_represent_equal_trajectory_counts(self):
    config = benchmark.agentic_config(self.w, self.path, self.path)
    env = benchmark.dist_environment(self.w, self.path, self.path)
    train = config["rl_training_config"]
    for key in ("train_micro_batch_size", "compute_logps_micro_batch_size"):
      self.assertEqual(
          int(env[key.upper()]), train[key] * self.w["generations"]
      )
    self.assertEqual(config["rollout_model_config"]["mesh"]["shape"], "(1,2)")
    self.assertIsNone(config["rollout_model_config"]["same_mesh_as"])
    self.assertEqual(env["CHECKPOINT_SAVE_INTERVAL_STEPS"], "0")
    self.assertIsNone(train["checkpoint_root_directory"])
    self.assertIsNone(train["metrics_logging_options"])
    self.assertIsNone(train["profiler_options"])
    self.assertIsNone(train["actor_optimizer_config"]["schedule_type"])

  def record_run(self):
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    # Initial sync must not count as a training full batch.
    recorder.finish("weight_sync", 0, 2)
    for step, (start, end) in enumerate(((2, 12), (12, 32), (32, 62))):
      for trajectory in range(64):
        rollout_start = start + 1 + trajectory / 1000
        rollout_end = end - 2 + trajectory / 1000
        recorder.emit(
            "generation",
            step=step,
            start=rollout_start,
            end=rollout_start + 1,
            seconds=1,
            ok=True,
            group_id=trajectory // 8,
            pair_index=trajectory % 8,
            prompt_tokens=10,
            generated_tokens=20,
        )
        recorder.emit(
            "rollout_trajectory",
            step=step,
            start=rollout_start,
            end=rollout_end,
            seconds=rollout_end - rollout_start,
            ok=True,
            group_id=trajectory // 8,
            pair_index=trajectory % 8,
            prompt_tokens=10,
            generated_tokens=20,
            environment_tokens=5,
            conversation_tokens=25,
            environment_seconds=0.25,
            reward_seconds=0.01,
            reward=1.0,
            turns=2,
            status="SUCCEEDED",
        )
      shapes = [{"prompt": [64, 10], "completion": [64, 20]}]
      recorder.emit(
          "metric", name="trainer/loss", value=step + 1, step=step, time=start + 1
      )
      recorder.finish(
          "actor_log_probs", start, start + 1, shapes=shapes, ok=True
      )
      recorder.finish(
          "training",
          start,
          end - 2,
          trajectories=64,
          shapes=shapes,
      )
      recorder.finish("weight_sync", end - 2, end)
    recorder.close()
    benchmark.write_json(
        self.path / "manifest.json",
        {"mode": "dist", "workload": self.w, "chips": 4},
    )
    benchmark.write_json(
        self.path / "result.json",
        {"returncode": 0, "total_run_time_seconds": 70},
    )

  def test_sync_inclusive_step_and_weighted_throughput(self):
    self.record_run()
    result = benchmark.summarize(self.path)
    self.assertEqual(
        result["end_to_end"]["steady_state_step_time_seconds"]["per_step"],
        [20, 30],
    )
    self.assertEqual(
        result["end_to_end"]["steady_state_step_time_seconds"]["median"], 25
    )
    self.assertEqual(
        result["end_to_end"]["trajectories_per_second"], 128 / 50
    )
    self.assertEqual(
        result["end_to_end"]["tpu_chip_seconds_per_trajectory"],
        4 * 50 / 128,
    )
    self.assertEqual(
        result["api_calls"]["weight_sync"]["host_time_per_step_seconds"], 2
    )
    self.assertAlmostEqual(
        result["rollout"]["output_tokens_per_second"], 128 * 20 / 44.126
    )
    self.assertEqual(
        result["rollout"]["output_tokens_per_trajectory"]["mean"], 20
    )
    self.assertEqual(result["rollout"]["average_turns_per_trajectory"], 2)
    self.assertEqual(result["rollout"]["average_reward_per_trajectory"], 1)
    self.assertEqual(
        set(result["rollout"]),
        {
            "collection_time_seconds",
            "trajectories_per_second",
            "output_tokens_per_second",
            "trajectory_latency_seconds",
            "model_call_latency_seconds",
            "output_tokens_per_trajectory",
            "average_prompt_tokens_per_model_call",
            "average_output_tokens_per_model_call",
            "average_turns_per_trajectory",
            "average_reward_per_trajectory",
            "average_time_per_trajectory_seconds",
            "trajectory_outcome_percent",
        },
    )
    self.assertEqual(
        result["pipeline"]["rollout_start_delay_seconds"]["median"], 1
    )
    self.assertEqual(result["training_input"]["trajectories_per_call"], 64)
    self.assertEqual(
        result["training_input"]["padded_token_slots_per_step"], 64 * 30
    )
    self.assertEqual(
        result["actor_log_probs_input"]["padded_token_slots_per_step"],
        64 * 30,
    )
    self.assertEqual(result["training_metrics"]["trainer/loss"], 2.5)

    agentic = json.loads(json.dumps(result))
    agentic["mode"] = "agentic"
    result["api_calls"]["worker_rpc_fwd_bwd"] = {
        "calls_per_step": 1,
        "host_time_per_step_seconds": 2,
        "median_call_time_milliseconds": 2000,
        "p90_call_time_milliseconds": 2000,
    }
    comparison = benchmark.compare_summaries([agentic, result])
    step_gap = comparison["metrics"]["end_to_end_step_time_seconds"]
    self.assertEqual(step_gap["agentic"], 25)
    self.assertEqual(step_gap["dist"], 25)
    self.assertEqual(step_gap["dist_relative_to_agentic_percent"], 0)
    self.assertEqual(step_gap["interpretation"], "same")
    self.assertTrue(
        comparison["parity_checks"]["training_tensor_shapes_match"]
    )
    rpc_gap = comparison["api_calls"]["worker_rpc_fwd_bwd"]
    self.assertEqual(
        rpc_gap["host_time_per_step_seconds"]["interpretation"],
        "one_stack_only",
    )

  def test_incomplete_and_failed_runs_rejected(self):
    self.record_run()
    manifest = json.loads((self.path / "manifest.json").read_text())
    manifest["workload"]["steps"] = 4
    benchmark.write_json(self.path / "manifest.json", manifest)
    with self.assertRaisesRegex(ValueError, "Incomplete"):
      benchmark.summarize(self.path)
    benchmark.write_json(
        self.path / "result.json",
        {"returncode": 124, "total_run_time_seconds": 70},
    )
    with self.assertRaisesRegex(ValueError, "Failed"):
      benchmark.summarize(self.path)

  def test_training_trajectory_mismatch_rejected(self):
    self.record_run()
    manifest = json.loads((self.path / "manifest.json").read_text())
    manifest["workload"]["generations"] = 2
    benchmark.write_json(self.path / "manifest.json", manifest)
    with self.assertRaisesRegex(ValueError, "work mismatch"):
      benchmark.summarize(self.path)

  def test_failed_sync_is_not_a_completed_batch(self):
    recorder = runtime.Recorder(self.path / "events.jsonl", clock=lambda: 0)
    recorder.finish("training", 0, 1, trajectories=64)
    recorder.finish("weight_sync", 1, 2, ok=False)
    recorder.close()
    events = [
        json.loads(line)
        for line in (self.path / "events.jsonl").read_text().splitlines()
    ]
    self.assertFalse(any(e["kind"] == "training_step" for e in events))

  def test_wrappers_preserve_values_exceptions_and_async_behavior(self):
    class Target:

      def run(self, value):
        return value + 1

      def remote(self, method_name):
        return method_name

      async def fail(self):
        raise ValueError("original exception")

    recorder = runtime.Recorder(self.path / "events.jsonl")
    recorder.wrap(Target, "run", "training", lambda args, kwargs: args[1])
    recorder.wrap(
        Target,
        "remote",
        lambda args, kwargs: "worker_rpc_" + args[1],
    )
    recorder.wrap(Target, "fail", "actor_log_probs")
    self.assertEqual(Target().run(4), 5)
    self.assertEqual(Target().remote("fwd_bwd"), "fwd_bwd")
    with self.assertRaisesRegex(ValueError, "original exception"):
      asyncio.run(Target().fail())
    recorder.close()
    events = [
        json.loads(line)
        for line in (self.path / "events.jsonl").read_text().splitlines()
    ]
    self.assertEqual(events[0]["trained_trajectories"], 4)
    self.assertEqual(events[1]["name"], "worker_rpc_fwd_bwd")
    self.assertFalse(events[2]["ok"])

  def test_rollout_token_fields_use_actual_outputs(self):
    output = types.SimpleNamespace(
        tokens=[[1, 2, 3], [4]],
        prompt_lengths=[5, 7],
        left_padded_prompt_tokens=[[0] * 99],
    )
    self.assertEqual(runtime._generation_counts(output), (12, 4))
    engine = types.SimpleNamespace(
        env=types.SimpleNamespace(
            task={},
            extra_kwargs={
                "benchmark_policy_version": 3,
                "group_id": "prompt-1",
                "pair_index": 2,
            },
        ),
        agent=types.SimpleNamespace(
            trajectory=types.SimpleNamespace(steps=[1, 2])
        ),
    )
    fields = runtime._trajectory_fields(
        engine,
        {
            "prompt_tokens": [1, 2, 3],
            "conversation_tokens": [4, 5, 6, 7],
            "conversation_masks": [1, 1, 0, 0],
            "env_time": {"reset": 0.1, "steps": [0.2, 0.3]},
            "reward_time": {"reward": 0.05},
            "trajectory_reward": 1.0,
            "status": "SUCCEEDED",
        },
    )
    self.assertEqual(fields["step"], 3)
    self.assertEqual(fields["prompt_tokens"], 3)
    self.assertEqual(fields["generated_tokens"], 2)
    self.assertEqual(fields["environment_tokens"], 2)
    self.assertAlmostEqual(fields["environment_seconds"], 0.6)
    self.assertEqual(fields["reward"], 1.0)
    self.assertEqual(fields["turns"], 2)

  def test_prepare_and_report_reject_mismatched_data(self):
    model = self.path / "model"
    model.mkdir()
    (model / "weights.safetensors").touch()
    (model / "config.json").write_text("{}")
    runs = []
    for mode in ("dist", "agentic"):
      output = self.path / mode
      args = argparse.Namespace(
          **self.w,
          mode=mode,
          output=output,
          model_dir=model,
          cache_root=self.path / "cache",
          hardware="v5p-4",
          execute=False,
          timeout=60
      )
      with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
          benchmark.subprocess, "Popen", wraps=benchmark.subprocess.Popen
      ) as popen:
        benchmark.prepare(args)
        # Provenance collection can run git/platform commands; planning must
        # never start the training launcher or agentic entry point.
        for call in popen.call_args_list:
          command = str(call.args[0])
          self.assertNotIn("launcher.sh", command)
          self.assertNotIn("frozenlake_agentic", command)
      manifest = json.loads((output / "manifest.json").read_text())
      self.assertEqual(manifest["environment"]["WANDB_MODE"], "disabled")
      benchmark.write_json(
          output / "dataset.json", {"sha256": mode, "rows": 24}
      )
      runs.append(output)
    with self.assertRaisesRegex(ValueError, "Dataset"):
      benchmark.report(runs)
    # Preparation must not overwrite a prior run directory.
    with self.assertRaises(FileExistsError):
      benchmark.prepare(args)

  def test_dataset_hash_preserves_order(self):
    rows = [{"seed": 1}, {"seed": 2}]
    digests = []
    for idx, ordered in enumerate((rows, list(reversed(rows)))):
      output = self.path / str(idx)
      output.mkdir()
      with mock.patch.dict(
          runtime.os.environ, {"TUNIX_BENCHMARK_DIR": str(output)}
      ):
        runtime.record_dataset(ordered)
      digests.append(
          json.loads((output / "dataset.json").read_text())["sha256"]
      )
    self.assertNotEqual(*digests)


if __name__ == "__main__":
  unittest.main()
