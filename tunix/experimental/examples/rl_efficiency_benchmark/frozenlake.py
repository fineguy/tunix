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

"""Prepare, run, and analyze distributed-vs-agentic RL benchmarks.

Planning and reporting need only the Python standard library. Training uses
the repository's normal TPU dependencies. The first workload is FrozenLake,
but the recorded step, rollout, generation, and token metrics are not specific
to single-turn or multi-turn agents. See README.md for methodology.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_MODULE = "tunix.experimental.examples.rl_efficiency_benchmark"
MODEL_ID = "google/gemma-4-E2B-it"


def write_json(path, value):
  Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def agentic_config(workload, output, model_dir):
  """Build explicit overrides of base_agentic_config, in JSON (valid YAML)."""
  w = workload
  mesh = {"shape": "(2,1)", "axis_names": "('fsdp','tp')"}
  return {
      "training_mode": "agentic_grpo",
      "model_config": {
          "model_name": "gemma-4-e2b",
          "model_id": MODEL_ID,
          # AutoModel's GCS source also accepts an existing local safetensors path.
          "model_source": "gcs",
          "model_path": str(model_dir),
          "model_download_path": str(model_dir),
          "mesh": mesh,
          "remat_config": "DECODER",
          "dtype": "bfloat16",
          "rng_seed": w["seed"],
          "use_flash_attention": True,
          "flash_attention_block_size": 256,
          "use_sliding_window_kv_cache": False,
      },
      "actor_model_config": {"load_dtype": "float32", "mesh": mesh},
      "reference_model_config": {"mesh": None, "same_mesh_as": "actor"},
      "rollout_model_config": {
          "mesh": {"shape": "(1,2)", "axis_names": "('fsdp','tp')"},
          "same_mesh_as": None,
      },
      "tokenizer_config": {
          "tokenizer_type": "huggingface",
          "tokenizer_path": str(model_dir),
          "add_bos": True,
      },
      "agent_class_path": "examples.frozenlake.agent.FrozenLakeAgent",
      "agent_kwargs": {"use_multistep_prompt": True},
      "env_class_path": "examples.frozenlake.env.FrozenLakeEnv",
      "env_kwargs": {"max_steps": w["turns"], "is_slippery": False},
      "data_module": BENCHMARK_MODULE + ".frozenlake_agentic",
      "data_config": {
          "size": w["dataset_size"],
          "seed": w["seed"],
          "limit": w["steps"] * w["batch"],
      },
      "apply_chat_template_to_dataset": False,
      "chat_parser_config": {"type": "gemma4", "enable_thinking": False},
      "batch_size": w["batch"],
      "num_batches": w["steps"],
      "num_train_epochs": 1,
      "train_fraction": 1.0,
      "rl_training_config": {
          "mini_batch_size": w["batch"],
          "train_micro_batch_size": w["micro_groups"],
          "compute_logps_micro_batch_size": w["micro_groups"],
          "compute_logps_chunk_size": 2048,
          "max_steps": w["steps"],
          "eval_every_n_steps": w["steps"] + 1,
          "checkpoint_root_directory": None,
          "checkpointing_options": None,
          "profiler_options": None,
          # Backend logging also enables trajectory logging in agentic.
          # Keep scalar monitoring (events.jsonl), but disable both backends.
          "metrics_logging_options": None,
          "actor_optimizer_config": {
              "opt_type": "adamw",
              "learning_rate": 1e-6,
              "schedule_type": None,
              "b1": 0.9,
              "b2": 0.95,
              "weight_decay": 0.0,
              "opt_chain_type": "clip_by_global_norm",
              "chain_kwargs": {"max_norm": 100.0},
          },
      },
      "rollout_engine": "vllm",
      "offload_to_cpu": False,
      "rollout_config": {
          "total_generation_steps": w["response"],
          "max_prompt_length": w["prompt"],
          "kv_cache_size": w["prompt"] + w["response"] + 256,
          "temperature": 0.7,
          "top_p": 1.0,
          "top_k": 0,
          "return_logprobs": True,
          "rollout_vllm_init_with_random_weights": True,
          "rollout_vllm_sampling_kwargs": {"skip_special_tokens": False},
      },
      "vllm_config": {
          "model_version": str(model_dir),
          "hbm_utilization": 0.2,
          "tpu_backend_type": "jax",
          "server_mode": True,
          "async_scheduling": False,
          "max_num_seqs": w["max_num_seqs"],
          "max_num_batched_tokens": w["batched_tokens"],
          "data_parallel_size": 1,
          "tensor_parallel_size": 2,
          "kwargs": {
              "kv_cache_metrics": True,
              "disable_log_stats": False,
              "enable_prefix_caching": False,
              "dtype": "bfloat16",
              "hf_overrides": {
                  "final_logit_softcapping": 30.0,
                  "text_config": {"final_logit_softcapping": 30.0},
                  "architectures": ["Gemma4ForCausalLM"],
              },
          },
      },
      "reward_functions": [],
      "agentic_grpo_config": {
          "num_generations": w["generations"],
          "num_iterations": 1,
          "beta": 0.0,
          "epsilon": 0.003,
          "epsilon_high": 0.005,
          "loss_algo": "gspo-token",
          "loss_agg_mode": "sequence-mean-token-mean",
          "kl_loss_mode": "low_var_kl",
          "advantage_estimator": "rloo",
          "sampler_is": "token",
          "sampler_is_threshold": 2.0,
          "max_concurrency": w["concurrency"],
          "off_policy_steps": 0,
          "episode_timeout": 600,
          "overlong_filter": False,
      },
  }


def dist_environment(w, output, model_dir):
  return {
      key: str(value)
      for key, value in {
          "MODEL_NAME": "gemma-4-e2b",
          "MODEL_ID": MODEL_ID,
          "MODEL_DIR": model_dir,
          "TOKENIZER_PATH": model_dir,
          "ARTIFACT_ROOT": output / "artifacts",
          "LOG_ROOT": output,
          "BATCH_SIZE": w["batch"],
          "MINI_BATCH_SIZE": w["batch"],
          "NUM_GENERATIONS": w["generations"],
          "NUM_BATCHES": w["steps"],
          "NUM_EPOCHS": 1,
          "NUM_ITERATIONS": 1,
          "MAX_STEPS": w["steps"],
          "DATASET_SIZE": w["dataset_size"],
          "SEED": w["seed"],
          "SHUFFLE": 1,
          "MAX_TURNS": w["turns"],
          "MAX_PROMPT_LENGTH": w["prompt"],
          "MAX_RESPONSE_LENGTH": w["response"],
          "VLLM_MAX_MODEL_LEN": w["prompt"] + w["response"] + 256,
          # Agentic micro-batches count prompt groups; dist counts trajectories.
          "TRAIN_MICRO_BATCH_SIZE": w["micro_groups"] * w["generations"],
          "COMPUTE_LOGPS_MICRO_BATCH_SIZE": (
              w["micro_groups"] * w["generations"]
          ),
          "COMPUTE_LOGPS_CHUNK_SIZE": 2048,
          "ROLLOUT_MAX_CONCURRENCY": w["concurrency"],
          "VLLM_MAX_NUM_SEQS": w["max_num_seqs"],
          "VLLM_MAX_NUM_BATCHED_TOKENS": w["batched_tokens"],
          "VLLM_HBM_UTILIZATION": 0.2,
          "TRAINER_TPU_CHIPS": "0,1",
          "TRAINER_FSDP": 2,
          "TRAINER_TP": 1,
          "ROLLOUT_TPU_CHIPS": "2,3",
          "ROLLOUT_FSDP": 1,
          "ROLLOUT_TP": 2,
          "TPU_CHIPS_PER_HOST_BOUNDS": "1,2,1",
          "TPU_HOST_BOUNDS": "1,1,1",
          "WEIGHT_SYNC_MODE": "raiden",
          "OFF_POLICY_STEPS": 0,
          "SAMPLER": "inprocess_vllm",
          "CHECKPOINT_SAVE_INTERVAL_STEPS": 0,
          "CHECKPOINT_ROOT_DIRECTORY": output / "checkpoints",
          "LOG_DIR": "",
          "TRAJECTORY_LOG_DIR": "",
          "IS_SLIPPERY": 0,
          "USE_MULTISTEP_PROMPT": 1,
          "TEMPERATURE": 0.7,
          "TOP_P": 1.0,
          "TOP_K": 0,
          "BETA": 0.0,
          "EPSILON": 0.003,
          "EPSILON_HIGH": 0.005,
          "LOSS_ALGO": "gspo-token",
          "LOSS_AGG_MODE": "sequence-mean-token-mean",
          "KL_LOSS_MODE": "low_var_kl",
          "ADVANTAGE_ESTIMATOR": "rloo",
          "SAMPLER_IS": "token",
          "SAMPLER_IS_THRESHOLD": 2,
          "USE_ROLLOUT_LOGPS": 1,
          "LEARNING_RATE": 1e-6,
          "ADAM_B1": 0.9,
          "ADAM_B2": 0.95,
          "WEIGHT_DECAY": 0,
          "OPT_CHAIN_TYPE": "clip_by_global_norm",
          "MAX_GRAD_NORM": 100,
          "FLASH_ATTENTION_BLOCK_SIZE": 256,
          "MODEL_PARAMETER_DTYPE": "float32",
          "EPISODE_TIMEOUT_SECS": 600,
          "DEBUG": 0,
      }.items()
  }


def prepare(args):
  w = {
      key: getattr(args, key)
      for key in (
          "batch",
          "generations",
          "steps",
          "warmup",
          "micro_groups",
          "prompt",
          "response",
          "turns",
          "seed",
          "dataset_size",
          "concurrency",
          "max_num_seqs",
          "batched_tokens",
      )
  }
  positive = set(w) - {"seed"}
  if any(w[key] <= 0 for key in positive):
    raise ValueError(
        "Counts must be positive (including at least one warmup step)."
    )
  if args.timeout <= 0:
    raise ValueError("timeout must be positive.")
  if w["steps"] <= w["warmup"] or w["generations"] < 2:
    raise ValueError("Require steps > warmup and generations >= 2.")
  if (
      w["batch"] % w["micro_groups"]
      or w["steps"] * w["batch"] > w["dataset_size"]
  ):
    raise ValueError(
        "Require batch divisible by micro_groups and enough dataset rows."
    )
  output = args.output.resolve()
  model_dir = args.model_dir.resolve()
  if not any(model_dir.glob("*.safetensors")):
    raise ValueError(
        "Pre-download the model; --model-dir must contain safetensors."
    )
  if not (model_dir / "config.json").is_file():
    raise ValueError("--model-dir must contain the model config.json.")
  # A new directory prevents accidental checkpoint resume or mixed-run events.
  output.mkdir(parents=True, exist_ok=False)
  env = {
      "WANDB_MODE": "disabled",
      "TUNIX_BENCHMARK_DIR": str(output),
      "PYTHON_BIN": sys.executable,
      "PYTHONUNBUFFERED": "1",
      "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
      "JAX_LOG_COMPILES": "1",
  }
  if args.mode == "dist":
    env.update(dist_environment(w, output, model_dir))
    command = [
        "bash",
        str(ROOT / "tunix/experimental/examples/frozenlake_dist/launcher.sh"),
    ]
  else:
    config = agentic_config(w, output, model_dir)
    write_json(output / "agentic.json", config)
    env.update({
        "JAX_PLATFORMS": "tpu,cpu",
        "TPU_VISIBLE_DEVICES": "0,1,2,3",
        "TPU_VISIBLE_CHIPS": "0,1,2,3",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,4,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "LIBTPU_INIT_ARGS": (
            "--deepsea_chips_per_host_bounds=1,4,1 --deepsea_host_bounds=1,1,1"
        ),
    })
    command = [
        sys.executable,
        "-m",
        BENCHMARK_MODULE + ".frozenlake_agentic",
        "tunix/cli/base_agentic_config.yaml",
        f"override_config_file={output / 'agentic.json'}",
    ]
  cache = args.cache_root.resolve() / args.mode
  env.update({
      "VLLM_CACHE_ROOT": str(cache / "vllm"),
      "VLLM_XLA_CACHE_PATH": str(cache / "vllm/xla_cache"),
      "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
  })
  versions = {}
  for name in (
      "jax",
      "jaxlib",
      "libtpu",
      "vllm",
      "tpu-inference",
      "tpu-raiden-jax",
      "flax",
      "optax",
  ):
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = None
  manifest = {
      "schema": 1,
      "mode": args.mode,
      "workload": w,
      "chips": 4,
      "hardware_label": args.hardware,
      "hostname": platform.node(),
      "python": sys.version,
      "host": platform.platform(),
      "versions": versions,
      "model_dir": str(model_dir),
      "cache_root": str(cache),
      "command": command,
      "environment": env,
      "revision": (
          subprocess.check_output(
              ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
          ).strip()
      ),
      "status": subprocess.check_output(
          ["git", "status", "--porcelain"], cwd=ROOT, text=True
      ),
  }
  manifest["diff_sha256"] = hashlib.sha256(
      subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
  ).hexdigest()
  manifest["benchmark_sha256"] = {
      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
      for p in Path(__file__).parent.glob("*.py")
  }
  manifest["model_metadata_sha256"] = {
      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
      for p in model_dir.glob("*.json")
  }
  manifest["model_shards"] = {
      p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
      for p in model_dir.glob("*.safetensors")
  }
  manifest["runtime_environment"] = {
      name: os.environ.get(name)
      for name in (
          "XLA_FLAGS",
          "JAX_ENABLE_COMPILATION_CACHE",
          "JAX_DEFAULT_MATMUL_PRECISION",
          "SKIP_JAX_PRECOMPILE",
      )
  }
  write_json(output / "manifest.json", manifest)
  print(json.dumps(manifest, indent=2))
  if not args.execute:
    print(
        "Plan only. Re-run with --execute and a NEW --output directory to"
        " train."
    )
    return
  start = time.monotonic()
  with (output / "launcher.log").open("x") as log:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
      code = process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
      try:
        process.wait(timeout=45)
      except subprocess.TimeoutExpired:
        pass
      # Children can survive after their launcher has exited.
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      process.wait()
      code = 124
  write_json(
      output / "result.json",
      {"returncode": code, "total_run_time_seconds": time.monotonic() - start},
  )
  if code:
    raise RuntimeError(
        f"Training failed ({code}); inspect {output}/launcher.log and worker"
        " logs."
    )
  summary = summarize(output)
  write_json(output / "summary.json", summary)


def percentile(values, percent):
  """Returns a linearly interpolated percentile without a NumPy dependency."""
  ordered = sorted(values)
  if not ordered:
    raise ValueError("Cannot compute a percentile of an empty sequence.")
  position = (len(ordered) - 1) * percent / 100
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = position - lower
  return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rollout_summary(events, steady_steps, expected):
  """Aggregates identical trajectory/model-call boundaries across both paths."""
  trajectories = [e for e in events if e["kind"] == "rollout_trajectory"]
  generations = [e for e in events if e["kind"] == "generation"]
  if any(not e.get("ok", True) for e in trajectories + generations):
    raise ValueError("Run contains failed rollout calls.")

  collection_times = []
  measured_trajectories = []
  measured_generations = []
  for step in steady_steps:
    step_trajectories = [e for e in trajectories if e.get("step") == step]
    step_generations = [e for e in generations if e.get("step") == step]
    if len(step_trajectories) != expected:
      raise ValueError(
          f"Rollout step {step} has {len(step_trajectories)} trajectories;"
          f" expected {expected}."
      )
    if not step_generations:
      raise ValueError(f"Rollout step {step} has no model generation calls.")
    start = min(e["start"] for e in step_trajectories)
    end = max(e["end"] for e in step_trajectories)
    seconds = end - start
    if seconds <= 0:
      raise ValueError("Invalid rollout timing interval.")
    collection_times.append(seconds)
    measured_trajectories.extend(step_trajectories)
    measured_generations.extend(step_generations)

  rollout_seconds = sum(collection_times)
  generated_tokens = sum(e["generated_tokens"] for e in measured_generations)
  prompt_tokens = sum(e["prompt_tokens"] for e in measured_generations)
  generated_by_trajectory = {}
  generation_seconds_by_trajectory = {}
  for event in measured_generations:
    identity = (
        event.get("step"),
        str(event.get("group_id")),
        event.get("pair_index"),
    )
    generated_by_trajectory[identity] = (
        generated_by_trajectory.get(identity, 0) + event["generated_tokens"]
    )
    generation_seconds_by_trajectory[identity] = (
        generation_seconds_by_trajectory.get(identity, 0) + event["seconds"]
    )
  if len(generated_by_trajectory) != len(measured_trajectories):
    raise ValueError(
        "Generation events do not map one-to-one to measured trajectories."
    )
  per_trajectory_tokens = list(generated_by_trajectory.values())
  trajectory_latencies = [e["seconds"] for e in measured_trajectories]
  generation_latencies = [e["seconds"] for e in measured_generations]
  total_environment_seconds = sum(
      e.get("environment_seconds", 0) for e in measured_trajectories
  )
  total_reward_seconds = sum(
      e.get("reward_seconds", 0) for e in measured_trajectories
  )
  rewards = [
      e["reward"] for e in measured_trajectories if e.get("reward") is not None
  ]
  total_generation_call_seconds = sum(generation_seconds_by_trajectory.values())
  other_seconds = []
  for event in measured_trajectories:
    identity = (
        event.get("step"),
        str(event.get("group_id")),
        event.get("pair_index"),
    )
    other_seconds.append(
        max(
            0,
            event["seconds"]
            - generation_seconds_by_trajectory[identity]
            - event.get("environment_seconds", 0)
            - event.get("reward_seconds", 0),
        )
    )
  return {
      "collection_time_seconds": {
          "median": statistics.median(collection_times),
          "per_step": collection_times,
      },
      "trajectories_per_second": len(measured_trajectories) / rollout_seconds,
      "output_tokens_per_second": generated_tokens / rollout_seconds,
      "trajectory_latency_seconds": {
          "median": percentile(trajectory_latencies, 50),
          "p90": percentile(trajectory_latencies, 90),
          "p99": percentile(trajectory_latencies, 99),
      },
      "model_call_latency_seconds": {
          "median": percentile(generation_latencies, 50),
          "p90": percentile(generation_latencies, 90),
          "p99": percentile(generation_latencies, 99),
      },
      "output_tokens_per_trajectory": {
          "mean": statistics.mean(per_trajectory_tokens),
          "median": percentile(per_trajectory_tokens, 50),
          "p90": percentile(per_trajectory_tokens, 90),
          "p99": percentile(per_trajectory_tokens, 99),
      },
      "average_prompt_tokens_per_model_call": (
          prompt_tokens / len(measured_generations)
      ),
      "average_output_tokens_per_model_call": (
          generated_tokens / len(measured_generations)
      ),
      "average_turns_per_trajectory": statistics.mean(
          e.get("turns", 0) for e in measured_trajectories
      ),
      "average_reward_per_trajectory": (
          statistics.mean(rewards) if rewards else None
      ),
      "average_time_per_trajectory_seconds": {
          "model_generation_api": (
              total_generation_call_seconds / len(measured_trajectories)
          ),
          "environment": (
              total_environment_seconds / len(measured_trajectories)
          ),
          "reward": total_reward_seconds / len(measured_trajectories),
          "agent_and_parsing": statistics.mean(other_seconds),
      },
      "trajectory_outcome_percent": {
          status: 100
          * sum(e.get("status") == status for e in measured_trajectories)
          / len(measured_trajectories)
          for status in sorted(
              {e.get("status", "") for e in measured_trajectories}
          )
      },
  }


def timing_summary(values):
  """Returns one readable timing summary without redundant derived fields."""
  return {"median": statistics.median(values), "per_step": values}


def api_call_summary(events, first, last, step_count):
  """Normalizes host-observed API usage by measured training steps."""
  durations = {}
  for event in events:
    if (
        event["kind"] == "span"
        and event["end"] > first
        and event["start"] < last
    ):
      durations.setdefault(event["name"], []).append(
          min(event["end"], last) - max(event["start"], first)
      )
  return {
      name: {
          "calls_per_step": len(values) / step_count,
          "host_time_per_step_seconds": sum(values) / step_count,
          "median_call_time_milliseconds": 1000 * percentile(values, 50),
          "p90_call_time_milliseconds": 1000 * percentile(values, 90),
      }
      for name, values in sorted(durations.items())
  }


def input_work_summary(events, name, steady_steps):
  """Summarizes logical padded input volume without reading device contents."""
  selected = [
      event
      for event in events
      if event["kind"] == "span"
      and event["name"] == name
      and event.get("step") in steady_steps
      and event.get("ok", True)
  ]
  shapes = [shape for event in selected for shape in event.get("shapes") or []]
  if not selected or not shapes:
    return None
  trajectories = 0
  padded_token_slots = 0
  unique_shapes = set()
  for shape in shapes:
    prompt_shape = shape["prompt"]
    completion_shape = shape["completion"]
    if not prompt_shape or not completion_shape:
      continue
    trajectories += int(prompt_shape[0])
    padded_token_slots += math.prod(prompt_shape) + math.prod(completion_shape)
    unique_shapes.add(json.dumps(shape, sort_keys=True))
  return {
      "trajectories_per_call": trajectories / len(selected),
      "padded_token_slots_per_step": padded_token_slots / len(steady_steps),
      "unique_tensor_shapes": [
          json.loads(shape) for shape in sorted(unique_shapes)
      ],
  }


def _merged_overlap(intervals, start, end):
  clipped = sorted(
      (max(left, start), min(right, end))
      for left, right in intervals
      if right > start and left < end
  )
  if not clipped:
    return 0.0
  total = 0.0
  current_start, current_end = clipped[0]
  for left, right in clipped[1:]:
    if left <= current_end:
      current_end = max(current_end, right)
    else:
      total += current_end - current_start
      current_start, current_end = left, right
  return total + current_end - current_start


def pipeline_summary(events, steady_steps):
  """Shows where rollout collection sits within each end-to-end step."""
  trajectories = [e for e in events if e["kind"] == "rollout_trajectory"]
  training_spans = [
      e
      for e in events
      if e["kind"] == "span" and e["name"] == "training"
  ]
  start_delays = []
  post_rollout_tails = []
  overlap_percentages = []
  for step_event in steady_steps:
    step = step_event["step"]
    step_trajectories = [e for e in trajectories if e.get("step") == step]
    rollout_start = min(e["start"] for e in step_trajectories)
    rollout_end = max(e["end"] for e in step_trajectories)
    rollout_seconds = rollout_end - rollout_start
    start_delays.append(max(0.0, rollout_start - step_event["start"]))
    post_rollout_tails.append(max(0.0, step_event["end"] - rollout_end))
    overlap = _merged_overlap(
        [
            (e["start"], e["end"])
            for e in training_spans
            if e.get("step") == step
        ],
        rollout_start,
        rollout_end,
    )
    overlap_percentages.append(100 * overlap / rollout_seconds)
  return {
      "rollout_start_delay_seconds": timing_summary(start_delays),
      "post_rollout_tail_seconds": timing_summary(post_rollout_tails),
      "training_overlap_with_rollout_percent": timing_summary(
          overlap_percentages
      ),
  }


def summarize(directory):
  directory = Path(directory)
  manifest = json.loads((directory / "manifest.json").read_text())
  result = json.loads((directory / "result.json").read_text())
  if result["returncode"] != 0:
    raise ValueError(f"Failed run: {directory}")
  event_files = sorted(directory.glob("events*.jsonl"))
  events = [
      json.loads(line)
      for path in event_files
      for line in path.read_text().splitlines()
  ]
  training_steps = [e for e in events if e["kind"] == "training_step"]
  w = manifest["workload"]
  expected = w["batch"] * w["generations"]
  if len(training_steps) != w["steps"] or any(
      e["trained_trajectories"] != expected for e in training_steps
  ):
    raise ValueError(
        "Incomplete run or training work mismatch; no valid speed comparison."
    )
  if any(e["kind"] == "span" and not e.get("ok", True) for e in events):
    raise ValueError(
        "Run contains failed calls; inspect events before comparing."
    )
  steady = training_steps[w["warmup"] :]
  warmup_durations = [e["seconds"] for e in training_steps[: w["warmup"]]]
  durations = [e["seconds"] for e in steady]
  if any(d <= 0 for d in durations):
    raise ValueError("Invalid timing interval.")
  first, last = steady[0]["start"], steady[-1]["end"]
  steady_step_ids = {e["step"] for e in steady}
  rollout = rollout_summary(events, sorted(steady_step_ids), expected)
  training_metric_values = {}
  for event in events:
    if (
        event["kind"] == "metric"
        and first <= event.get("time", 0) <= last
    ):
      training_metric_values.setdefault(event["name"], []).append(
          event["value"]
      )
  return {
      "mode": manifest["mode"],
      "end_to_end": {
          "steady_state_step_time_seconds": timing_summary(durations),
          "warmup_step_time_seconds": (
              timing_summary(warmup_durations) if warmup_durations else None
          ),
          "trajectories_per_second": (
              expected * len(steady) / sum(durations)
          ),
          "tpu_chip_seconds_per_trajectory": (
              manifest["chips"] * sum(durations) / (expected * len(steady))
          ),
      },
      "rollout": rollout,
      "pipeline": pipeline_summary(events, steady),
      "api_calls": api_call_summary(events, first, last, len(steady)),
      "training_input": input_work_summary(
          events, "training", steady_step_ids
      ),
      "actor_log_probs_input": input_work_summary(
          events, "actor_log_probs", steady_step_ids
      ),
      "training_metrics": {
          name: statistics.median(values)
          for name, values in sorted(training_metric_values.items())
      },
      "total_run_time_seconds": result["total_run_time_seconds"],
  }


def _path_value(value, path):
  for key in path:
    if value is None:
      return None
    value = value.get(key)
  return value


def _compare_values(agentic_values, dist_values, lower_is_better):
  if not agentic_values and not dist_values:
    return None
  agentic = statistics.median(agentic_values) if agentic_values else None
  dist = statistics.median(dist_values) if dist_values else None
  if agentic is None or dist is None:
    interpretation = "one_stack_only"
  elif math.isclose(agentic, dist, rel_tol=1e-9, abs_tol=1e-12):
    interpretation = "same"
  elif lower_is_better is None:
    interpretation = "different_not_ranked"
  elif (dist < agentic) == lower_is_better:
    interpretation = "dist_faster"
  else:
    interpretation = "dist_slower"
  return {
      "agentic": agentic,
      "dist": dist,
      "dist_minus_agentic": (
          dist - agentic if dist is not None and agentic is not None else None
      ),
      "dist_relative_to_agentic_percent": (
          100 * (dist / agentic - 1)
          if dist is not None and agentic
          else None
      ),
      "lower_is_better": lower_is_better,
      "interpretation": interpretation,
  }


def compare_summaries(summaries):
  """Builds a direct, multi-run distributed-vs-agentic gap report."""
  by_mode = {
      mode: [summary for summary in summaries if summary["mode"] == mode]
      for mode in ("agentic", "dist")
  }
  if not all(by_mode.values()):
    raise ValueError("Comparison requires at least one agentic and one dist run.")

  def compare(path, lower_is_better):
    return _compare_values(
        [
            value
            for summary in by_mode["agentic"]
            if (value := _path_value(summary, path)) is not None
        ],
        [
            value
            for summary in by_mode["dist"]
            if (value := _path_value(summary, path)) is not None
        ],
        lower_is_better,
    )

  metrics = {}
  for name, path, lower_is_better in (
      (
          "end_to_end_step_time_seconds",
          ("end_to_end", "steady_state_step_time_seconds", "median"),
          True,
      ),
      (
          "end_to_end_trajectories_per_second",
          ("end_to_end", "trajectories_per_second"),
          False,
      ),
      (
          "warmup_step_time_seconds",
          ("end_to_end", "warmup_step_time_seconds", "median"),
          True,
      ),
      (
          "tpu_chip_seconds_per_trajectory",
          ("end_to_end", "tpu_chip_seconds_per_trajectory"),
          True,
      ),
      (
          "rollout_collection_time_seconds",
          ("rollout", "collection_time_seconds", "median"),
          True,
      ),
      (
          "rollout_trajectories_per_second",
          ("rollout", "trajectories_per_second"),
          False,
      ),
      (
          "rollout_output_tokens_per_second",
          ("rollout", "output_tokens_per_second"),
          False,
      ),
      (
          "trajectory_latency_median_seconds",
          ("rollout", "trajectory_latency_seconds", "median"),
          True,
      ),
      (
          "trajectory_latency_p90_seconds",
          ("rollout", "trajectory_latency_seconds", "p90"),
          True,
      ),
      (
          "trajectory_latency_p99_seconds",
          ("rollout", "trajectory_latency_seconds", "p99"),
          True,
      ),
      (
          "model_call_latency_median_seconds",
          ("rollout", "model_call_latency_seconds", "median"),
          True,
      ),
      (
          "model_call_latency_p90_seconds",
          ("rollout", "model_call_latency_seconds", "p90"),
          True,
      ),
      (
          "model_call_latency_p99_seconds",
          ("rollout", "model_call_latency_seconds", "p99"),
          True,
      ),
      (
          "rollout_start_delay_seconds",
          ("pipeline", "rollout_start_delay_seconds", "median"),
          True,
      ),
      (
          "post_rollout_tail_seconds",
          ("pipeline", "post_rollout_tail_seconds", "median"),
          True,
      ),
      (
          "training_overlap_with_rollout_percent",
          ("pipeline", "training_overlap_with_rollout_percent", "median"),
          None,
      ),
      (
          "training_trajectories_per_call",
          ("training_input", "trajectories_per_call"),
          None,
      ),
      (
          "training_padded_token_slots_per_step",
          ("training_input", "padded_token_slots_per_step"),
          None,
      ),
      (
          "actor_log_probs_trajectories_per_call",
          ("actor_log_probs_input", "trajectories_per_call"),
          None,
      ),
      (
          "actor_log_probs_padded_token_slots_per_step",
          ("actor_log_probs_input", "padded_token_slots_per_step"),
          None,
      ),
      (
          "average_output_tokens_per_trajectory",
          ("rollout", "output_tokens_per_trajectory", "mean"),
          None,
      ),
      (
          "average_prompt_tokens_per_model_call",
          ("rollout", "average_prompt_tokens_per_model_call"),
          None,
      ),
      (
          "average_output_tokens_per_model_call",
          ("rollout", "average_output_tokens_per_model_call"),
          None,
      ),
      (
          "average_turns_per_trajectory",
          ("rollout", "average_turns_per_trajectory"),
          None,
      ),
      (
          "average_reward_per_trajectory",
          ("rollout", "average_reward_per_trajectory"),
          None,
      ),
      (
          "model_generation_time_per_trajectory_seconds",
          (
              "rollout",
              "average_time_per_trajectory_seconds",
              "model_generation_api",
          ),
          True,
      ),
      (
          "environment_time_per_trajectory_seconds",
          ("rollout", "average_time_per_trajectory_seconds", "environment"),
          True,
      ),
      (
          "reward_time_per_trajectory_seconds",
          ("rollout", "average_time_per_trajectory_seconds", "reward"),
          True,
      ),
      (
          "agent_and_parsing_time_per_trajectory_seconds",
          (
              "rollout",
              "average_time_per_trajectory_seconds",
              "agent_and_parsing",
          ),
          True,
      ),
      (
          "total_run_time_seconds",
          ("total_run_time_seconds",),
          True,
      ),
  ):
    metrics[name] = compare(path, lower_is_better)

  stages = sorted(
      {
          stage
          for summary in summaries
          for stage in summary.get("api_calls", {})
      }
  )
  api_calls = {}
  for stage in stages:
    api_calls[stage] = {}
    for field, lower_is_better in (
        ("calls_per_step", None),
        ("host_time_per_step_seconds", True),
        ("median_call_time_milliseconds", True),
        ("p90_call_time_milliseconds", True),
    ):
      if stage == "rollout_poll":
        lower_is_better = None
      api_calls[stage][field] = compare(
          ("api_calls", stage, field), lower_is_better
      )

  outcomes = sorted(
      {
          outcome
          for summary in summaries
          for outcome in summary["rollout"]["trajectory_outcome_percent"]
      }
  )
  training_metric_names = sorted(
      {
          name
          for summary in summaries
          for name in summary.get("training_metrics", {})
      }
  )

  def shapes_match(section):
    values = {
        json.dumps(
            (summary.get(section) or {}).get("unique_tensor_shapes"),
            sort_keys=True,
        )
        for summary in summaries
    }
    return len(values) == 1 and "null" not in values

  def scalar_match(section, field):
    values = [
        (summary.get(section) or {}).get(field) for summary in summaries
    ]
    return all(value is not None for value in values) and all(
        math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-12)
        for value in values[1:]
    )

  return {
      "run_count": {mode: len(values) for mode, values in by_mode.items()},
      "parity_checks": {
          "training_tensor_shapes_match": shapes_match("training_input"),
          "actor_log_probs_tensor_shapes_match": shapes_match(
              "actor_log_probs_input"
          ),
          "training_trajectories_per_call_match": scalar_match(
              "training_input", "trajectories_per_call"
          ),
          "training_padded_token_slots_per_step_match": scalar_match(
              "training_input", "padded_token_slots_per_step"
          ),
          "actor_log_probs_trajectories_per_call_match": scalar_match(
              "actor_log_probs_input", "trajectories_per_call"
          ),
          "actor_log_probs_padded_token_slots_per_step_match": scalar_match(
              "actor_log_probs_input", "padded_token_slots_per_step"
          ),
      },
      "metrics": metrics,
      "api_calls": api_calls,
      "trajectory_outcome_percent": {
          outcome: compare(
              ("rollout", "trajectory_outcome_percent", outcome), None
          )
          for outcome in outcomes
      },
      "training_metrics": {
          name: compare(("training_metrics", name), None)
          for name in training_metric_names
      },
  }


def report(runs):
  manifests = [json.loads((p / "manifest.json").read_text()) for p in runs]
  for manifest in manifests[1:]:
    for key in (
        "workload",
        "versions",
        "revision",
        "model_dir",
        "diff_sha256",
        "benchmark_sha256",
        "hardware_label",
        "hostname",
        "python",
        "model_metadata_sha256",
        "model_shards",
        "runtime_environment",
    ):
      if manifest[key] != manifests[0][key]:
        raise ValueError(
            f"Runs differ in {key}; report each separately or rerun matched"
            " configurations."
        )
  datasets = [json.loads((p / "dataset.json").read_text()) for p in runs]
  if any(d != datasets[0] for d in datasets[1:]):
    raise ValueError("Dataset order/content mismatch.")
  summaries = [summarize(p) for p in runs]
  return compare_summaries(summaries)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="action", required=True)
  run = commands.add_parser("run")
  run.add_argument(
      "--mode",
      choices=("dist", "agentic"),
      required=True,
  )
  run.add_argument("--output", type=Path, required=True)
  run.add_argument("--model-dir", type=Path, required=True)
  run.add_argument("--cache-root", type=Path, required=True)
  run.add_argument(
      "--hardware",
      required=True,
      help="Hardware label, e.g. v5p-4; confirm device inventory in logs.",
  )
  run.add_argument("--execute", action="store_true")
  run.add_argument("--timeout", type=int, default=14400)
  for name, default in {
      "batch": 8,
      "generations": 8,
      "steps": 12,
      "warmup": 2,
      "micro_groups": 1,
      "prompt": 2048,
      "response": 2048,
      "turns": 8,
      "seed": 42,
      "dataset_size": 10000,
      "concurrency": 64,
      "max_num_seqs": 32,
      "batched_tokens": 8192,
  }.items():
    run.add_argument("--" + name.replace("_", "-"), type=int, default=default)
  report_parser = commands.add_parser("report")
  report_parser.add_argument("runs", nargs="+", type=Path)
  args = parser.parse_args()
  if args.action == "run":
    prepare(args)
  else:
    print(json.dumps(report(args.runs), indent=2))


if __name__ == "__main__":
  main()
