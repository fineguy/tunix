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

"""Opt-in host-side tracing for RL efficiency experiments.

Only the benchmark entry points install these hooks. Timings include RPC waits
and framework dispatch; they are not TPU kernel durations. No per-call device
barriers are inserted, since those would change pipeline overlap. Hooks are
installed at generic rollout and sampler boundaries, so they work for both
single-turn and multi-turn agents.
"""

import atexit
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time


def _length(value):
  if value is None:
    return 0
  shape = getattr(value, "shape", None)
  if shape is not None:
    size = 1
    for dimension in shape:
      size *= int(dimension)
    return size
  try:
    return len(value)
  except TypeError:
    return 0


def _sum(value):
  if value is None:
    return 0.0
  if isinstance(value, dict):
    return sum(_sum(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return sum(_sum(item) for item in value)
  array_sum = getattr(value, "sum", None)
  if callable(array_sum):
    return float(array_sum())
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def _identity(engine, result=None):
  env = getattr(engine, "env", None)
  task = getattr(env, "task", {}) or {}
  extra = getattr(env, "extra_kwargs", {}) or {}
  result = result if isinstance(result, dict) else {}
  step = result.get("policy_version")
  if step is None:
    step = task.get("policy_version", extra.get("benchmark_policy_version", -1))
  return {
      "step": int(step) if step is not None else -1,
      "group_id": result.get("group_id", extra.get("group_id")),
      "pair_index": extra.get("pair_index"),
  }


def _generation_counts(output):
  generated = sum(_length(row) for row in getattr(output, "tokens", []) or [])
  prompt_lengths = getattr(output, "prompt_lengths", None)
  if prompt_lengths is not None:
    prompt = int(_sum(prompt_lengths))
  else:
    # Both benchmark paths issue one request per model call. In that case the
    # left-padded width is the real prompt length, not cross-request padding.
    prompt = _length(getattr(output, "left_padded_prompt_tokens", None))
  return prompt, generated


def _trajectory_fields(engine, result):
  result = result if isinstance(result, dict) else {}
  masks = result.get("conversation_masks")
  conversation_tokens = _length(result.get("conversation_tokens"))
  generated_tokens = int(_sum(masks)) if masks is not None else 0
  prompt_length = result.get("prompt_length")
  if prompt_length is None:
    prompt_length = _length(result.get("prompt_tokens"))
  status = result.get("status", "")
  if hasattr(status, "name"):
    status = status.name
  trajectory = getattr(getattr(engine, "agent", None), "trajectory", None)
  reward = result.get("trajectory_reward")
  if reward is None:
    reward = getattr(trajectory, "reward", None)
  return {
      **_identity(engine, result),
      "prompt_tokens": int(prompt_length),
      "generated_tokens": generated_tokens,
      "environment_tokens": max(0, conversation_tokens - generated_tokens),
      "conversation_tokens": conversation_tokens,
      "turns": len(getattr(trajectory, "steps", []) or []),
      "environment_seconds": _sum(result.get("env_time")),
      "reward_seconds": _sum(result.get("reward_time")),
      "reward": None if reward is None else float(reward),
      "status": str(status),
  }


def record_dataset(rows):
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if directory:
    value = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    with (Path(directory) / "dataset.json").open("x") as file:
      json.dump(
          {
              "rows": len(rows),
              "sha256": hashlib.sha256(value.encode()).hexdigest(),
          },
          file,
      )


def batch_shapes(batches):
  """Inspects shapes only; never copies TPU arrays to the host."""
  return [
      {
          "prompt": list(b.prompt_ids.shape),
          "completion": list(b.completion_ids.shape),
      }
      for b in batches
  ]


def token_shapes(prompt_tokens, completion_tokens):
  """Returns token-array shapes without transferring array contents."""
  return [{
      "prompt": list(prompt_tokens.shape),
      "completion": list(completion_tokens.shape),
  }]


class Recorder:
  """Records successful sync boundaries and API spans on one host clock."""

  def __init__(self, path, clock=time.monotonic):
    self.clock = clock
    self.file = Path(path).open("x", encoding="utf-8", buffering=1)
    self.lock = threading.Lock()
    self.previous_sync = clock()
    self.pending_trajectories = 0
    self.step = 0

  def emit(self, kind, **fields):
    with self.lock:
      if self.file.closed:
        return
      self.file.write(
          json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n"
      )

  def close(self):
    with self.lock:
      self.file.close()

  def finish(self, name, start, end, trajectories=0, **fields):
    self.emit(
        "span",
        name=name,
        step=self.step,
        start=start,
        end=end,
        seconds=end - start,
        trained_trajectories=trajectories,
        **fields,
    )
    if fields.get("ok", True):
      self.pending_trajectories += trajectories
      if name == "weight_sync":
        if self.pending_trajectories:
          self.emit(
              "training_step",
              step=self.step,
              start=self.previous_sync,
              end=end,
              seconds=end - self.previous_sync,
              trained_trajectories=self.pending_trajectories,
          )
          self.step += 1
          self.pending_trajectories = 0
        else:
          self.emit("initial_sync", end=end)
        self.previous_sync = end

  def wrap(self, cls, method, name, trajectories_fn=None, shapes_fn=None):
    original = getattr(cls, method)

    def finish(start, args, kwargs, ok):
      event_name = name(args, kwargs) if callable(name) else name
      trajectories = (
          trajectories_fn(args, kwargs) if ok and trajectories_fn else 0
      )
      end = self.clock()
      shapes = shapes_fn(args, kwargs) if ok and shapes_fn else None
      self.finish(
          event_name,
          start,
          end,
          trajectories=trajectories,
          ok=ok,
          shapes=shapes,
      )

    if inspect.iscoroutinefunction(original):

      @functools.wraps(original)
      async def wrapped(*args, **kwargs):
        start = self.clock()
        ok = False
        try:
          result = await original(*args, **kwargs)
          ok = True
          return result
        finally:
          finish(start, args, kwargs, ok)

    else:

      @functools.wraps(original)
      def wrapped(*args, **kwargs):
        start = self.clock()
        ok = False
        try:
          result = original(*args, **kwargs)
          ok = True
          return result
        finally:
          finish(start, args, kwargs, ok)

    setattr(cls, method, wrapped)


def _install_trajectory_hooks(recorder):
  """Records the same episode/model-call boundaries in both implementations."""
  from tunix.rl.agentic.trajectory import trajectory_collect_engine

  TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine

  original_collect = TrajectoryCollectEngine.collect
  if getattr(original_collect, "_tunix_benchmark_hook", False):
    raise RuntimeError("Trajectory benchmark hooks were installed twice.")

  def instrument_model_call(engine, model_call):
    def finish(start, output, ok):
      end = recorder.clock()
      prompt, generated = _generation_counts(output) if ok else (0, 0)
      recorder.emit(
          "generation",
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          prompt_tokens=prompt,
          generated_tokens=generated,
          **_identity(engine),
      )

    if inspect.iscoroutinefunction(model_call):

      @functools.wraps(model_call)
      async def async_call(*args, **kwargs):
        start = recorder.clock()
        output = None
        ok = False
        try:
          output = await model_call(*args, **kwargs)
          ok = True
          return output
        finally:
          finish(start, output, ok)

      return async_call

    @functools.wraps(model_call)
    def sync_call(*args, **kwargs):
      start = recorder.clock()
      output = None
      ok = False
      try:
        output = model_call(*args, **kwargs)
        ok = True
        return output
      finally:
        finish(start, output, ok)

    return sync_call

  @functools.wraps(original_collect)
  async def collect(engine, *args, **kwargs):
    start = recorder.clock()
    result = None
    ok = False
    model_call = engine.model_call
    engine.model_call = instrument_model_call(engine, model_call)
    try:
      result = await original_collect(engine, *args, **kwargs)
      ok = True
      return result
    finally:
      engine.model_call = model_call
      end = recorder.clock()
      fields = _trajectory_fields(engine, result) if ok else _identity(engine)
      recorder.emit(
          "rollout_trajectory",
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          **fields,
      )

  collect._tunix_benchmark_hook = True
  TrajectoryCollectEngine.collect = collect


def _install_dist_rollout_identity_hook():
  """Makes request identity visible to the shared inner collector hook."""
  from tunix.experimental.rollout.collector import TrajectoryCollectorEngine

  original = TrajectoryCollectorEngine.run_episode

  @functools.wraps(original)
  async def run_episode(collector, *args, **kwargs):
    extra = getattr(collector.env, "extra_kwargs", None)
    if isinstance(extra, dict):
      extra["benchmark_policy_version"] = (
          collector.request.target_policy_version
      )
      extra["group_id"] = collector.request.prompt_id
      extra["pair_index"] = collector.request.group_index
    return await original(collector, *args, **kwargs)

  TrajectoryCollectorEngine.run_episode = run_episode


def install(mode, label=None):
  """Installs benchmark-only hooks after the entry point's normal imports."""
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if not directory:
    return
  filename = "events.jsonl"
  if mode == "rollout-worker":
    safe_label = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(label or "0")
    )
    filename = f"events-rollout-{safe_label}.jsonl"
  recorder = Recorder(Path(directory) / filename)
  atexit.register(recorder.close)
  if mode == "rollout-worker":
    _install_dist_rollout_identity_hook()
    _install_trajectory_hooks(recorder)
    return
  if mode == "dist":
    from tunix.experimental.orchestrator import distributed_rl_engine

    DistributedRLEngine = distributed_rl_engine.DistributedRLEngine

    def trajectories(args, kwargs):
      payload = args[1] if len(args) > 1 else kwargs["payload"]
      return int(payload.completion_ids.shape[0])

    recorder.wrap(
        DistributedRLEngine,
        "train_step",
        "training",
        trajectories,
        lambda args, kwargs: batch_shapes(
            [args[1] if len(args) > 1 else kwargs["payload"]]
        ),
    )
    for method, name in (
        ("dispatch_rollouts", "rollout_dispatch"),
        ("poll_rollouts", "rollout_poll"),
        ("sync_weights", "weight_sync"),
        ("save_checkpoint", "checkpoint_save"),
        ("get_metrics", "metrics_fetch"),
    ):
      recorder.wrap(DistributedRLEngine, method, name)
    recorder.wrap(
        DistributedRLEngine,
        "_invoke_worker",
        lambda args, kwargs: "worker_rpc_"
        + str(args[2] if len(args) > 2 else kwargs["method_name"]),
    )
    recorder.wrap(
        DistributedRLEngine,
        "per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda args, kwargs: token_shapes(
            (args[2] if len(args) > 2 else kwargs["items"]).prompt_tokens,
            (args[2] if len(args) > 2 else kwargs["items"]).completion_tokens,
        ),
    )
  else:
    from tunix.rl.rl_cluster import RLEngine

    def trajectories(args, kwargs):
      batches = args[1] if len(args) > 1 else kwargs["train_ds"]
      return sum(int(batch.completion_ids.shape[0]) for batch in batches)

    recorder.wrap(
        RLEngine,
        "update_actor",
        "training",
        trajectories,
        lambda args, kwargs: batch_shapes(
            args[1] if len(args) > 1 else kwargs["train_ds"]
        ),
    )
    recorder.wrap(
        RLEngine,
        "get_actor_per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda args, kwargs: token_shapes(
            args[1] if len(args) > 1 else kwargs["prompt_tokens"],
            args[2] if len(args) > 2 else kwargs["completion_tokens"],
        ),
    )
    recorder.wrap(RLEngine, "sync_weights", "weight_sync")
    _install_trajectory_hooks(recorder)

  from tunix.rl import common as rl_common

  recorder.wrap(
      rl_common,
      "sampler_trainer_agreement",
      "sampler_trainer_agreement",
  )

  # Preserve existing metric names and steps; their aggregation scopes differ.
  import jax

  def scalar(name, value, **kwargs):
    if any(
        key in name for key in ("reward", "length", "clip", "loss", "staleness")
    ):
      try:
        number = float(value)
        if math.isfinite(number):
          recorder.emit(
              "metric",
              name=name,
              value=number,
              step=int(kwargs.get("step", -1)),
              time=time.monotonic(),
          )
      except (TypeError, ValueError):
        pass

  jax.monitoring.register_scalar_listener(scalar)
