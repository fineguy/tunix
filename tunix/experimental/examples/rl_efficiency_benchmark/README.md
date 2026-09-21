# Distributed vs. Agentic RL Efficiency Benchmarks

## Goal and scope

This directory contains infrastructure-efficiency benchmarks for online RL.
The common tracing layer measures optimizer steps, rollout collection, model
generation, tokens, RPCs, and weight synchronization at boundaries shared by
single-turn and multi-turn agents. It does not require an episode to have more
than one turn: a single-turn sample is one complete rollout with `turns=1`.

The first included workload compares online FrozenLake GRPO training with
Gemma4 E2B on a single four-chip TPU VM. Code inspection can identify extra
calls, data movement, and synchronization boundaries, but this repository does
not include TPU benchmark results. This document does not assume that the
distributed path is slower or claim an unmeasured speedup.

The baselines are the agentic CLI path configured by
`examples/frozenlake/configs/gemma4_e2b.yaml` and the distributed launcher in
`../frozenlake_dist/`. Both paths reuse the same agent, environment, and
dataset generation function. The harness explicitly aligns the optimizer to constant
LR=1e-6, AdamW b1=0.9/b2=0.95, weight decay 0, and global-norm clipping at 100.
This matches the Python recipe optimizer; the warmup/cosine schedule in the
original CLI YAML is not used. This configuration alignment does not imply
bitwise-identical training results.

## Directory layout and workload scope

- `runtime.py` is workload-independent host-side instrumentation. A recipe can
  opt in by setting `TUNIX_BENCHMARK_DIR`, installing the hooks in its trainer
  and rollout processes, and recording its ordered input data.
- `frozenlake.py` is the matched FrozenLake runner and reporter.
- `frozenlake_agentic.py` supplies identical FrozenLake input data to the
  agentic CLI.

The summary schema treats a rollout as the unit of work, not a multi-turn
conversation. Generation call count and turn count are reported separately,
so the same instrumentation can compare single-call tasks such as math QA,
multi-turn environments such as FrozenLake, or tool-using agents. A new
workload still needs a small runner that constructs matched agentic and
distributed configurations; benchmark-only code should be added here instead
of inside the recipe directory.

## Performance risks found in the code

| Stack difference visible in the code | Expected effect | Metric that confirms or rejects it |
| --- | --- | --- |
| Agentic defines training micro-batches in prompt groups, while distributed `PaddedBatchAssembler` defines them in trajectories | This is a unit mismatch, not an intended performance difference. The harness converts `micro_groups * G` for distributed, so `micro_groups=2, G=8` means 16 trajectories per training call on both stacks | `training_input.trajectories_per_call`, `training_input.padded_token_slots_per_step`, and `unique_tensor_shapes` must match before any speed result is accepted |
| Distributed `rollout_dispatch_stage` calls `dispatch_rollouts([prompt])` once per prompt group, and `dispatch_rollout_requests` then awaits each generated request | B prompt groups create B top-level dispatch calls and B*G worker dispatches. Agentic feeds an in-process prompt queue, so short episodes can expose distributed control-plane latency | `api_calls.rollout_dispatch`, `pipeline.rollout_start_delay_seconds`, rollout collection time, and trajectories/s |
| Distributed long-polls remote rollout workers; agentic receives results from its in-process rollout orchestrator queue | Distributed adds long-poll RPCs, response deserialization, and queue handoff. Poll time includes useful waiting and must not be labeled pure waste | `api_calls.rollout_poll`, the `worker_rpc_*` entries, rollout p90/p99 latency, and post-rollout tail |
| Each distributed training chunk performs a remote actor-logps call and remote trainer calls; the final chunk also invokes update and metrics RPCs | Extra serialization, scheduling, and RPC latency can dominate when chunks are small. Agentic invokes its trainer in-process | `api_calls.actor_log_probs`, `api_calls.training`, `worker_rpc_per_token_logps`, `worker_rpc_fwd_bwd`, `worker_rpc_update`, `worker_rpc_get_metrics`, and normalized calls per step |
| Both stacks recompute actor logps and apply token sampler-IS, but distributed converts returned logps to NumPy on the control process before sending updated payloads back | The model forward pass is common work; distributed may add device-to-host, host computation, serialization, and host-to-device cost | Match `actor_log_probs_input`; compare actor-logps host time and `sampler_trainer_agreement`. RPC spans include transfer plus remote compute, so a TPU trace is still needed to isolate DMA |
| Distributed weight sync crosses worker/process boundaries through Raiden; agentic calls `rollout.update_params` inside its process | The amount of model data is similar, but transport, registration, and synchronization mechanisms differ | `api_calls.weight_sync` host time per step and call latency. Treat the end-to-end step metric as authoritative because asynchronous work may extend outside the API call |
| Both implementations pipeline rollout and training, but their queue and capacity controls differ | A slower tail or a lost overlap opportunity can leave trainer or rollout chips idle even when individual API latency looks similar | `pipeline.training_overlap_with_rollout_percent`, rollout-start delay, post-rollout tail, and the sync-to-sync end-to-end step time |
| Both implementations use fixed padded tensors | Different padded shapes would change TPU compute and invalidate an infrastructure comparison even if raw output lengths look similar | `training_input` and `actor_log_probs_input` token slots/shapes, plus output tokens per trajectory |
| Both paths use a fixed 2+2 mesh | Trainer and rollout each have two chips. This removes shared-mesh contention, but role imbalance can still leave half of the VM idle | Confirm device assignment in logs; use a TPU trace for utilization. Shared mesh is intentionally outside this benchmark |
| Agentic's built-in global-step metric ends before weight sync, while distributed's existing step metric has a different boundary | Direct comparison of the existing dashboards is biased | Use only `end_to_end.steady_state_step_time_seconds`, whose boundary is successful sync completion to successful sync completion on both stacks |
| Cold compilation, model startup, logging, checkpointing, or a full cache disk can dominate | These effects can hide the steady-state stack difference | Compare warmup step time and total run time separately. The benchmark disables W&B, evaluation, trajectory output, and scheduled checkpoints and isolates cache paths |

The recommended order is: align microbatch units and timing boundaries, measure
RPC/logps round trips, and only then evaluate batched dispatch, packing, and
pipeline overlap. Do not change the scheduler or loss in the baseline, because
that would also change the algorithmic workload.

### Expected gaps before measurement

Code inspection supports the following hypotheses, not measured conclusions:

1. **Rollout model compute should be close.** Both modes use Gemma4 E2B, the
   same tokenizer and sampling settings, a two-chip TP rollout mesh, and the
   same trajectory collector. Similar model-call latency but worse distributed
   rollout collection indicates dispatch, polling, serialization, or queueing;
   worse model-call latency indicates a backend/configuration/device-placement
   difference rather than orchestrator overhead.
2. **Distributed control-plane overhead is most exposed by FrozenLake.** Each
   prompt group and generation is short relative to DeepSWE, while distributed
   performs repeated dispatch and poll operations across process boundaries.
   A larger rollout-start delay, more dispatch/poll host time, or lower rollout
   trajectories/s would confirm this gap.
3. **Trainer tensor work should match.** The harness converts group-based
   agentic microbatch units to trajectory-based distributed units. If padded
   token slots, tensor shapes, or trajectories per call differ, the run is not
   an infrastructure-only comparison. With matching inputs, excess distributed
   training/logps time points to RPC, serialization, transfer, or worker queue
   overhead around otherwise similar TPU work.
4. **Sampler-IS adds common compute but different data movement.** Both stacks
   run the actor-logps forward pass and sampler/trainer agreement. Distributed
   additionally returns logps to the control process and sends the adjusted
   payload back to the trainer. The actor-logps and agreement timings show the
   combined host-visible gap; only a TPU trace can separate model compute from
   DMA and serialization.
5. **Weight sync has no obvious winner from code alone.** Distributed uses
   Raiden across workers; agentic updates the rollout model inside one process
   but still moves parameters between dedicated meshes. Compare weight-sync
   latency and the sync-inclusive step, rather than assuming in-process is
   automatically cheaper.
6. **Pipeline scheduling can dominate isolated API costs.** Both stacks overlap
   rollout and training. A distributed run may have similar generation and
   training calls yet lose end-to-end time through delayed rollout start, less
   overlap, or a longer post-rollout tail. API host times overlap and must not
   be summed into a synthetic step time.

## Main experiments

| Mode | Actor mesh | Rollout mesh | Total chips | Purpose |
| --- | --- | --- | --- | --- |
| `dist` | chips 0-1, FSDP=2/TP=1 | chips 2-3, DP=1/TP=2 | 4 | Current distributed infrastructure |
| `agentic` | dedicated two-chip FSDP=2/TP=1 | dedicated two-chip DP=1/TP=2 | 4 | Infrastructure control with the same resource split |

For agentic, inspect the actual device allocation, visible vLLM devices, and
rollout mesh in the logs. A configuration declaration alone does not prove
hardware isolation.

The default matched workload is B=mini_batch=8, G=8, one iteration, and 12
full batches with the first two discarded. It uses prompt=2048, response=2048,
vLLM context=4352, eight turns, temperature=0.7, top_p=1, top_k=0,
micro_groups=1 (eight actual trajectories), 32 vLLM sequences, 8192 batched
tokens, concurrency=64, and HBM utilization=0.2. The algorithm settings are
GSPO-token, RLOO, clip=0.003/0.005, token TIS threshold=2, beta=0, and
staleness=0. Every run reloads the same local checkpoint and must not resume a
previous training state.

The same seed guarantees only the input map sequence. Concurrent sampling
order, TP/DP sharding, and kernel numerics can change trajectory length, reward,
and gradients. Inspect raw reward, raw length, clip ratio, and loss as well.
The report does not automatically claim quality equivalence. If lengths differ
materially, add a fixed-token trajectory replay or single-stage benchmark;
online trajectories/s alone cannot establish equal low-level compute work.

## Running the benchmark

Run from the repository root. Prepare a local Gemma4 model directory containing
the safetensors, config, and tokenizer files. Agentic uses AutoModel's `gcs`
source with this existing local path, which that loader supports, to avoid
changing model revisions or downloading files during a run.

```bash
python3 tunix/experimental/examples/rl_efficiency_benchmark/frozenlake.py run \
  --mode dist --hardware v5p-4 \
  --model-dir /path/to/gemma-4-E2B-it \
  --cache-root /path/with/free-space/frozenlake-cache \
  --output /path/to/results/dist-plan
```

By default, the command only writes the manifest and agentic configuration; it
does not start training. After inspection, use a new output path and add
`--execute`. The output directory must not already exist, which prevents stale
logs or checkpoints from being mixed into a run. `--hardware` is a
user-supplied label; the manifest also records the hostname, dependency
versions, and launch command. Do not run both modes concurrently on one VM.

```bash
python3 tunix/experimental/examples/rl_efficiency_benchmark/frozenlake.py run \
  --mode dist --hardware v5p-4 \
  --model-dir /path/to/gemma-4-E2B-it \
  --cache-root /path/with/free-space/frozenlake-cache \
  --output /path/to/results/dist-1 --execute

python3 tunix/experimental/examples/rl_efficiency_benchmark/frozenlake.py run \
  --mode agentic --hardware v5p-4 \
  --model-dir /path/to/gemma-4-E2B-it \
  --cache-root /path/with/free-space/frozenlake-cache \
  --output /path/to/results/agentic-1 --execute

python3 tunix/experimental/examples/rl_efficiency_benchmark/frozenlake.py report \
  /path/to/results/dist-1 /path/to/results/agentic-1
```

Run each mode independently at least three times and alternate order
(dist/agentic, then agentic/dist). Warm-cache experiments reuse `cache-root`,
with separate subdirectories for each mode. Cold-start experiments use a new
cache root; do not delete an existing user cache. Check `df -h` and `df -i`,
confirm the cache paths in the logs, and verify that significant compilation
spikes do not continue after 12 steps. If compilation continues, increase
warmup and total steps equally for both modes. Warmup alone is not proof that
all compilation has completed.

Suggested matched sweeps:

```bash
# Move toward the reference full-batch and concurrency scale after the small run succeeds.
--batch 64 --concurrency 512 --micro-groups 2

# Keep the prompt budget and increase only the episode budget; context becomes 6400.
--response 4096 --batched-tokens 16384
```

`micro-groups=1/2/4` corresponds to 8/16/32 actual trajectories. Record OOM as
a capacity result. If one mode OOMs, apply the same microbatch adjustment to
both modes before rerunning; lowering only one side is not a matched-shape
comparison. The harness fixes one full batch to one optimizer update.
Multi-mini-batch, off-policy, multi-host, and DeepSWE sandbox benchmarks need
separate experiments and are not part of this result.

## Metrics and measurement boundaries

| Output field | Question it answers |
| --- | --- |
| `end_to_end.steady_state_step_time_seconds` | How long does one complete sync-to-sync training step take after warmup? |
| `end_to_end.warmup_step_time_seconds` | How expensive are compilation and cold caches before steady state? |
| `end_to_end.trajectories_per_second` | What is the useful end-to-end training throughput? |
| `end_to_end.tpu_chip_seconds_per_trajectory` | How much fixed four-chip capacity is consumed per trained trajectory? |
| `rollout.collection_time_seconds` | How long from the first trajectory start until the last trajectory finishes for a policy step? |
| `rollout.trajectories_per_second` | How quickly does the stack complete whole episodes? |
| `rollout.output_tokens_per_second` | How many actual model output tokens are completed per rollout-collection second? |
| `rollout.trajectory_latency_seconds` | What are median, p90, and p99 whole-episode latencies? |
| `rollout.model_call_latency_seconds` | What are median, p90, and p99 sampler-call latencies independent of episode turn count? |
| `rollout.output_tokens_per_trajectory` | Did one stack become faster only because it produced shorter trajectories? |
| `rollout.average_*_per_model_call` | Are prompt and output token workloads per sampler call aligned? |
| `rollout.average_turns_per_trajectory`, reward, and outcome percent | Are episode behavior and quality sufficiently comparable to interpret throughput? |
| `rollout.average_time_per_trajectory_seconds` | Is episode time spent in model generation, environment calls, reward, or agent/parsing code? |
| `pipeline.rollout_start_delay_seconds` | How much of the step passes before rollout work begins? |
| `pipeline.post_rollout_tail_seconds` | How much training, metrics, or sync work remains after the final trajectory? |
| `pipeline.training_overlap_with_rollout_percent` | How much of the rollout collection window overlaps trainer API calls? This is diagnostic; higher is not automatically better if device contention increases. |
| `api_calls.<operation>.calls_per_step` | Does distributed fragment equivalent work into more control-plane or trainer calls? |
| `api_calls.<operation>.host_time_per_step_seconds` | How much host-observed time per step is associated with this operation? |
| `api_calls.<operation>.median/p90_call_time_milliseconds` | Is the operation consistently slow or dominated by tail latency? |
| `training_input` and `actor_log_probs_input` | Are trajectories per call, padded token slots per step, and tensor shapes identical? |
| `training_metrics` | Do steady-state reward, length, clipping, loss, and staleness diagnostics remain comparable? Values are medians under their original metric names. |
| `total_run_time_seconds` | What is total operational time including startup, compilation, steady state, and shutdown? |

- `events.jsonl` records training, actor-log-probability, and weight-sync API
  spans on a monotonic clock. Distributed also records rollout dispatch, rollout
  polling, worker RPC, metrics fetch, and checkpoint-save spans. Training events
  include actual prompt/completion shapes without copying TPU tensor contents
  to CPU. The distributed rollout worker writes `events-rollout-*.jsonl` so
  separate processes never share a file.
- Both implementations instrument the same `TrajectoryCollectEngine.collect`
  boundary. A `rollout_trajectory` event records a complete FrozenLake episode:
  start/end time, policy step, turns, prompt/model-output/environment token
  counts, and environment/reward wall time. Every actual model call emits a
  `generation` event. A single-turn rollout emits one generation call; a
  multi-turn rollout may emit several. Token counts come from sampler outputs,
  never from the configured response limit.
- A `training_step` runs from one successful weight-sync return to the next.
  An initial sync with no trained trajectories is not a step. At least one
  warmup step is required. The final step's sync is included; shutdown and
  logger close are excluded from steady state.
- The hooks do not insert per-microbatch `block_until_ready` calls, so existing
  overlap is preserved. API time includes RPC queueing, compute, return, and
  existing framework synchronization; it is not TPU kernel time. Agentic may
  still perform an asynchronous anchor snapshot after sync. Consecutive
  sync-to-sync intervals include cross-step work, while a TPU trace is required
  to diagnose the tail precisely.
- `summary.json` has one compact `end_to_end` section: sync-inclusive
  steady-state step time, discarded warmup-step time, trajectories/s, and TPU
  chip-seconds/trajectory. Timing objects contain a median and per-step values;
  min/max and measured-step count are omitted because they are directly
  derivable.
- The `rollout` section defines one rollout-collection window from the first
  episode collector start to the final collector completion. It reports
  collection time, trajectories/s, output tokens/s, trajectory and model-call
  latency, output-token distribution, average turns, and a per-trajectory time
  breakdown for model generation, environment, reward, and agent/parsing work.
  Average reward and trajectory outcome percentages are parity checks, not
  performance claims.
- `pipeline` places rollout collection inside the common sync-to-sync step. It
  reports delay before the first rollout starts, time after the final rollout
  until sync completes, and the percentage of rollout collection overlapped by
  training calls. On this single-host benchmark, all processes use the same
  monotonic system clock.
- `training_input` and `actor_log_probs_input` report trajectories per call,
  padded token slots per step, and unique tensor shapes. These fields establish
  compute-shape parity and expose accidental microbatch fragmentation without
  copying device arrays to the host.
- Only one output-token throughput is reported. Its denominator is the complete
  rollout-collection window. Prompt-token throughput and generation-window
  throughput were removed because they are easy to misinterpret as isolated
  prefill or decode performance.
- The current vLLM result API does not expose TTFT, queue time, or separate
  prefill/decode wall time. They are documented here instead of being emitted
  as empty metrics. Use a TPU or vLLM trace for kernel-level decomposition.
- Host instrumentation also cannot separate TPU kernel time, device
  utilization, HBM peak, collective/DMA time, cloudpickle serialization, and
  physical RPC bytes. The benchmark deliberately does not invent proxies for
  them. Capture the same TPU-profiler window for both modes after the host-side
  comparison identifies the stage that needs kernel-level attribution.
- `total_run_time_seconds` covers the entire child process, including startup,
  compilation, and shutdown. Do not compare it with steady-state step time.
  The distributed launcher also has a worker cleanup grace period.
- `api_calls` reports calls per step, host time per step, median call latency,
  and p90 call latency for clearly named operations such as `training`,
  `actor_log_probs`, `weight_sync`, `rollout_dispatch`, and
  `worker_rpc_fwd_bwd`. Distributed worker RPCs are split by remote method so
  serialization/control overhead is not hidden in one aggregate. Calls overlap:
  worker RPC is nested inside higher-level operations and rollout polling
  includes wait time. Do not sum these values or label polling time as waste.
- `dataset.json` stores a SHA256 over ordered input data. The manifest stores
  configuration, Git revision/diff fingerprints, dependency versions, model
  metadata hashes, and shard size/mtime. It does not hash all weight contents;
  verify checkpoint identity separately after copying across machines.
- `metric` events preserve raw reward/length/clip/loss/staleness metric names and
  steps. `training_metrics` reports each steady-state median and the comparison
  includes matching names from both stacks. The learners can use different
  aggregation and asynchronous flush boundaries, so similar names still need
  semantic review before drawing a quality conclusion.

The report rejects failed runs, missing steps, mismatched trajectory counts, and
differences in configuration, data hash, or software versions. These checks
establish run integrity and configuration parity, not training-quality or
numerical equivalence. Every step must train exactly B*G trajectories;
packing/replay accounting is not supported in this initial harness.

Each run keeps its own `summary.json`; the `report` command emits only the
direct comparison, avoiding a second copy of every run metric. Multiple
repetitions are reduced by taking the median run value within each mode. Every
comparison entry contains `agentic`, `dist`,
`dist_minus_agentic`, and `dist_relative_to_agentic_percent`.
`lower_is_better` states how to interpret the value: it is true for time/cost,
false for throughput, and null for diagnostic or workload-parity values such as
overlap, token count, reward, turns, shapes, and calls per step. A
distributed-only API has a null agentic value rather than being incorrectly
treated as zero work.
`interpretation` makes the result explicit: `dist_faster`, `dist_slower`,
`same`, `different_not_ranked` for parity/diagnostic values, or
`one_stack_only` for a distributed-only control-plane operation.
`parity_checks` separately confirms exact training and actor-logps tensor-shape,
trajectories-per-call, and padded-token-volume agreement.

A final results table should include three-run median end-to-end step time,
trajectories/s, TPU chip-seconds/trajectory, rollout collection time, rollout
output tokens/s, training/logps/sync API time, RPC counts, output length, clip
ratio, reward, and OOM/failure count. Define infrastructure speedup as agentic
median divided by distributed median. If time improves while reward or output
length changes materially, limit the conclusion to throughput for that online
workload; do not claim equal training quality.

## Code and validation

`frozenlake.py` generates configuration, isolates output/cache paths, launches
processes, handles timeouts, and produces reports. `runtime.py` installs
generic tracing hooks only when `TUNIX_BENCHMARK_DIR` is set.
`frozenlake_agentic.py` reuses the agentic CLI and the same ordered map data as
distributed. Normal recipe execution does not install any benchmark hooks.

CPU-only validation (no JAX, TPU, or pytest required):

```bash
python3 tests/experimental/examples/rl_efficiency_benchmark/frozenlake_test.py
```

The tests cover real batch-unit conversion, topology, sync-inclusive timing,
rejection of failed/incomplete runs, dry runs for both modes, ordered dataset
hashing, rollout token/latency aggregation, warmup-filtered throughput, failed
sync handling, and preservation of sync/async hook return values and
exceptions. TPU execution, HBM behavior, and final efficiency results must be
measured on the target VM.
