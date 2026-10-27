# DeepSeek-V4.1-Flash remote DSpark drafter: DGX Spark GB10 to x86 RTX 3090 over RDMA

Move DSpark drafting from two DGX Spark nodes to an x86 RTX 3090 worker, preserving target-side verification. Capacity mode removes the local drafter and makes room for a larger target KV pool; the production configuration now also uses async rounds.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

<!-- pli:update:2026-10-27:start -->

**Update 2026-10-27:** the full DSV41 source is now public — [dsv41-offload](https://github.com/ProprietaryLegal/dsv41-offload) (every component; `production`, `main` and per-experiment `lane/*` branches), with focused repos for the [measurement harness](https://github.com/ProprietaryLegal/dsv41-window-harness) and [sampled drafting](https://github.com/ProprietaryLegal/dsv41-sampled-drafting). Status and lessons: [2026-10-27 post](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/dgx-spark-offload/2026-10-27-status-and-code.md).

<!-- pli:update:2026-10-27:end -->


## Status

2026-10-23 — production: one-card RTX 3090 worker, capacity mode, remote Engram, expanded KV and async process-proxy rounds; TP2 promotion is pending.

The capacity and failure gates passed on the tested image and checkpoint. Historical synchronous measurements below establish the memory trade; they are not a new benchmark of the async production switch. The async sibling contains the post-switch checks.

## What it is

DeepSeek-V4.1-Flash-1017 ships a 3-layer DSpark drafter (mtp.0–2: sliding window 128, block 5, Markov rank-256 head, 128
experts top-3). The DGX Spark recipe runs it on the Sparks, next to the 40-layer target. All-in (weights, CUDA graphs,
buffers, draft KV) it costs about 5 GiB of unified memory per Spark.

This package moves the drafter to an RTX 3090 on the x86 host:

- **Worker** (`code/dspark/worker/`): a standalone PyTorch drafter.
  - Bit-exact EXL3 decoding of the drafter's own K2 experts, K5 dense layers and K6 head. These K labels identify EXL3 quantization levels.
  - CUDA graphs for 1–4 requests and every n_out, with a 128-row context window per slot.
  - Speaks the CONTRACT v1.8 wire protocol over the [ofl](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) RDMA
    transport.
  - Safety:
    - It refuses tensors not covered by its hash-verified receipt.
    - It validates positions, generations and replies.
    - It exits for a systemd restart if its CUDA backend dies.
- **Spark overlay** (`code/dspark/spark/`): a `remote_dspark` speculator package, plus one guarded branch in vLLM's
  `spec_decode/__init__.py`. When `VLLM_DSPARK_REMOTE` is unset, the image path runs unchanged.
  - Rank 0 gathers the target's tap rows (layers 37/38/39) with one fused Triton kernel, sends them, and broadcasts the
    returned drafts and confidences to rank 1.
  - Replies are validated (token ids, confidence range, shapes, generations) before the engine consumes them. Failures give
    zero drafts on both ranks.
  - A bounded circuit breaker trips after 3 failures, cools down for 256 rounds, then probes.
  - The reconnect backoff is capped.
- **Policies:**
  - `shadow` compares remote against local drafts.
  - `resilience` falls back to the local drafter.
  - `capacity` removes the drafter from the Sparks entirely: no draft weights, Markov cache, draft graphs or draft KV layers.

**The memory win comes from `capacity`.** The deployment spends the freed unified memory on the target KV pool; the measured allocation is below.

The receipt checks content hashes and loader destination coverage. It is a trust anchor, not a cryptographic signature or authentication protocol. A fatal CUDA backend exits so the service manager can restart the worker. All ranks consume the same validated proposal and fallback decision.

## Why

Keeping DSpark's acceptance while freeing its GB10 memory buys context room and concurrency. Removing speculation saves memory but loses substantial decode throughput. Capacity mode keeps remote drafting and spends the memory on target KV. Remote Engram independently improves prompt reading; its gains must not be credited to the drafter alone.

## When to use it

Use it when context capacity or concurrent long requests matter, and an x86 GPU and low-latency direct link are available. The memory expansion has a measured capacity benefit even where synchronous single-stream decode is slower.

## When not to use it

Do not choose synchronous remote drafting solely for lower single-request latency. Do not assume a larger KV pool increases the qualified model context length. If the worker fails in capacity mode, requests continue target-only at lower delivered-token throughput. Use resilience policy if retaining a local drafter is preferable to reclaiming its memory.

## Requirements

An RTX 3090 with sufficient free VRAM; the receipt-verified drafter tensors and Markov state from your authorized model copy; PyTorch, Triton and matching exllamav3 kernels; the ofl transport on both ends; and the tested `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5` target image. The overlay checks the image dispatch file's digest. The deployment is two DGX Spark (GB10) nodes plus an x86 offload host with RTX 3090s.

The observed one-card working set is 8.1–8.3 GiB including a 2 GiB Markov cache. That is runtime memory, distinct from the packed tensor inventory. Inspect [TENSORS.md](code/dspark/TENSORS.md) before provisioning; model files are not included here.

## Measured effect

Model: the [EXL3 2.75 bpw build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark). A is the stock recipe with the drafter on GB10; B adds remote Engram; C moves only the drafter in capacity mode with the original KV budget; D-kv6 combines both offloads with expanded KV. Cold prefix cache unless a row says warm. These historical comparisons were collected across 2026-10-22–2026-10-23.

### Why preserve speculation

On 2026-10-22, both DGX Spark GB10 nodes ran the stock configuration with no remote worker, at concurrency 1. The DSpark cold workloads had three repeats; the target-only diagnostics had two repeats each. Context is in the row; memory is the serving observation.

| | A (DSpark) | A, speculation off |
|---|---|---|
| short, decode tok/s | 55.96 ± 1.65 | 41.9 |
| 32K, decode tok/s | 64.57 ± 0.80 | 41.0 |
| MemAvailable per Spark while serving | 5–7 GiB | 11–12 GiB |

The rates above show the delivered-token benefit of speculation at both contexts: +57 % at 32K, at about 5 GiB per Spark all-in. A target-only step returns one token, so a lower time per target-only step is not a delivered-token speed gain.

### Drafter alone, synchronous capacity mode

These synchronous single-stream tests used two DGX Spark GB10 nodes and an x86 one-card RTX 3090 worker across 2026-10-22–2026-10-23, with temperature 1, high thinking and 1,500-token budgets. Cold workloads had three repeats; the warm 32K workload had two. A is the stock local drafter; C moves only the drafter in capacity mode.

| context | A ms/round | C ms/round | change | A TTFT | C TTFT | A stream tok/s | C stream tok/s |
|---|---|---|---|---|---|---|---|
| short | 37.2 | 40.0 | +7.6 % | 0.36 s | 0.38 s | 55.96 ± 1.65 | 50.25 ± 0.67 |
| 32K cold | 40.1 | 42.9 | +6.9 % | 15.53 ± 0.12 s | 15.01 ± 0.83 s | 64.57 ± 0.80 | 59.25 ± 1.67 |
| 32K warm | 40.0 | 43.0 | +7.5 % | 0.53 s | 0.56 s | 63.72 ± 4.87 | 61.33 ± 0.44 |
| 150K cold | 39.2 | 42.7 | +8.9 % | 73.62 ± 0.11 s | 74.46 ± 2.80 s | 56.10 ± 1.07 | 54.83 ± 2.12 |

The same hardware and synchronous A/C configurations were measured across 2026-10-22–2026-10-23. Startup values are allocation logs per rank, not context or concurrency tests; rate rows report aggregate delivered tokens per second.

| Measurement | Stock A | Capacity C | Conditions |
|---|---|---|---|
| Model load per rank | 94.92 GiB | 92.4 GiB | Startup: one log per rank; 2026-10-22. |
| KV budget per rank | 3.75 GiB | 3.75 GiB | Same startup logs. |
| KV token pool | 989,460 | 1,003,083 | Same startup logs. |
| MemAvailable per Spark | 5–7 GiB | 10 GiB | After the 150K stages; one suite. |
| 32K × 2 aggregate tok/s | 40.59 | 40.52 | Two groups of two requests. |
| 32K × 3 aggregate tok/s | 47.21 | 46.68 | Two groups of three requests. |
| 32K mixed aggregate tok/s | 60.48 | 59.40 | Two mixed groups: a 32K request arriving during two 8K decodes. |
| 150K mixed aggregate tok/s | 25.49 | 25.19 | Two mixed groups: a 150K request arriving during two 8K decodes. |

Synchronous C frees about 3–5 GiB per Spark but adds about 7–9 % per decode round at one stream; the concurrency cost is ≤ 2 %. Its round-trip p50 was 3.76 ms, including 3.44 ms in the worker, with 0 failed rounds and 0 fallbacks in that suite. The later async production configuration addresses part of that round cost.

### Combined offloads and expanded KV: D-kv6

Across 2026-10-22–2026-10-23, two DGX Spark GB10 nodes and the x86 one-card RTX 3090 worker compared stock A with combined offloads and expanded KV (D-kv6).

| Capacity | Stock A | D-kv6 | Conditions |
|---|---|---|---|
| KV pool per rank | 3.75 GiB | 6.0 GiB | Startup allocation per rank; one log per rank. |
| KV tokens | 989,460 | 1,604,926 (+62 %) | Same startup allocation logs. |
| Max. concurrency at 327,680 tokens/request | ~3.0× | 4.90× | Engine-reported capacity at startup; throughput was not measured by this figure. |
| MemAvailable per Spark while serving | 5–7 GiB | 5–8 GiB | Observed serving range across the suite. |

The historical synchronous comparison used the same model and hardware across 2026-10-22–2026-10-23, temperature 1, high thinking, 1,500-token budgets and cold prefixes. Single-stream rows use concurrency 1; concurrent and mixed workloads name their request counts. Stock A and Engram-only B cold workloads had three repeats. D-kv6 cold aggregates also had three runs per context: the initial 32K and 150K files had two each, plus one cold run in each warm-cache series. Separate cold 32K repeats had two runs; the separate cold 150K repeat had one. Each concurrent or mixed workload had two groups; 256K, 300K and four concurrent 300K workloads ran once each.

| | A | B | D-kv6 | D vs A |
|---|---|---|---|---|
| short, stream tok/s | 55.96 ± 1.65 | — | 51.07 ± 0.78 | −8.7 % |
| short, step p50 (direct timer) | — | — | 39.0–39.6 ms | |
| 32K cold TTFT | 15.53 ± 0.12 s | 14.70 ± 0.07 s | 14.81 ± 0.54 s (repeats 14.55 ± 0.01) | −4.6 % |
| 32K cold stream tok/s | 64.57 ± 0.80 | — | 60.82 ± 2.30 | −5.8 % |
| 32K task time (1,500 tokens) | 38.75 s | 37.71 s | 39.47 ± 0.77 s | +1.9 % |
| 150K cold TTFT | 73.62 ± 0.11 s | 69.44 ± 0.04 s | 68.79 ± 0.14 s | −6.6 % |
| 150K cold stream tok/s | 56.10 ± 1.07 | — | 54.23 ± 2.37 | −3.3 % |
| 150K task time | 100.35 ± 0.55 s | 95.17 ± 1.61 s | 96.46 ± 1.11 s | **−3.9 %** |
| 256K / 300K TTFT | not run | not run | 118.7 / 140.3 s | |
| 256K / 300K stream tok/s, step p50 | — | — | 51.6 / 52.2; 42.2 / 42.8 ms | |
| 32K × 2 / × 3 aggregate tok/s | 40.59 / 47.21 | — | 40.81 / 47.36 | +0.5 / +0.3 % |
| 32K mixed / 150K mixed aggregate tok/s | 60.48 / 25.49 | — | 59.34 / 25.81 | −1.9 / +1.3 % |
| **4 × 300K concurrent (cap4)** | does not fit (989,460 tokens) | does not fit | **all 4 completed**: TTFT 620–640 s (1.2 M tokens prefilled), 16.9–23.4 tok/s per stream, step p50 ~110 ms | capacity |

Each mixed group is a long request arriving while two 8K requests decode. The first-token and full-task gains include remote Engram. D-kv6 pays the synchronous drafter cost at one stream, stays near level under concurrency, and completes the long concurrent group that does not fit the baseline KV pool. The later async switch retains this allocation.

### Exactness and prefix reuse

The following exactness and prefix checks are separate from engine throughput measurements.

| Check | Result | Conditions |
|---|---|---|
| Shadow draft agreement by position | 0.99 / 0.97 / 0.96 / 0.95 / 0.93 | 2026-10-22; shadow mode, local GB10 vs one RTX 3090; short/8K/32K checks, one sequence per workload. Greedy proposal agreement, not sampled-output identity. |
| Prefix repeat | 36,016 prompt tokens; 35,968 cached | 2026-10-22; same hardware, concurrency 1; one repeat each under shadow and capacity, matching the stock cache count. |
| Remote Engram row exactness | 1,678,080 entries across both ranks; 0 bad rows | Reconciled 2026-10-23; Engram-only B, two GB10 nodes and x86 RAM server; both layers, TCP and RDMA, one suite per rank. |

The Engram module output was bit-identical at its boundary. Whole-engine greedy runs at longer prompts were already nondeterministic in stock-vs-stock controls, so those checks establish no offload-specific drift rather than universal deterministic output. The later async switch also passed its bounded fixed-prompt greedy comparison.

### Failure drill

One historical stop/restart drill used synchronous D-kv6 on two DGX Spark GB10 nodes and the x86 one-card RTX 3090 worker across 2026-10-22–2026-10-23. It used one request with a 6,000-token output budget; the prompt length was not recorded in the comparison summary.

| Event or check | Observation |
|---|---|
| Stop/restart worker | Stopped at +25 s, started at +56 s; request completed, 4,758 output tokens |
| Drafted → target-only → drafted | Step p50 39.6 → 27.7 ms; drafting resumed at +88 s |
| Failure accounting | 2,101 not-ready rounds, 1 timeout, 5 connect failures; breaker never tripped |
| Reconnect policy | Old cap 30 s; about 30 s recovery after the worker returned. Production cap 5 s is a setting, not a measured recovery time. |

### Link interference

Across 2026-10-22–2026-10-23, synchronous D-kv6 used the same two GB10 nodes and one-card RTX 3090 worker, with both direct links loaded toward the Sparks. Each row is one request, at concurrency 1. Loads are achieved per-link rates; context and the partial load duration are in the rows.

| run | load per link (achieved) | TTFT | step p50 | vs unloaded D repeats |
|---|---|---|---|---|
| 32K | 10 Gb/s (9.93) | 14.59 s | 42.8 ms | TTFT +0.3 %, step +1–2 % |
| 32K | 35 Gb/s (34.71) | 14.91 s | 42.6 ms | TTFT +2.5 %, step +1 % |
| 150K (load covers 60 s of the 69 s prefill) | 35 Gb/s (34.67) | 70.31 s | — | TTFT +2.1 % |

These were achieved bulk loads, not nominal interface speed. Near-saturating traffic costs about 2 % of prefill; ordinary offload traffic is far below the lower test load. No interface change was justified by this test.

### Current one-card async behavior

On 2026-10-23, both DGX Spark GB10 nodes used capacity mode, remote Engram and expanded KV with the x86 one-card RTX 3090 worker and the process proxy. The matched comparison and post-switch checks are distinct.

| Check | Result | Conditions |
|---|---|---|
| Step p50 change vs sync | −1.0 to −4.0 % | Matched synchronous/async comparison: short and 32K each three repeats, 150K two, concurrent 32K one group of three; temperature 1 and 1,500-token budgets. |
| Post-switch short requests | TTFT 0.37 / 0.38 s; step p50 38.4 / 38.5 ms | Production after short warm-up; concurrency 1, two repeats. |
| First post-switch 32K request | TTFT 16.24 s; step p50 40.5 ms | Production first long request; concurrency 1, one run, lazy long-prefill loads. |
| Production round counters | 2,048 ok; 0 failed; 0 watchdog fires; 0 respawns | One production counter snapshot across the post-switch checks. |

## Upstream claims

DeepSeek provides the model and DSpark drafter design. HeNryous and sudoingX provide the DGX Spark image and patch lineage, and vLLM provides the target runtime. No upstream speed claim is included in our measured tables. An upstream kernel capability or checkpoint size alone does not establish memory fit, draft agreement or engine throughput.

## Deploy

Prepare the receipt-verified model directory outside the repository. Build the architecture-matched ofl transport using [its Makefile](code/transport/Makefile), install its Python binding alongside the library, and set the worker's `DIRECT_LINK_IPS` allowlist to your direct-link addresses. The allowlist is explicit; supplying a new address on the command line does not bypass it.

For a local worker launch, from the repository root with your own endpoint supplied through `WORKER_BIND`, use an immutable deployment copy and a supervised service:

```sh
cd code/dspark
export CUDA_VISIBLE_DEVICES=GPU-DRAFTER0-UUID
export DSPARK_WORKER_GPU_UUID=GPU-DRAFTER0-UUID
export OFL_PY_DIR=/data/ofl
export OFL_LIB=/data/ofl/libofl.so
python -m worker.server --tp 1 --root /data/dspark \
  --ofl "$WORKER_BIND" --max-model-len 327680 \
  --status-file /data/run/dspark-worker.status.json
```

The model-length setting must match the engine; the parser allows up to 1,048,576, but that upper bound is not an end-to-end qualification. Make both Spark ranks use the same overlay, identities and policy. Mount `code/dspark/spark/remote_dspark` under vLLM's `spec_decode/remote_dspark`, its guarded dispatch file at the corresponding `spec_decode/__init__.py`, and the transport at `/data/ofl` in the container. Configure these persistent engine variables using your endpoint and content identities:

```sh
export VLLM_DSPARK_REMOTE="$REMOTE_ENDPOINT"
export VLLM_DSPARK_REMOTE_BIND="$SPARK_BIND"
export VLLM_DSPARK_REMOTE_POLICY=capacity
export VLLM_DSPARK_REMOTE_IDENTITY="$DRAFTER_IDENTITY"
export VLLM_DSPARK_REMOTE_MARKOV_IDENTITY="$MARKOV_IDENTITY"
export VLLM_DSPARK_REMOTE_ROUND_MS="$VERIFY_ONLY_ROUND_COSTS"
export VLLM_DSPARK_REMOTE_DRAFT_MS="$REMOTE_ROUND_COST"
export VLLM_DSPARK_REMOTE_TIMEOUT_MS=50
export VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS=5000
```

`REMOTE_ENDPOINT` has the form `o1rdma://<address>:<port>`. ofl TCP and the contract TCP stub are different wire protocols. Calibrate both cost inputs on your system; do not reuse a local-drafter table. After measuring headroom, the tested expanded allocation is `--kv-cache-memory-bytes 6442450944` per rank. Enable the async process proxy using the [async package](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff), then restart with your normal engine launcher.

Confirm worker identity at HELLO, capacity-policy logs, KV allocation, fresh success/fallback counters and recovery after an idle restart. Run probes only while the engine is disconnected: a new HELLO session discards its old worker windows. Enable service restart-on-fatal and verify the unit returns after a reboot.

## Rollback

Remove every `VLLM_DSPARK_REMOTE` variable from the persistent engine configuration on both ranks, remove the remote-DSpark overlay mounts, restore the original KV size, and restart. For this baseline the original allocation was `--kv-cache-memory-bytes 4026531840`. The image's local drafter then returns. Remote Engram is independent; retaining it gives the Engram-only B configuration.

To disable only async, unset `VLLM_DSPARK_REMOTE_ASYNC` and restart with the remote endpoint retained. This restores synchronous remote rounds without changing capacity mode.

## Limits and known issues

The systemd units and staging helpers assume the original deployment layout and pinned binaries. Configure their paths and pins for your installation; `stage_overlay.sh` requires changes for this standalone layout. [INTEGRATION.md](code/dspark/INTEGRATION.md) contains historical thread-proxy advice; use the async sibling for current process-proxy settings.

One worker serves one engine session; only rank 0 connects and rank 1 receives the broadcast. The tested worker supports up to 4 concurrent requests. Draft proposals are greedy; unsupported draft sampling and parallelism modes are refused at startup. Target sampling is still controlled by the engine.

The hash receipt does not secure a mutable model directory against a concurrent hostile writer; keep model files and receipts immutable during loading. The worker checks identities and refuses bad frames and replies, but the transport is intended for a trusted direct-link deployment. Capture tooling is optional diagnostic instrumentation: do not leave it enabled for timing or expose its outputs. Saved capture data and model weights are absent from this release.

## Repository layout

| Path | Purpose |
|---|---|
| [code/dspark/worker](code/dspark/worker) | Drafter loader/model, stateful contract server, fatal-backend handling and service templates. |
| [code/dspark/worker/exl3](code/dspark/worker/exl3) | Vendored plugin wrappers, decoding checks and optional exllamav3 accumulation patch. |
| [code/dspark/worker/run](code/dspark/worker/run) | Deployment-specific start/status/stop, pinned transport and production helpers. |
| [code/dspark/worker/tools](code/dspark/worker/tools) | Benchmarks, capture parity, conformance and optional donor experiments; donor sources are not bundled. |
| [code/dspark/spark/remote_dspark](code/dspark/spark/remote_dspark) | Spark-side gather, wire client, validation, policies, async handoff and recovery. |
| [code/dspark/spark/overlay](code/dspark/spark/overlay) | Guarded vLLM remote-speculator dispatch. |
| [code/dspark/capture](code/dspark/capture) | Instrumented vLLM DSpark speculator and original-file hash for privately collecting diagnostic captures. |
| [code/transport](code/transport) | ofl C/C++ transport, Python binding, probes and Makefile. |
| [code/dspark/INTEGRATION.md](code/dspark/INTEGRATION.md) · [CONTRACT.md](code/dspark/CONTRACT.md) · [TENSORS.md](code/dspark/TENSORS.md) | Integration history, wire/state rules and tensor inventory. |
| [code/dspark/worker/tests](code/dspark/worker/tests) · [Spark tests](code/dspark/spark/tests) | Contract/identity regressions and simulated engine tests. |

## Tests

With Python, pytest, NumPy, torch and Triton installed, run these from the repository root. CPU tests use no live engine or worker:

```sh
cd code/dspark
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider -q worker/tests --ignore=worker/tests/test_fused.py
cd spark
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider -q tests
```

`test_fused.py` contains GPU tests without no-device skips; its exclusion is the documented publication baseline, not a CPU pass for those kernels. Other device cases skip when CUDA is hidden. Native hostflag tests need a C compiler. Optional ofl tests require a built library.

On 2026-10-23 the initial publication recorded worker 39 passed and Spark 167 passed / 2 skipped. The updated publication recorded Spark 221 passed / 2 skipped and the shared worker suite 64 passed / 40 skipped with the GPU-only file excluded. The later production Spark check was 223 passed / 1 skipped. These are historical results from different trees and dependency setups; GPU parity and RDMA require separate qualification.

## Credits and licences

Our code is Apache-2.0. The copied `vllm_exl3_moe` plugin wrappers are Apache-2.0, as recorded in [VENDORED.md](code/dspark/worker/exl3/VENDORED.md); they call turboderp/exllamav3's MIT extension. The included patch modifies that extension and its build script. The dispatch and capture overlays retain vLLM Apache-2.0 notices. HeNryous and sudoingX supply the image and patch lineage, and DeepSeek supplies the model and DSpark drafter. See [NOTICE](NOTICE) for the component inventory and bundled third-party licence texts.

## Related

- dsv41-async-hostflag-handoff: [GitHub](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).
- dsv41-dspark-drafter-tp2-nvlink: [GitHub](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).
- dsv41-remote-engram-rdma: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- ofl-rdma-transport: [GitHub](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [Hugging Face](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
- dsv41-longctx-bench-tools: [GitHub](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
- dsv41-native-1017-drafter-sm86: [GitHub](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
- dsv41-spark-offload-findings: [GitHub](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).
- DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload: [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
- dsv41-exact-acceptance-ceiling: [GitHub](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
- Image forks: [HeNryous lineage](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [sudoingX lineage](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41).
- [Research hub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload) · [Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
