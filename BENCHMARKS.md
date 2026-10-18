# Benchmarks - the numbers behind this build

## Method

* Every claim is at least two repeats, most points four. On this host a single
  measurement proves nothing: decode moves 10-20 % run to run, and before patch
  06 the cold start was bimodal (190 s to over 10 minutes with one configuration).
* The first request after a start is a Triton/JIT spike and is discarded.
* Benchmarks use an anti-cache salt so repeated prompts do not hit the prefix
  cache (the bench script of the recipe project; ours is the same idea).
* Engine start = time from process launch to the first HTTP 200 on `/health`. It
  is the slowest PP rank, so stage imbalance is pure added time.
* `Checkpoint size: 76.05 GiB` in the log counts files, not tensors - real I/O is
  measured from `/proc/diskstats` samples, not from that line.
* `Loading weights took` is printed by one rank only (`info_once`); subtracting
  it from `Model loading took` means nothing.

## Engine start (patches 05/05b-05d and 06)

| configuration | start | page cache peak | samples with swap | NVMe `await` |
|---|---|---|---|---|
| mmap (default), two baselines | 680 s / 628 s | 22.5 GiB | 66-69 % | 261-269 ms |
| `VLLM_PLE_NONPLE_LOADER=instanttensor` | **166 s / 150 s** | 5.35 GiB | 2-3 % | 6-16 ms |
| `--safetensors-load-strategy=eager` (rejected) | 491 s vs 343 s the same day | - | 63 GiB swapped on 21 GiB RAM | - |

Bytes read from disk during start dropped from 124 GiB to 96 GiB with the PP
weight filter (05/05b-05d): each stage reads only its own layers. Time did not
improve from that alone - it is the enabler for patch 06 and it removes 28 GiB of
pointless I/O from the drive that also holds the PLE table.

Caveat recorded here: one start of a variant took 760 s while the loader mode was
correct. Memory was clean (132 MiB swapped and not growing, 11.6 GiB free, page
cache 7.5 GiB), but the drive was at 84.7 % `util` with `await` 12.1 s and queue
depth 1274, against 8.1 % / 0.12 s / 236 MB/s in a fast start. Repeats by the
same path gave 150/160/164 s, so that episode was the carrier, not the build -
which is why a slow start should be diagnosed with `iostat`/`sar -d` before
blaming memory.

## Batch tokens (four repeats per point)

| `MAX_BATCH_TOKENS` | prefill @32 768 | prefill @8 192 | decode | KV cache |
|---|---|---|---|---|
| 2048 | 7 522 tok/s | 6 746 tok/s | 97 tok/s | 967 470 tok |
| **4096 (used)** | **8 330 tok/s** | **6 915 tok/s** | **121 tok/s** | 960 229 tok (-0.7 %) |
| 8192 | 8 132 tok/s | 5 910 tok/s | - | -6.3 % KV |

The KV column was measured at `GPU_MEMORY=0.95`. The production profile uses
**0.90**, which gives **744 431** tokens at ctx 262 144 with a BF16 KV cache and
**1 352 663** with the INT8 cache of patch 16 (see "KV cache quantization on
SM80"); the fraction was forced by the free-VRAM plateau of a long-running engine
(see below), not by KV arithmetic.

vLLM logs a hint that `max_num_scheduled_tokens` "should" be 2048 under
speculative decoding; on this hardware 4096 dominates on every axis measured, so
the hint points away from the optimum here.

## Pipeline stage split

| split | rank 0 / rank 1 weights | VRAM per card | KV cache | concurrency |
|---|---|---|---|---|
| 24,24 | 39.49 / 45.53 GiB | 53.9 / 62.1 GiB | 840 019 tok | 3.20x |
| **26,22 (used)** | 42.49 / 42.52 GiB | 60.4 / 61.0 GiB | **967 470 tok** (+15 %) | 3.69x |

## The drive (the same NVMe that holds the checkpoint, the PLE table and the OS)

Measured on the checkpoint files themselves, with no engine memory pressure:

| path | throughput |
|---|---|
| `dd iflag=direct`, 1 MiB requests | 1.66 GB/s |
| mmap read (`safetensors` default) | 2.66 GB/s |
| `instanttensor` | 1.01 GB/s |
| `fastsafetensors`, no GDS | 0.98 GB/s |
| `fastsafetensors`, GDS | 1.01 GB/s (GDS is unusable on these cards: no `/dev/nvidia-fs0`) |

So mmap is faster than direct I/O in isolation; the production win of patch 06
came from *not* keeping 23 GiB of page cache resident, not from raw throughput.
During decode the PLE path keeps about 940 random 4 KiB reads per second going
(peak 37 742 r/s, 159 MB/s, mean `r_await` 0.97 ms).

## PLE row cache in host RAM (`PLE_SSD_CACHE_MB`, `PLE_SSD_OFFLOAD=auto`)

The SSD iterator keeps every row it has read in a per-worker LRU dict
(`additional_config` key `ple_ssd_cache_mb`). The rows are plain process
memory and nothing is reserved up front: the cache fills as traffic touches
rows. On a host with enough RAM this means steady-state inference stops
reading the PLE file at all - only the start-up weight load touches the drive.

Measured on a 91 GiB-RAM host (four repeats per point, `iostat` medians for
the system NVMe, warm-up excluded in the steady column):

| `PLE_SSD_CACHE_MB` | decode tok/s | prefill @2k/@8k/@32k | bench reads | steady reads |
|---:|---:|---:|---:|---:|
| 512 | 90.6–99.2 | 4541 / 6216 / 7359 | 1440 rkB/s | sustained |
| 24576 | 107.8–117.3 | 4687–4804 / 6527–6585 / 8041–8073 | 1160 rkB/s | near zero |
| 56320 | 109.9–115.9 | 4666–4703 / 6425–6509 / 7825–7930 | 1208 rkB/s (warm-up burst) | **0 r/s** |

A cache larger than the table is not required and on this host not possible:
the BF16 table is 95.37 GiB, while a 56320 MiB cache holds ~11.2 M of its
20 M rows. It still goes to zero reads because only the *working set* of real
traffic has to fit, and a 55 GiB cache holds it with wide margin. `auto`
turns free RAM into exactly this effect: `serve.sh` reads `MemAvailable` at
startup and picks the tier (>=60 GiB -> 56320, 40-60 -> 24576, <40 -> 512),
always overriding `PLE_SSD_CACHE_MB`; a 30 GiB host therefore runs exactly
the 512 MiB configuration the published numbers above were taken with.
Mode `0` (upstream CPU offload) is never auto-selected: it allocates the
whole table on the CUDA device and OOMs on 64 GiB cards.

## GPU memory over time (`GPU_MEMORY`, the reason for 0.90)

The free-VRAM plateau of a running engine falls: a fresh start at
`GPU_MEMORY=0.92` left ~5.9 / 3.9 GiB free, while after 11 hours of continuous
service the same engine had 2531 / 1875 MiB left - below the safety margin a
co-resident process needs. The production profile therefore uses **0.90**, which
leaves 7479 / 6905 MiB free right after a start.

## End-to-end acceptance

Prompt: "Write a most beautiful cat themed Conway's Game of Life in html",
65 536 token quota, thinking mode on (the answer is mostly reasoning tokens, so a
small quota yields an empty `content` and `finish=length`).

| run | result |
|---|---|
| batch 2048, earlier profile | 567 s |
| **batch 4096, this build, cards exclusive** | **45 507 tokens in 358 s (about 127 tok/s)**, `finish=stop`, valid HTML, zero `NVRM: Xid` before and after |

## KV cache quantization on SM80 (patch 16)

`sm_80` has no FP8 tensor cores, and Triton on `sm_80` has no `fp8e4nv` dtype
either, so an FP8 KV cache has to be emulated on these cards. INT8 does not:
int8 -> bf16 is a native widening, and a scale per (token, head) costs 16 bytes
per token per layer (two fp32 scales for the two KV heads, about +1.6 % of the
pool). Those scales live in their own tensors, outside the KV page, so the page
size - and with it the hybrid block-size arithmetic - does not change.

One tree, one profile (`GPU_MEMORY=0.90`, MTP=2, `MAX_BATCH_TOKENS=4096`), the
engine started per variant, four repeats per prompt length, the first request of a
start discarded as a JIT spike:

| KV cache | pool at ctx 262 144 | prefill @2 048 / @8 192 / @32 768 | decode | start |
|---|---|---|---|---|
| `bfloat16` (before patch 16) | 744 431 tok (2.84x) | 5 050 / 6 920 / 8 300 tok/s | ~125 tok/s | 258-367 s |
| `fp8_e4m3`, emulated (`fp8e4b15`) | 1 357 905 tok (5.18x) | 3 270 / 5 650 / 7 855 tok/s | ~80 tok/s (-36 %) | 310 s |
| **`int8_per_token_head` (patch 16, shipped)** | **1 352 663 tok (5.16x)** | 4 990 / 6 750 / 8 250 tok/s | **~123 tok/s (-1.5 %)** | 243 s |

The middle row is why the shipped build does not use FP8 for the KV cache. The
memory is exactly as cheap as INT8's; what costs is the dequantization the
emulation forces (`e4b15` -> fp32 -> bf16, and it must pass through fp32 because
Triton maps an `fp8e4b15` cast to bf16 onto fp16, after which `tl.dot` refuses a
bf16 x fp16 pair). With twelve fully-attentive layers and two KV heads the KV
traffic is small next to the decode path, so that emulation is pure loss.

Quality was checked with a needle-in-a-haystack probe - one secret passphrase
buried in a ~34 000 token document, depths 0.1 / 0.5 / 0.9, greedy: **3/3** on the
shipped INT8 cache. The end-to-end acceptance run above also completes on INT8 KV
with `finish=stop` and zero `NVRM: Xid` before and after.

## Refusal abliteration (patch 14)

Behaviour is probed with eight deliberately borderline prompts plus two neutral
controls, greedy decoding, thinking off, refusal detected by a regular expression
at the **beginning** of the answer (with typographic-apostrophe normalization).
Run the probe in both states of the engine (`QWEN38_ABLITERATION` set and unset).
Throughput on a benign prompt is unchanged within output-length variance (268 s
against 397 s on the acceptance prompt is a different generated length, not a
speed difference), because the projection is a rank-1 update on activations that
are already in cache - no weight is touched, and INT4 experts stay as they are.

## Platform limits measured on this machine

| probe | reading |
|---|---|
| CPU, aggregate during two 16 384-token generations | 13.6 % mean, 18.5 % peak of 16 threads |
| Busiest engine threads | 68-78 % of one core |
| GPU link | Gen2 x4 on both cards (about 2 GB/s), no peer-to-peer |
| NVMe temperature under load | peak 50.9 °C against 89.9/94.9 °C thresholds - no throttling observed |