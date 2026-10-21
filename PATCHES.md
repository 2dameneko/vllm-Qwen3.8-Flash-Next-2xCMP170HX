# PATCHES.md - every change against "upstream + recipe"

Three layers, applied in this order:

```
vllm-project/vllm @ a5a30471f  (0.29.1rc1.dev402+ga5a30471f)
  + base/qwen38-ple-ssd.patch        the third-party recipe (iIIusi0n, Apache-2.0)
    + patches/01 .. 17               our changes
      + serve/serve.sh, serve/awq-pp2.env
```

The scripts in `patches/` are numbered in **apply order** (the order `apply.sh`
uses; later patches depend on text inserted by earlier ones). `diffs/` has the
same changes as unified diffs against a clean base, and `overlay/` is the
resulting end state.

## Tree patches

| file | file(s) touched | what it does | origin |
|---|---|---|---|
| `01_ple_ssd_tp.py` | `nvidia/ple_ssd.py` | allows `TP>1` with PLE offload: every TP rank keeps its own `PLESSDTable` and reads the table file independently. `DP=1` stays required (replicas would each open their own handle) | ours - the recipe hard-required `tensor_parallel_size == 1`, so a two-card layout could not start |
| `02_ple_ssd_mixed.py` | `nvidia/ple_ssd.py` | accepts **mixed** safetensors shards (PLE tensors next to ordinary weights): PLE is yielded as meta (the SSD table reads it by `data_offsets`), the rest of the shard is read with plain `safe_open` | ours - real checkpoints interleave PLE and normal tensors in a tail shard |
| `03_ple_ssd_fp8.py` | `nvidia/ple_ssd.py` | **reserve path** for an FP8 PLE table: item size from the shard dtype, the per-tensor `weight_scale` read from the checkpoint, dequantization to BF16 on the host | ours; validated 2026-10-18 with a converted FP8-PLE checkpoint (47.68 GiB vs 95.37 GiB BF16): quality probe passed, decode on par with BF16, prefill 5-7.5 % slower, start 631 s vs 907 s; production keeps the BF16 table |
| `04_ple_ssd_fp8_iter.py` | `nvidia/ple_ssd.py` | **reserve path**: the PLE shard iterator accepts `F8_E4M3` tensors and yields them as `float8_e4m3fn` meta tensors instead of failing on a BF16-only check | ours; validated 2026-10-18 with a converted FP8-PLE checkpoint (47.68 GiB vs 95.37 GiB BF16): quality probe passed, decode on par with BF16, prefill 5-7.5 % slower, start 631 s vs 907 s; production keeps the BF16 table |
| `05_ple_ssd_ppfilter.py` | `nvidia/ple_ssd.py` | passes the stage's layer range into the PLE iterator | ours (part of the PP weight filter) |
| `05b_pp_filter_loader.py` | `model_loader/default_loader.py` | hands the stage's layer range to the safetensors iterator and to the parallel loaders | ours |
| `05c_pp_filter_ep.py` | `model_loader/ep_weight_filter.py` | the filter itself: the local layer range from `get_pp_indices` (so `VLLM_PP_LAYER_PARTITION` is respected), `should_skip_weight()` and the `mtp.*` rule (last stage only) | ours |
| `05d_pp_filter_weights.py` | `model_loader/weight_utils.py` | the safetensors iterator skips foreign layers by name before reading them | ours |
| `06_ple_ssd_nonple_loader.py` | `nvidia/ple_ssd.py` | **the big one**: with `ple_ssd_offload` the PLE iterator captures the whole weight stream, so the `instanttensor`/`fastsafetensors` branches of the loader were unreachable and non-PLE files went through mmap. `VLLM_PLE_NONPLE_LOADER` = `safetensors` (mmap, default) / `instanttensor` / `fastsafetensors`, both single-rank (`process_group=None` / `SingleGroup`), with the PP/EP filter applied twice (header pre-check and stream filter) | ours; the loaders themselves are upstream vLLM code. Engine start 680/628 s -> 150-166 s |
| `07_pp2_ple_config.py` | `model_executor/models/config.py` | removes the unconditional `PP=1` refusal; the real constraint is placement | **upstream recipe, PR [#54709](https://github.com/vllm-project/vllm/pull/54709)** (reported by @hackxixi on 8x CMP 170HX), adapted to this tree |
| `07b_pp2_ple_state.py` | `nvidia/model_state.py` | replaces the second `PP=1` refusal with a **placement** check: rank 0 must own every layer from `ple_layer_ids`, and no other rank may own one | same PR, adapted |
| `08_kv_diag.py` | `v1/worker/utils.py` | `allocate_kv_cache()` raised `StopIteration` for the PLE layer, which belongs to no KV cache group; such tensors are skipped with a debug line | same PR, adapted |
| `09_mtp_pp.py` | `nvidia/mtp.py` | the MTP drafter lives entirely on the last stage and is called with `hidden_states`, so branching on `get_pp_group().is_first_rank` was wrong; it branches on the incoming tensors instead | same PR; upstream issues #46994 / #52295 cover neighbouring cases and were not needed here |
| `10_ngram_compressed_tensors.py` | `nvidia/ngram_embedding.py` | `CompressedTensorsConfig` (the AWQ-INT4 checkpoints) fell through to `NotImplementedError` at PLE construction; it now maps to the unquantized PLE method | ours - checked against the checkpoint data: the 137 PLE tensors carry no scale/zero-point, so treating them as quantized was simply wrong |
| `11_ngram_skip_scale.py` | `nvidia/ngram_embedding.py` | with `ple_ssd_offload`, skip `ngram_embedding.weight_scale` (the SSD table owns the scale and reads it from disk) | ours |
| `12_autoround_moe_wna16.py` | `nvidia/ngram_embedding.py` | AutoRound (GPTQ) experts through `moe_wna16`: the PLE embedding stays bf16-meta, `MoeWNA16Config` maps to the unquantized method | ours; measured slower on `sm_80` than AWQ-INT4 |
| `12b_autoround_moe_wna16_skip.py` | `layers/quantization/moe_wna16.py` | for GPTQ-style checkpoints `modules_to_not_convert` was forced empty; it now comes from `config.json`, so attention and norms are not sent to AutoGPTQ | ours |
| `13_staged_repack_envs.py` + `13b_staged_repack_utils.py` | `envs.py`, `model_loader/utils.py` | declare `VLLM_MARLIN_REPACK_STAGED` and, between quantized modules, collect garbage and drop the allocator cache so host RAM is returned | ours - without it a 32 GiB host cannot repack a 176 GB checkpoint |
| `14_abliteration.py` | new `nvidia/abliteration.py`, `nvidia/model.py`, `nvidia/mtp.py`, `nvidia/ple_layer.py` | runtime refusal abliteration: one refusal direction is removed from every residual writer (`o_proj`, `out_proj`, expert `down_proj`, `ple.value_proj`, the embedding row of `embed_tokens`); switch `QWEN38_ABLITERATION`. No weight file is rewritten, INT4 experts stay as in the checkpoint | ported from **DominikBucko/qwen38-flash-next-2x3090** (Apache-2.0 overlay); the direction belongs to **orcarouter/Qwen3.8-Flash-Next-Uncensored** (Arditi et al., 2024) and is *not* shipped |
| `16_kv_int8_qsa.py` | `nvidia/qsa.py`, `nvidia/ops/qsa.py` | INT8 KV cache with a scale per (token, head) on the QSA path: the backend accepts `int8_per_token_head`/`fp8_per_token_head`, the impl keeps its own fp32 scale caches outside the KV page, the store goes through the core quantizing kernel, and the attention kernel gains a `QUANT` branch (native int8 -> bf16 widening plus per-token K/V scales). Twenty edits; without `--kv-cache-dtype int8_per_token_head` the new branch is never taken | ported from **DominikBucko/qwen38-flash-next-3090** (Apache-2.0 overlay, same checkpoint on sm_86); the storage side - the quantizing store kernel and the `INT8_PER_TOKEN_HEAD` quant mode - is upstream code of this tree |
| `17_xgrammar_spec_filter.py` | `v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py` | grammar filter for **accepted speculative blocks** under structured output: before the scheduler commits a block, a new `filter_speculative_grammar_tokens` validates it with the non-advancing `validate_tokens` and commits only the longest grammar-valid prefix; the rejected suffix is rolled back (`num_computed_tokens` / `num_output_placeholders`) and resampled next step with the mask engaged. Kills the `[248069, 271, 90]` "grammar rejected tokens" crash (FINISHED_ERROR -> HTTP 500) when an accepted MTP block crosses the reasoning boundary or the grammar completion; terminated grammars and blocks ending on a stop token are left to the commit path | **upstream PR [#52452](https://github.com/vllm-project/vllm/pull/52452)** (ours; fixes #53181 / #46118), adapted to this tree: `_get_constraint_start` + `validate_tokens` instead of `should_advance` / `trim_reasoning_for_advance` |

## Launcher and profile (not scripts)

The recipe patches its own `serve.sh`; this package ships a complete English
launcher instead, so nothing outside the tree needs patching:

| what | where | origin |
|---|---|---|
| PP flags and the `VLLM_PP_LAYER_PARTITION` export | `serve/serve.sh` | ours (complements patch 07) |
| profile selection (`serve.sh <profile.env>`, `RUN_ENV` honoured) | `serve/serve.sh` | ours, generalised from the recipe launcher |
| export allowlist (`VLLM_PLE_CPU_OFFLOAD`, `VLLM_PLE_NONPLE_LOADER`, `QWEN38_ABLITERATION`, ...) | `serve/serve.sh` | ours; without it a profile line silently never reaches the engine |
| optional Bearer auth (`API_KEY` -> `--api-key`) | `serve/serve.sh`, `serve/awq-pp2.env` | ours |
| KV cache dtype (`KV_DTYPE` -> `--kv-cache-dtype`) | `serve/serve.sh`, `serve/awq-pp2.env` | ours; `int8_per_token_head` needs patch 16 |
| the production profile (PP2, MTP2, 262 144 ctx, PLE on NVMe, port 8008) | `serve/awq-pp2.env` | ours, measured (`BENCHMARKS.md`) |

## Markers

Each patch leaves a string that proves it is present; `verify.sh` greps for these
(run from the tree root):

```bash
grep -c 'requires DP=1'                 vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 01
grep -c 'mixed shard'                   vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 02
grep -c '_read_weight_scale'            vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 03
grep -c 'F8_E4M3 checkpoint tensors'    vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 04
grep -c 'local_layers'                  vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 05
grep -c 'VLLM_PLE_NONPLE_LOADER'        vllm/models/qwen4_exp/nvidia/ple_ssd.py             # 06
grep -c 'local_layers'                  vllm/model_executor/model_loader/ep_weight_filter.py  # 05c
grep -c 'vllm#54709'                    vllm/model_executor/models/config.py                # 07
grep -c 'get_pp_indices'                vllm/models/qwen4_exp/nvidia/model_state.py         # 07b
grep -c 'kv_cache_group'                vllm/v1/worker/utils.py                             # 08
grep -c 'intermediate_tensors is None'  vllm/models/qwen4_exp/nvidia/mtp.py                 # 09
grep -c 'CompressedTensorsConfig'       vllm/models/qwen4_exp/nvidia/ngram_embedding.py     # 10
grep -c '_ssd_active'                   vllm/models/qwen4_exp/nvidia/ngram_embedding.py     # 11
grep -c 'MoeWNA16Config'                vllm/models/qwen4_exp/nvidia/ngram_embedding.py     # 12
grep -c 'modules_to_not_convert'        vllm/model_executor/layers/quantization/moe_wna16.py  # 12b
grep -c 'VLLM_MARLIN_REPACK_STAGED'     vllm/envs.py                                        # 13
grep -c 'abliteration.register'         vllm/models/qwen4_exp/nvidia/abliteration.py        # 14
grep -c 'QUANT: tl.constexpr'           vllm/models/qwen4_exp/nvidia/ops/qsa.py             # 16
grep -c 'self._pth_caches'              vllm/models/qwen4_exp/nvidia/qsa.py                 # 16
grep -c 'filter_speculative_grammar_tokens' vllm/v1/structured_output/__init__.py           # 17
grep -c 'num_grammar_rejected'          vllm/v1/core/sched/scheduler.py                     # 17
```

## Edits that are *not* in the build

| item | what it was | why it is absent |
|---|---|---|
| `--safetensors-load-strategy=eager` | load all weights up front | measured worse: 491 s against 343 s, 63 GiB swapped on a 21 GiB host |
| CPU offload of cold experts (`--cpu-offload-gb`) | keep experts in host RAM | not needed, the split fits both cards; the switch still exists in `serve/serve.sh` |
| `--kv-cache-dtype fp8_e4m3` | halve KV bytes per token with one host-side scale per tensor | measured on this build and rejected: Triton on `sm_80` has no `fp8e4nv`, so the dtype has to be faked (`fp8e4b15`, value/256 with the factor folded into the host scales) and the emulation costs about 36 % of decode. The shipped profile uses `int8_per_token_head` instead (patch 16) |
| checkpoint metadata tools (`fix_embedding_dtype.py`, `fix_index_ple_names.py`, `fix_topk_kmax.py`) | repair `config.json` or the index of a converted checkpoint | not needed for the public checkpoint this package serves (`MODEL.md`) |

## Result

After `patches/apply.sh` the nineteen touched files are byte-identical to the tree
this package was built from - the hashes are in `manifest/files.sha256`, and
`overlay/` carries the same content. `verify.sh` is the one command that checks
it.