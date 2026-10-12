#!/usr/bin/env bash
# serve.sh - launch Qwen3.8-Flash-Next (AWQ-INT4 + MTP) with the PLE table on NVMe.
#
# Derived from the recipe launcher scripts/serve.sh of
# github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx (Apache-2.0), extended for the
# patches in ../patches: pipeline parallel, MTP drafting, the non-PLE loader
# switch and runtime abliteration.
#
# Everything is driven by a profile file (KEY=VALUE lines, default awq-pp2.env
# next to this script) that is sourced below; environment values win, because the
# calling shell passes its own environment down:
#   PP_SIZE=1 GPU_MEMORY=0.95 bash serve.sh awq-pp2.env
#
# Only an allowlist of variables is exported to the engine (see the `for v in`
# loop below): a plain `source` does not export, so a profile line that is not in
# that list never reaches the process. The banner printed before the exec shows
# the state of every switch - read it, not the profile.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- engine pid file. It is written before the exec so a supervisor can tell a
# slow start from a dead process: a process with "vllm serve" in its command
# line only exists AFTER the exec at the bottom of this script.
FNNX_PIDFILE="${FLASHNEXT_PIDFILE:-/run/flashnext-engine.pid}"
if echo $$ > "$FNNX_PIDFILE" 2>/dev/null; then
  echo "=== engine pid: $$ (file $FNNX_PIDFILE) ==="
else
  echo "=== WARNING: cannot write $FNNX_PIDFILE (set FLASHNEXT_PIDFILE) ==="
fi

# The profile may carry CRLF line endings (edited on Windows); strip the \r,
# otherwise bash trips over "$'\r': command not found".
PROFILE="${1:-${RUN_ENV:-$HERE/awq-pp2.env}}"
if [ -f "$PROFILE" ]; then
  # shellcheck disable=SC1090
  source <(sed 's/\r$//' "$PROFILE")
  echo "=== profile: $PROFILE"
else
  echo "=== no profile at $PROFILE: using the environment and the defaults ==="
fi

# --- paths and knobs (override in the environment or in the profile) ----------
: "${VLLM_VENV:=/srv/models/vllm/flashnext-env}"
: "${VLLM_BIN:=$VLLM_VENV/bin/vllm}"
: "${MODEL:=/srv/models/vllm/flashnext-awq-int4}"
: "${SERVED_NAME:=Qwen3.8-Flash-Next-AWQ-INT4}"
: "${HOST:=0.0.0.0}" "${PORT:=8008}"
: "${CUDA_DEVICES:=0,1}" "${MAX_NUM_SEQS:=8}" "${MAX_BATCH_TOKENS:=4096}"
: "${MAX_MODEL_LEN:=262144}" "${GPU_MEMORY:=0.90}" "${MTP:=2}" "${API_KEY:=}"
: "${PP_SIZE:=2}" "${TP_SIZE:=1}"
# LOAD_STRATEGY is vLLM's LoadConfig.safetensors_load_strategy:
#   lazy     mmap on demand (default; on a host with less RAM than the
#            checkpoint this reads by page faults and can be very slow);
#   eager    read each file into RAM with plain read() first;
#   prefetch warm the page cache from background threads.
# The production profile keeps the default and relies on patch 06
# (VLLM_PLE_NONPLE_LOADER), which bypasses mmap for the non-PLE weights.
: "${LOAD_STRATEGY:=lazy}"
: "${PLE_SSD_WORKERS:=16}" "${PLE_SSD_CACHE_MB:=512}" "${PLE_SSD_IO_DEPTH:=256}"
: "${PLE_SSD_PREFETCH:=16384}"
# The native helper built by the recipe (scripts/install-runtime.sh, or
# tools/setup_flashnext.sh in the recipe). Point this at your build.
: "${PLE_SSD_LIB:=$VLLM_VENV/lib/ple_ssd_io.so}"
: "${GRAPH_MODE:=FULL_AND_PIECEWISE}" "${VLLM_USE_BREAKABLE_CUDAGRAPH:=1}"
: "${CAPTURE_SIZES:=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,64,128,256,512,1024,2048]}"

# --- where the PLE table lives ------------------------------------------------
#   PLE_SSD_OFFLOAD=1 (default) - recipe path: the n-gram table is read from
#       local NVMe with asynchronous Linux AIO and never enters VRAM.
#       BF16 PLE only; an FP8 PLE table makes the recipe refuse to start
#       (patches 03/04 are the reserve path for that case).
#   PLE_SSD_OFFLOAD=0 + VLLM_PLE_CPU_OFFLOAD=1 - upstream path: the table is
#       kept in host RAM (needed for FP8-PLE checkpoints, costs a lot of RAM).
PLE_SSD_OFFLOAD="${PLE_SSD_OFFLOAD:-1}"
if [ "$PLE_SSD_OFFLOAD" = "1" ]; then
  PLE_ARGS=(--additional-config "{\"ple_ssd_offload\":true,\"ple_ssd_workers\":$PLE_SSD_WORKERS,\"ple_ssd_cache_mb\":$PLE_SSD_CACHE_MB,\"ple_ssd_native_library\":\"$PLE_SSD_LIB\",\"ple_ssd_io_depth\":$PLE_SSD_IO_DEPTH,\"ple_ssd_prefetch_tokens\":$PLE_SSD_PREFETCH}")
  echo "=== PLE: offloaded to SSD (recipe path, BF16 table only)"
  [ -f "$PLE_SSD_LIB" ] || echo "=== WARNING: PLE_SSD_LIB does not exist: $PLE_SSD_LIB"
else
  PLE_ARGS=(--additional-config "{\"mamba_cache_mode\":\"align\"}")
  export VLLM_PLE_CPU_OFFLOAD=1
  echo "=== PLE: kept in host RAM (required for FP8-PLE checkpoints)"
fi

# --- compilation. COMPILE_MODE: 0 = no torch.compile at all, 1 = cudagraph
# without Inductor, 3 = full VLLM_COMPILE. The "mode" key is only added when the
# variable is set, so an unset COMPILE_MODE keeps vLLM's own default.
COMPILATION_CONFIG="{\"cudagraph_mode\":\"$GRAPH_MODE\",\"cudagraph_capture_sizes\":$CAPTURE_SIZES"
[ -n "${COMPILE_MODE:-}" ] && COMPILATION_CONFIG="$COMPILATION_CONFIG,\"mode\":${COMPILE_MODE}"
COMPILATION_CONFIG="$COMPILATION_CONFIG}"

# --- environment for the engine process ---------------------------------------
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# humming-kernels compiles kernels with NVRTC and looks for
# libnvrtc-builtins.so.13.0; if the system CUDA toolkit is older than the torch
# build, take it from the venv first.
VENV_NV="$VLLM_VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
[ -d "$VENV_NV" ] && export LD_LIBRARY_PATH="$VENV_NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The flashinfer sampler is JIT-compiled and needs a matching nvcc; with an older
# system nvcc it fails, so fall back to the Triton sampler.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # the weights are on local disk
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=${MAX_JOBS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VLLM_USE_BREAKABLE_CUDAGRAPH="$VLLM_USE_BREAKABLE_CUDAGRAPH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- speculative decoding (MTP). MTP_MODEL points at a separate draft model
# when the MTP head is not part of the main checkpoint; the number is the count
# of speculative tokens (2 is the production setting).
spec=()
if [ "$MTP" -gt 0 ]; then
  if [ -n "${MTP_MODEL:-}" ] && [ -d "${MTP_MODEL:-}" ]; then
    spec=(--speculative-config "{\"method\":\"mtp\",\"model\":\"$MTP_MODEL\",\"num_speculative_tokens\":$MTP}")
  else
    spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
  fi
fi

# --- optional switches --------------------------------------------------------
keyargs=()
[ -n "$API_KEY" ] && keyargs=(--api-key "$API_KEY")
# KV cache dtype: unset = bfloat16; int8_per_token_head (patch 16) halves the
# bytes per token and more, fp8_e4m3 needs an emulation on sm_80.
kvargs=()
[ -n "${KV_DTYPE:-}" ] && kvargs=(--kv-cache-dtype "$KV_DTYPE")
# Prefix cache: keys are matched in units of the mamba block, so with MTP nothing
# shorter than about 1600 tokens is ever reused; a small unit lowers that
# threshold (recipe wtdcode/vllm-backport). Unset = vLLM's behaviour.
pfxargs=()
[ -n "${PREFIX_MATCH_UNIT:-}" ] && pfxargs=(--prefix-match-unit "$PREFIX_MATCH_UNIT")
# Sampling defaults for thinking mode, applied engine-side so a client that sends
# no temperature/top_p still gets the intended profile.
smpargs=()
[ -n "${SAMPLING:-}" ] && smpargs=(--override-generation-config "$SAMPLING")
# Cold experts on the CPU: needed only when the non-PLE weights do not fit in
# VRAM. CPU_OFFLOAD_GB=0 (default) adds nothing to the command line.
CPU_OFFLOAD_ARGS=()
COGB="${CPU_OFFLOAD_GB:-0}"
if [ "$COGB" != "0" ] && [ -n "$COGB" ]; then
  CPU_OFFLOAD_ARGS=(--offload-backend "${OFFLOAD_BACKEND:-uva}"
                    --cpu-offload-gb "$COGB"
                    --cpu-offload-params "${CPU_OFFLOAD_PARAMS:-experts}")
  echo "=== CPU offload: ${COGB} GB, backend=${OFFLOAD_BACKEND:-uva}, params=${CPU_OFFLOAD_PARAMS:-experts}"
fi

# --- tensor / pipeline parallel flags ----------------------------------------
TP_ARGS=()
[ "$TP_SIZE" != "1" ] && TP_ARGS=(--tensor-parallel-size "$TP_SIZE")
# Patch 07 allows PP>1 for this model: on cards without peer-to-peer access TP
# pays one all-reduce per layer through the host, while PP needs one handover per
# stage. VLLM_PP_LAYER_PARTITION sets the layer split, for example 26,22 for two
# 64 GiB cards (the last stage also carries the MTP drafter).
PP_ARGS=()
[ "$PP_SIZE" != "1" ] && PP_ARGS=(--pipeline-parallel-size "$PP_SIZE")

# --- free-form extra flags, space separated, from the profile -----------------
read -r -a EXTRA_ARGS <<< "${EXTRA_FLAGS:-}"

# --- variables that must reach the engine process ----------------------------
# A profile line is NOT exported by `source`; without this loop the engine would
# silently start without it (that is how the non-PLE loader switch was lost once:
# no error, no line in the log, mmap instead of instanttensor).
for v in VLLM_PLE_CPU_OFFLOAD VLLM_WNA16_DYNAMIC_LRU VLLM_WNA16_STATIC_HOT_CACHE_FILE \
         VLLM_WNA16_MIXED_VMM_HOT_CACHE VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING \
         VLLM_PP_LAYER_PARTITION VLLM_PLE_NONPLE_LOADER QWEN38_ABLITERATION \
         QWEN38_HOT_ONLY QWEN38_GPU_SHARE_FRAC QWEN38_CPU_EXPERTS_CPUS; do
  [ -n "${!v:-}" ] && export "$v"
done

echo "=== $MODEL on GPU $CUDA_DEVICES, port $PORT, ctx $MAX_MODEL_LEN, MTP=$MTP"
echo "=== safetensors load strategy: $LOAD_STRATEGY"
[ "$PLE_SSD_OFFLOAD" = "1" ] && echo "=== PLE on SSD: workers=$PLE_SSD_WORKERS cache=${PLE_SSD_CACHE_MB}MiB depth=$PLE_SSD_IO_DEPTH prefetch=$PLE_SSD_PREFETCH"
[ -n "${VLLM_PP_LAYER_PARTITION:-}" ] && echo "=== pipeline layer split: $VLLM_PP_LAYER_PARTITION (PP=$PP_SIZE)"
if [ -n "${VLLM_PLE_NONPLE_LOADER:-}" ]; then
  echo "=== non-PLE weight loader: $VLLM_PLE_NONPLE_LOADER (patch 06)"
else
  echo "=== non-PLE weight loader: safetensors/mmap (default)"
fi
if [ -n "${KV_DTYPE:-}" ]; then
  echo "=== KV cache dtype: $KV_DTYPE (per-token-head modes need patch 16)"
else
  echo "=== KV cache dtype: bfloat16 (engine default)"
fi
[ -n "$API_KEY" ] && echo "=== API key: enabled (Bearer required on /v1)" || echo "=== API key: not set (open access)"
[ -n "${PREFIX_MATCH_UNIT:-}" ] && echo "=== prefix-match-unit: $PREFIX_MATCH_UNIT" || echo "=== prefix-match-unit: default (one mamba block)"
[ -n "${SAMPLING:-}" ] && echo "=== sampling (thinking mode): $SAMPLING" || echo "=== sampling: engine and model defaults"
# The abliteration state is printed on purpose: a switch that silently does not
# arrive is the most expensive mistake here (the engine looks healthy, the
# behaviour is not what you think). A supervisor can grep this line.
if [ -n "${QWEN38_ABLITERATION:-}" ]; then
  echo "=== refusal abliteration (patch 14): ON  QWEN38_ABLITERATION=$QWEN38_ABLITERATION"
else
  echo "=== refusal abliteration (patch 14): off (the model's own refusals)"
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --safetensors-load-strategy "$LOAD_STRATEGY" \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "${PLE_ARGS[@]}" \
  "${CPU_OFFLOAD_ARGS[@]}" \
  "${TP_ARGS[@]}" \
  "${PP_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --compilation-config "$COMPILATION_CONFIG" \
  "${kvargs[@]}" \
  "${pfxargs[@]}" \
  "${smpargs[@]}" \
  "${keyargs[@]}" \
  "${spec[@]}" \
  "$@"