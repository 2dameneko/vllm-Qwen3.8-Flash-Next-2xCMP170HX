#!/usr/bin/env bash
# verify.sh - check a patched tree against this release.
#
#   bash verify.sh [<vllm-tree-root>]      # default: $VLLM_TREE
#
# Four checks, all read-only:
#   1. the nineteen files this release changes match manifest/files.sha256
#   2. those python files compile
#   3. every patch of the set left its marker in the tree
#   4. the tree is LF-only and has no leftovers from a fuzzed recipe patch
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-${VLLM_TREE:-}}"
[ -n "$ROOT" ] || { echo "usage: bash verify.sh <vllm-tree-root>"; exit 2; }
[ -d "$ROOT/vllm" ] || { echo "not a vLLM tree: $ROOT"; exit 2; }

fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; }

echo "== 1. hashes (manifest/files.sha256)"
miss=0
while read -r hash rel; do
  [ -n "${rel:-}" ] || continue
  got=$(sha256sum "$ROOT/$rel" 2>/dev/null | cut -d' ' -f1)
  if [ "$got" = "$hash" ]; then
    note OK "$rel"
  else
    note "FAIL" "$rel  (expected ${hash:0:12}…, got ${got:0:12}…)"
    miss=$((miss + 1))
  fi
done < <(grep -E '^[0-9a-f]{64} ' "$HERE/manifest/files.sha256")
[ "$miss" -eq 0 ] || fail=1

echo "== 2. python syntax of the patched files"
if command -v python3 >/dev/null; then
  if python3 -m py_compile $(awk '{print "'"$ROOT"'/" $2}' "$HERE/manifest/files.sha256" | tr '\n' ' '); then
    note OK "all files compile"
  else
    note FAIL "a patched file does not compile"; fail=1
  fi
else
  note SKIP "python3 not found"
fi

echo "== 3. patch markers"
while IFS='|' read -r needle rel; do
  [ -n "${rel:-}" ] || continue
  if grep -q -- "$needle" "$ROOT/$rel" 2>/dev/null; then
    note OK "$needle"
  else
    note FAIL "$needle missing in $rel"; fail=1
  fi
done <<'MARKERS'
requires DP=1|vllm/models/qwen4_exp/nvidia/ple_ssd.py
mixed shard|vllm/models/qwen4_exp/nvidia/ple_ssd.py
_read_weight_scale|vllm/models/qwen4_exp/nvidia/ple_ssd.py
F8_E4M3 checkpoint tensors|vllm/models/qwen4_exp/nvidia/ple_ssd.py
VLLM_PLE_NONPLE_LOADER|vllm/models/qwen4_exp/nvidia/ple_ssd.py
local_layers|vllm/models/qwen4_exp/nvidia/ple_ssd.py
local_layers|vllm/model_executor/model_loader/ep_weight_filter.py
local_layers|vllm/model_executor/model_loader/default_loader.py
local_layers|vllm/model_executor/model_loader/weight_utils.py
vllm#54709|vllm/model_executor/models/config.py
get_pp_indices|vllm/models/qwen4_exp/nvidia/model_state.py
kv_cache_group|vllm/v1/worker/utils.py
intermediate_tensors is None|vllm/models/qwen4_exp/nvidia/mtp.py
CompressedTensorsConfig|vllm/models/qwen4_exp/nvidia/ngram_embedding.py
_ssd_active|vllm/models/qwen4_exp/nvidia/ngram_embedding.py
MoeWNA16Config|vllm/models/qwen4_exp/nvidia/ngram_embedding.py
modules_to_not_convert|vllm/model_executor/layers/quantization/moe_wna16.py
VLLM_MARLIN_REPACK_STAGED|vllm/envs.py
VLLM_MARLIN_REPACK_STAGED|vllm/model_executor/model_loader/utils.py
QWEN38_ABLITERATION|vllm/models/qwen4_exp/nvidia/abliteration.py
abliteration.register|vllm/models/qwen4_exp/nvidia/model.py
abliteration.register|vllm/models/qwen4_exp/nvidia/mtp.py
abliteration.register|vllm/models/qwen4_exp/nvidia/ple_layer.py
QUANT: tl.constexpr|vllm/models/qwen4_exp/nvidia/ops/qsa.py
self._pth_caches|vllm/models/qwen4_exp/nvidia/qsa.py
filter_speculative_grammar_tokens|vllm/v1/structured_output/__init__.py
num_grammar_rejected|vllm/v1/core/sched/scheduler.py
MARKERS

echo "== 4. tree hygiene"
crlf=""
while read -r _hash rel; do
  [ -n "${rel:-}" ] || continue
  head -c 2000000 "$ROOT/$rel" 2>/dev/null | grep -qU $'\r' && crlf="$crlf $rel"
done < <(grep -E '^[0-9a-f]{64} ' "$HERE/manifest/files.sha256")
if [ -z "$crlf" ]; then
  note OK "all patched files are LF"
else
  note WARN "CRLF inside a patched file (the scripts write LF):$crlf"
fi
left=$(find "$ROOT/vllm" -name '*.rej' 2>/dev/null | head -3)
if [ -z "$left" ]; then
  note OK "no failed-hunk leftovers (*.rej)"
else
  note INFO "*.rej from applying the recipe with GNU patch (harmless, but it means"
  note INFO "     the patch fuzzed there): $left"
fi
unexpected=$(find "$ROOT/vllm" -name '*.bak-*' -not -name '*.bak-01_*' 2>/dev/null | head -3)
[ -z "$unexpected" ] || note INFO "backup files from earlier edits: $unexpected"

echo
if [ "$fail" -eq 0 ]; then
  echo "verify.sh: OK - this tree matches the release"
else
  echo "verify.sh: FAILED - see the lines marked FAIL above"
  echo "  the file is either not patched, or patched with a different revision"
fi
exit "$fail"