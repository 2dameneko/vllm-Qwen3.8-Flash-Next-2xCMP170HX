#!/usr/bin/env bash
# apply.sh - run every patch script of this directory, in order, on a vLLM tree.
#
#   bash apply.sh [<vllm-tree-root>] [--check]
#
# Order matters: later patches reuse text that earlier ones inserted (the FP8
# iterator works on the mixed-shard branch, the loader switch extends the
# iterator tail, and so on). Each script is idempotent, keeps a
# <file>.bak-<patch> copy of what it changes and can be reverted on its own:
#
#   python 01_ple_ssd_tp.py check  /path/to/vllm
#   python 01_ple_ssd_tp.py remove /path/to/vllm
#
# The scripts expect and keep LF line endings.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-/srv/llm/src/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472}"
CMD=install
[ "${2:-}" = "--check" ] && CMD=check
[ "${1:-}" = "--check" ] && { CMD=check; ROOT="${2:-.}"; }

[ -d "$ROOT/vllm" ] || { echo "not a vLLM tree: $ROOT"; exit 2; }

ORDER="01_ple_ssd_tp 02_ple_ssd_mixed 03_ple_ssd_fp8 04_ple_ssd_fp8_iter
05_ple_ssd_ppfilter 06_ple_ssd_nonple_loader 05b_pp_filter_loader
05c_pp_filter_ep 05d_pp_filter_weights 07_pp2_ple_config 07b_pp2_ple_state
08_kv_diag 09_mtp_pp 10_ngram_compressed_tensors 11_ngram_skip_scale
12_autoround_moe_wna16 12b_autoround_moe_wna16_skip 13_staged_repack_envs
13b_staged_repack_utils 14_abliteration"

failed=""
for tag in $ORDER; do
  printf '== %-32s ' "$tag"
  if out=$(python3 "$HERE/$tag.py" "$CMD" "$ROOT" 2>&1); then
    echo "$(echo "$out" | tail -1)"
  else
    echo "FAILED"
    echo "$out" | sed 's/^/     /'
    failed="$failed $tag"
  fi
done

if [ -n "$failed" ]; then
  echo "== patches that did not apply:$failed"
  echo "   (the tree may already carry them, or it is not the pinned revision)"
  exit 1
fi
echo "== all patches are $([ "$CMD" = check ] && echo 'present' || echo 'in place')"
echo "   verify the result against ../manifest/files.sha256"