#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallel for Qwen4Exp: model_state.py

Replaces the second PP=1 refusal with a placement check: rank 0 must own every
layer from ple_layer_ids, and no other rank may own one. Layer ranges come
from get_pp_indices, that is from VLLM_PP_LAYER_PARTITION.

usage: python 07b_pp2_ple_state.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-07b_pp2_ple_state next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "07b_pp2_ple_state"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/model_state.py',
        # ---- before
"""            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1""",
        # ---- after
"""            return

        # PATCH (vllm#54709): the check is on PLACEMENT, not on PP>1 itself.
        # Here pp_group is already initialized, so the rank is known.
        from vllm.distributed import get_pp_group
        from vllm.distributed.utils import get_pp_indices

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            _pp_size = vllm_config.parallel_config.pipeline_parallel_size
            _pp_rank = get_pp_group().rank_in_group
            _num_layers = int(config.num_hidden_layers)
            _start, _end = get_pp_indices(_num_layers, _pp_rank, _pp_size)
            _ple_idx = sorted(int(i) for i in config.ple_layer_ids)
            _here = [i for i in _ple_idx if _start <= i < _end]
            if _pp_rank == 0 and len(_here) != len(_ple_idx):
                raise RuntimeError(
                    f"Qwen4Exp PLE layers {sorted(set(_ple_idx) - set(_here))} "
                    f"fall outside pipeline rank 0 (layers {_start}..{_end - 1}). "
                    f"Give rank 0 more layers via VLLM_PP_LAYER_PARTITION, "
                    f"or run with PP=1."
                )
            if _pp_rank != 0 and _here:
                raise RuntimeError(
                    f"Qwen4Exp PLE layers {_here} landed on pipeline rank "
                    f"{_pp_rank} (layers {_start}..{_end - 1}), but PLE must "
                    f"live on rank 0. Adjust VLLM_PP_LAYER_PARTITION."
                )

        self.ngram_context_len = int(config.ngram_size) - 1""",
    ),
)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args and args[0] in ("install", "check", "remove") else "check"
    rest = args[1:] if args and args[0] in ("install", "check", "remove") else args
    root = Path(rest[0]).resolve() if rest else Path(__file__).resolve().parent.parent
    if not (root / "vllm").is_dir():
        print(f"not a vLLM tree: {root} (pass its root as an argument)", file=sys.stderr)
        return 2

    bad, soft = [], []
    for rel, old, new in EDITS:
        path = root / rel
        if not path.is_file():
            print(f"  MISSING  {rel}")
            bad.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"  already  {rel}")
            continue
        if old not in text:
            # A later patch in the chain may have rewritten this very region, so
            # neither the before nor the after text is here any more.
            print(f"  neither  {rel} - before and after text are both absent")
            soft.append(rel)
            continue
        if cmd == "check":
            print(f"  would    {rel}")
            continue
        if cmd == "remove":
            print(f"  skip     {rel} (nothing installed here)")
            continue
        bak = path.with_name(path.name + ".bak-" + TAG)
        shutil.copy2(path, bak)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
        print(f"  patched  {rel}  (backup {bak.name})")

    if cmd == "remove":
        for rel, _old, _new in EDITS:
            path = root / rel
            bak = path.with_name(path.name + ".bak-" + TAG)
            if bak.is_file():
                shutil.copy2(bak, path)
                print(f"  restored {rel} <- {bak.name}")
    if soft:
        print(f"note: {len(soft)} region(s) were rewritten by a later patch of the set;"
              f" check the end state with ../verify.sh")
    if bad and cmd == "install":
        print(f"WARNING: {len(bad)} file(s) were not patched", file=sys.stderr)
        return 1
    if cmd == "check":
        print("check only, nothing written" if not bad
              else f"check: {len(bad)} file(s) do not match this patch")
        return 1 if bad else 0
    print(f"{TAG}: {len(EDITS)} edit(s) in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
