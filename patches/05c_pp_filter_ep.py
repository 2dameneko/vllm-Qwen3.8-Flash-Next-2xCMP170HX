#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP weight filter core

Adds the local layer range (from get_pp_indices, so VLLM_PP_LAYER_PARTITION is
respected) and should_skip_weight(), plus the mtp.* rule (last stage only).

usage: python 05c_pp_filter_ep.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-05c_pp_filter_ep next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "05c_pp_filter_ep"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/model_executor/model_loader/ep_weight_filter.py',
        # ---- before
"""
import regex as re

# Matches per-expert weight names like ".experts.42.gate_proj.weight".""",
        # ---- after
"""
import regex as re
from typing import NamedTuple

# Matches per-expert weight names like ".experts.42.gate_proj.weight".""",
    ),
    (
        'vllm/model_executor/model_loader/ep_weight_filter.py',
        # ---- before
        '# tensor is loaded and sliced later by RoutedExperts.weight_loader.' + '\n'
        '_EXPERT_ID_RE = re.compile(r"\\.experts\\.(\\d+)\\.")' + '\n'
        '' + '\n'
        '',
        # ---- after
        '# tensor is loaded and sliced later by RoutedExperts.weight_loader.' + '\n'
        '_EXPERT_ID_RE = re.compile(r"\\.experts\\.(\\d+)\\.")' + '\n'
        '' + '\n'
        '# Matches the transformer-layer index in names such as' + '\n'
        '# ".layers.17.mlp.experts.3.gate_proj.weight". Attention, MoE, norm' + '\n'
        '# and PLE sub-modules all carry it, which is what makes pipeline' + '\n'
        '# filtering possible (see compute_local_layer_ids).' + '\n'
        '_LAYER_ID_RE = re.compile(r"\\.layers\\.(\\d+)\\.")' + '\n'
        '' + '\n'
        '',
    ),
    (
        'vllm/model_executor/model_loader/ep_weight_filter.py',
        # ---- before
        '' + '\n'
        '' + '\n'
        'def should_skip_weight(' + '\n'
        '    weight_name: str,' + '\n'
        '    local_expert_ids: set[int] | None,' + '\n'
        ') -> bool:' + '\n'
        '    """Return ``True`` if *weight_name* is an expert weight that does not' + '\n'
        '    belong to the local rank and should be skipped during loading."""' + '\n'
        '    if local_expert_ids is None:' + '\n'
        '        return False',
        # ---- after
        '' + '\n'
        '' + '\n'
        'class LocalLayerIds(NamedTuple):' + '\n'
        '    """Hidden layers owned by one pipeline stage, plus MTP ownership."""' + '\n'
        '' + '\n'
        '    layers: range' + '\n'
        '    owns_mtp: bool' + '\n'
        '' + '\n'
        '' + '\n'
        'def compute_local_layer_ids(' + '\n'
        '    num_hidden_layers: int,' + '\n'
        '    pp_size: int,' + '\n'
        '    pp_rank: int,' + '\n'
        ') -> LocalLayerIds | None:' + '\n'
        '    """Return the hidden layers owned by *pp_rank* (``None`` when PP==1).' + '\n'
        '' + '\n'
        '    Every pipeline stage builds only its own slice of the transformer,' + '\n'
        '    yet the loader walks the whole checkpoint and the model silently' + '\n'
        '    drops foreign layers: on a 76 GiB checkpoint with PP=2 each stage' + '\n'
        '    reads about twice what it needs, and both stages compete for the' + '\n'
        '    same NVMe.  Knowing the slice upfront lets the iterator skip those' + '\n'
        '    tensors *before* they are read (see should_skip_weight).' + '\n'
        '' + '\n'
        "    The range comes from vLLM's own :func:`get_pp_indices`, i.e. it" + '\n'
        '    honours VLLM_PP_LAYER_PARTITION and cannot drift from the actual' + '\n'
        '    placement - models/qwen4_exp/nvidia/model_state.py validates PLE' + '\n'
        '    the same way.  ``mtp.*`` tensors are NOT part of that range: the' + '\n'
        '    multi-token-prediction draft sits on the last stage, which is' + '\n'
        '    reported separately as :attr:`LocalLayerIds.owns_mtp`.' + '\n'
        '    """' + '\n'
        '    if pp_size <= 1:' + '\n'
        '        return None' + '\n'
        '    from vllm.distributed.utils import get_pp_indices' + '\n'
        '' + '\n'
        '    start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)' + '\n'
        '    return LocalLayerIds(range(start, end), pp_rank == pp_size - 1)' + '\n'
        '' + '\n'
        '' + '\n'
        'def should_skip_weight(' + '\n'
        '    weight_name: str,' + '\n'
        '    local_expert_ids: set[int] | None,' + '\n'
        '    local_layers: LocalLayerIds | None = None,' + '\n'
        ') -> bool:' + '\n'
        '    """Return ``True`` if *weight_name* does not belong to the local rank' + '\n'
        '    and should be skipped *before* it is read from disk.' + '\n'
        '' + '\n'
        '    Two independent filters apply, both of them pure I/O savings:' + '\n'
        '    pipeline parallelism (layers owned by another stage) and expert' + '\n'
        '    parallelism (experts owned by another rank).  A skipped tensor is' + '\n'
        '    by construction one the local model would never consume.' + '\n'
        '    """' + '\n'
        '    if local_layers is not None:' + '\n'
        '        if weight_name.startswith("mtp."):' + '\n'
        '            if not local_layers.owns_mtp:' + '\n'
        '                return True' + '\n'
        '        else:' + '\n'
        '            m = _LAYER_ID_RE.search(weight_name)' + '\n'
        '            if m is not None and int(m.group(1)) not in local_layers.layers:' + '\n'
        '                return True' + '\n'
        '    if local_expert_ids is None:' + '\n'
        '        return False',
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
