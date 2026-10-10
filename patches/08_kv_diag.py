#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KV cache allocation with a group-less PLE layer

allocate_kv_cache() looked up a KV cache group for every tensor and raised
StopIteration for the PLE layer, which has no KV cache at all. Such tensors are
skipped with a debug line.

usage: python 08_kv_diag.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-08_kv_diag next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "08_kv_diag"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/v1/worker/utils.py',
        # ---- before
"""    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):""",
        # ---- after
"""    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        # PATCH (vllm#54709 + diagnostics): the PLE layer shows up in
        # kv_cache_tensors, but belongs to no kv_cache_group - it has
        # no KV cache, which is normal. With PP=1 the lists of layers and groups
        # matched, with PP>1 the PLE tensor stayed "group-less" and next()
        # crashed with StopIteration.
        #
        # A tensor whose layer is in no group contributes nothing to the
        # still contributes nothing (its strides come from group.kv_cache_spec),
        # so skipping it is the correct thing to do.
        _found = None
        for _gid, _g in enumerate(kv_cache_config.kv_cache_groups):
            if layer_name in _g.layer_names:
                _found = (_gid, _g)
                break
        if _found is None:
            # The layer has no KV cache (e.g. PLE) - its tensor is created,
            # but it belongs to no group. On allocation it contributes nothing
            # gives: its strides are derived from group.kv_cache_spec.
            logger.debug(
                "KV: skip tensor for layer %r - not in any kv_cache_group",
                layer_name,
            )
            continue
        group_id, group = _found
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):""",
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
