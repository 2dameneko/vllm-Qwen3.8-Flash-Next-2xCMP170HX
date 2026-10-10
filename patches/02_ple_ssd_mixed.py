#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mixed safetensors shards

Real checkpoints keep a tail shard where PLE tensors sit next to ordinary
weights. PLE tensors are yielded as meta (the SSD table reads them by
data_offsets), the rest of the shard is read with plain safe_open.

usage: python 02_ple_ssd_mixed.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-02_ple_ssd_mixed next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "02_ple_ssd_mixed"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                )
        else:
            if any(".ngram_embedding.shard_" in k for k in tensors):
                raise ValueError("PLE SSD requires PLE tensors in dedicated shards")
            regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids""",
        # ---- after
"""                )
        else:
            # PATCH: mixed shard (PLE + regular weights) is allowed.
            # PLE is yielded as meta - PLESSDTable reads it from disk
            # by data_offsets. The other tensors are read with plain safe_open,
            # otherwise they would be lost or land in VRAM together with PLE.
            if any(".ngram_embedding.shard_" in k for k in tensors):
                from safetensors import safe_open

                with safe_open(path, framework="pt") as fp:
                    for name, info in tensors.items():
                        if ".ngram_embedding.shard_" in name:
                            if info["dtype"] != "BF16":
                                raise ValueError(
                                    "PLE SSD requires BF16 checkpoint tensors"
                                )
                            yield (
                                name,
                                torch.empty(
                                    info["shape"],
                                    dtype=torch.bfloat16,
                                    device="meta",
                                ),
                            )
                        else:
                            yield (name, fp.get_tensor(name))
            else:
                regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids""",
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
