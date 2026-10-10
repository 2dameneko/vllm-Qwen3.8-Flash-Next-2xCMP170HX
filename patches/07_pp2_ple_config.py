#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallel for Qwen4Exp: config.py

The unconditional PP=1 refusal is removed. The real constraint is placement:
the PLE layers must live on rank 0. That check needs a rank, so it lives in
model_state.py, where the process group exists.

usage: python 07_pp2_ple_config.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-07_pp2_ple_config next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "07_pp2_ple_config"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/model_executor/models/config.py',
        # ---- before
"""        # Checked again in Qwen4ExpModelState; rejecting it here keeps the
        # engine from loading weights first.
        if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
                "because non-first pipeline ranks do not receive the raw input_ids "
                "it needs. Please run with PP=1."
            )
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:""",
        # ---- after
"""        # Checked again in Qwen4ExpModelState; rejecting it here keeps the
        # engine from loading weights first.
        # PATCH (vllm#54709): the blanket PP>1 ban is lifted - PLE is needed not
        # all stages, but only the one that actually holds its layer.
        #
        # The placement check cannot live here: ParallelConfig
        # is process-independent and knows no rank of its own (verified - it has
        # pipeline_parallel_rank attribute). The check lives in
        # Qwen4ExpModelState, where pp_group is already initialized.
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:""",
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
