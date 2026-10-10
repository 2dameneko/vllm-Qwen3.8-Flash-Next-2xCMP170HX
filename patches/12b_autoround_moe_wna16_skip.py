#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AutoRound (GPTQ) modules_to_not_convert

For GPTQ-style checkpoints MoeWNA16Config forced modules_to_not_convert to an
empty list; it now reads the list from config.json, so attention and norms are
not sent to AutoGPTQ.

usage: python 12b_autoround_moe_wna16_skip.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-12b_autoround_moe_wna16_skip next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "12b_autoround_moe_wna16_skip"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/model_executor/layers/quantization/moe_wna16.py',
        # ---- before
"""        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method in ("awq", "awq_marlin"):
            has_zp = cls.get_from_keys(config, ["zero_point"])""",
        # ---- after
"""        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            # PATCH: AutoRound-GPTQ checkpoints quantize the experts only;
            # without the attention/norm list they would go to AutoGPTQ and fail.
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        elif linear_quant_method in ("awq", "awq_marlin"):
            has_zp = cls.get_from_keys(config, ["zero_point"])""",
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
