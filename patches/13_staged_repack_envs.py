#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Staged repack env switch

Declares VLLM_MARLIN_REPACK_STAGED in vllm/envs.py (used by the loader patch).

usage: python 13_staged_repack_envs.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-13_staged_repack_envs next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "13_staged_repack_envs"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/envs.py',
        # ---- before
"""    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    VLLM_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None""",
        # ---- after
"""    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MARLIN_REPACK_STAGED: bool = False
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    VLLM_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None""",
    ),
    (
        'vllm/envs.py',
        # ---- before
        '    ),' + '\n'
        '    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.' + '\n'
        '    "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: (' + '\n'
        '        os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1"',
        # ---- after
        '    ),' + '\n'
        '    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.' + '\n'
        '        "VLLM_MARLIN_REPACK_STAGED":' + '\n'
        '        lambda: os.environ.get("VLLM_MARLIN_REPACK_STAGED") == "1",' + '\n'
        '    "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: (' + '\n'
        '        os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1"',
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
