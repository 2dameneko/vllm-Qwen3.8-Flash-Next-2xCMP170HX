#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP weight filter in weight_utils

The safetensors iterator skips foreign layers by name before reading them.

usage: python 05d_pp_filter_weights.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-05d_pp_filter_weights next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "05d_pp_filter_weights"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/model_executor/model_loader/weight_utils.py',
        # ---- before
"""    safetensors_load_strategy: str | None = None,
    local_expert_ids: set[int] | None = None,
    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,""",
        # ---- after
"""    safetensors_load_strategy: str | None = None,
    local_expert_ids: set[int] | None = None,
    local_layers=None,
    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,""",
    ),
    (
        'vllm/model_executor/model_loader/weight_utils.py',
        # ---- before
"""                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
        elif safetensors_load_strategy == "torchao":""",
        # ---- after
"""                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids, local_layers):
                    yield name, param
        elif safetensors_load_strategy == "torchao":""",
    ),
    (
        'vllm/model_executor/model_loader/weight_utils.py',
        # ---- before
"""                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)""",
        # ---- after
"""                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids, local_layers):
                        continue
                    state_dict[name] = f.get_tensor(name)""",
    ),
    (
        'vllm/model_executor/model_loader/weight_utils.py',
        # ---- before
"""            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)""",
        # ---- after
"""            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids, local_layers):
                        continue
                    param = f.get_tensor(name)""",
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
