#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP weight filter in the PLE iterator

Passes the stage's layer range into the iterator so a pipeline stage never
reads tensors of layers it does not own.

usage: python 05_ple_ssd_ppfilter.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-05_ple_ssd_ppfilter next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "05_ple_ssd_ppfilter"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
        '' + '\n'
        '' + '\n'
        'def ple_ssd_weights_iterator(files, use_tqdm, strategy, local_expert_ids=None):' + '\n'
        '    """Avoid mmap\'s commit charge for checkpoint files containing only PLE."""' + '\n'
        '    from vllm.model_executor.model_loader.weight_utils import (' + '\n'
        '        safetensors_weights_iterator,',
        # ---- after
        '' + '\n'
        '' + '\n'
        'def ple_ssd_weights_iterator(' + '\n'
        '    files, use_tqdm, strategy, local_expert_ids=None, local_layers=None' + '\n'
        '):' + '\n'
        '    """Avoid mmap\'s commit charge for checkpoint files containing only PLE.' + '\n'
        '' + '\n'
        '    *local_layers* is handed over to ``safetensors_weights_iterator``' + '\n'
        '    so tensors of layers owned by another pipeline stage are never' + '\n'
        '    read at all (see ep_weight_filter.should_skip_weight).' + '\n'
        '    """' + '\n'
        '    from vllm.model_executor.model_loader.ep_weight_filter import (' + '\n'
        '        should_skip_weight,' + '\n'
        '    )' + '\n'
        '    from vllm.model_executor.model_loader.weight_utils import (' + '\n'
        '        safetensors_weights_iterator,',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                                ),
                            )
                        else:
                            yield (name, fp.get_tensor(name))
            else:""",
        # ---- after
"""                                ),
                            )
                        elif not should_skip_weight(
                            name, local_expert_ids, local_layers
                        ):
                            yield (name, fp.get_tensor(name))
            else:""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids
    )
""",
        # ---- after
"""                regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids, local_layers
    )
""",
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
