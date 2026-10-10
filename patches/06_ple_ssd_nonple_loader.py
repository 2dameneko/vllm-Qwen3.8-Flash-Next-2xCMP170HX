#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Non-PLE weight loader switch

With ple_ssd_offload set, this iterator captures the whole weight stream, so
the fastsafetensors/instanttensor branches of the loader were unreachable and
the non-PLE files were loaded through mmap. VLLM_PLE_NONPLE_LOADER selects
safetensors (mmap, default), instanttensor or fastsafetensors; both run
single-rank (process_group=None / SingleGroup) because with PP2 each rank
owns a different file list. The PP/EP filter is applied twice: a header
pre-check and a stream filter.

usage: python 06_ple_ssd_nonple_loader.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-06_ple_ssd_nonple_loader next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "06_ple_ssd_nonple_loader"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""            else:
                regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids, local_layers
    )

""",
        # ---- after
"""            else:
                regular_files.append(path)
    import os as _os

    _nonple = (
        _os.environ.get("VLLM_PLE_NONPLE_LOADER", "safetensors").strip() or "safetensors"
    )
    if _nonple == "safetensors":
        yield from safetensors_weights_iterator(
            regular_files, use_tqdm, strategy, local_expert_ids, local_layers
        )
        return

    # --- PATCH #13: direct I/O instead of mmap for non-PLE files --------------
    # mmap (page faults) starves the NVMe, see docs/benchmarks.md section 4d.
    # We run SINGLE-RANK (process_group=None / SingleGroup): with PP2 the list
    # of files differs per rank, the libraries' TP mode does not apply here.
    import logging as _logging

    _log = _logging.getLogger(__name__)
    _kept = []
    for _path in regular_files:
        with open(_path, "rb") as _f:
            _hlen = struct.unpack("<Q", _f.read(8))[0]
            _header = json.loads(_f.read(_hlen))
        if any(
            not should_skip_weight(_n, local_expert_ids, local_layers)
            for _n in _header
            if _n != "__metadata__"
        ):
            _kept.append(_path)
    _log.info(
        "PLE non-PLE loader=%s: reading %d of %d files (PP/EP filter by header)",
        _nonple,
        len(_kept),
        len(regular_files),
    )
    if not _kept:
        _log.info("PLE non-PLE loader=%s: no non-PLE files of ours - skipping", _nonple)
        return

    if _nonple == "instanttensor":
        import instanttensor

        _device = torch.cuda.current_device()
        with instanttensor.safe_open(
            _kept,
            framework="pt",
            device=_device,
            process_group=None,
            copy=True,
        ) as _fp:
            for _name, _tensor in _fp.tensors():
                if should_skip_weight(_name, local_expert_ids, local_layers):
                    continue
                yield (_name, _tensor)
    elif _nonple == "fastsafetensors":
        from fastsafetensors import SingleGroup
        from fastsafetensors.parallel_loader import ParallelLoader

        _device = torch.device(f"cuda:{torch.cuda.current_device()}")
        _pl = ParallelLoader(
            pg=SingleGroup(),
            hf_weights_files=sorted(_kept),
            use_tqdm_on_load=bool(use_tqdm),
            device=str(_device),
            nogds=True,
        )
        try:
            for _name, _tensor in _pl.iterate_weights():
                if should_skip_weight(_name, local_expert_ids, local_layers):
                    continue
                yield (_name, _tensor)
        finally:
            _pl.close()
    else:
        raise ValueError(
            "VLLM_PLE_NONPLE_LOADER must be safetensors|instanttensor|"
            f"fastsafetensors, got {_nonple!r}"
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
