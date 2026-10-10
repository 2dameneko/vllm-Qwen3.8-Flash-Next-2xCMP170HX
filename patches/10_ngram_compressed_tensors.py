#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CompressedTensors PLE checkpoints (AWQ-INT4)

CompressedTensorsConfig (the AWQ-INT4 checkpoints) fell through to
NotImplementedError at PLE construction. compressed-tensors quantizes Linear
weights only, so the PLE table stays unquantized and needs the same method as
INCConfig.

usage: python 10_ngram_compressed_tensors.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-10_ngram_compressed_tensors next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "10_ngram_compressed_tensors"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ngram_embedding.py',
        # ---- before
"""    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc.inc import INCConfig""",
        # ---- after
"""    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc.inc import INCConfig""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ngram_embedding.py',
        # ---- before
"""            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):""",
        # ---- after
"""            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # AWQ / CompressedTensors: the experts are quantized to INT4, while the PLE table
        # is left as-is. The branch is needed UNCONDITIONALLY, without
        # an ignore list, and here is why:
        #   * compressed-tensors quantizes Linear weights only. The PLE table is
        #     conv1d and norms, so it is not in ignore - nothing to skip;
        #   * checked on the checkpoint: 137 PLE tensors, 0 of them carry
        #     scale/zero_point/g_idx, while the expert linear layers do
        #     73 728 tensors carry a scale out of 294 914;
        #   * with the is_layer_skipped check the branch never fired for ple.conv1d
        #     and ple.norm_*, and execution fell through to the next raise
        #     NotImplementedError with the very same message - from the outside
        #     it looked as if the patch had not been applied at all.
        # Therefore PLE needs the same unquantized method as INCConfig.
        if isinstance(quant_config, CompressedTensorsConfig):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):""",
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
