#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP weight filter in the default loader

Hands the stage's layer range to the safetensors iterator and to the parallel
loaders, so only local tensors are read.

usage: python 05b_pp_filter_loader.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-05b_pp_filter_loader next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "05b_pp_filter_loader"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (""",
        # ---- after
"""from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    LocalLayerIds,
    compute_local_expert_ids,
    compute_local_layer_ids,
)
from vllm.model_executor.model_loader.weight_utils import (""",
    ),
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config""",
        # ---- after
"""        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None
        self.local_layers: LocalLayerIds | None = None

        extra_config = load_config.model_loader_extra_config""",
    ),
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""                    self.load_config.safetensors_load_strategy,
                    self.local_expert_ids,
                )
            elif self.load_config.load_format == "fastsafetensors":""",
        # ---- after
"""                    self.load_config.safetensors_load_strategy,
                    self.local_expert_ids,
                    self.local_layers,
                )
            elif self.load_config.load_format == "fastsafetensors":""",
    ),
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads""",
        # ---- after
"""                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        local_layers=self.local_layers,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads""",
    ),
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )
""",
        # ---- after
        '            fall_back_to_pt=True,' + '\n'
        '            allow_patterns_overrides=None,' + '\n'
        '        )' + '\n'
        '' + '\n'
        '    def _init_pp_weight_filter(self, model_config: ModelConfig) -> None:' + '\n'
        '        """Record the hidden layers owned by this pipeline stage.' + '\n'
        '' + '\n'
        '        Kept separate from _init_ep_weight_filter on purpose: the EP' + '\n'
        '        filter only arms itself when expert parallelism actually spans' + '\n'
        '        more than one rank (its ep_size = dp*pcp*tp excludes PP), so it' + '\n'
        '        stays dormant in the PP2 profile while the pipeline filter does' + '\n'
        '        not.  Failure to determine the layer count disables the filter' + '\n'
        '        instead of breaking the start.' + '\n'
        '        """' + '\n'
        '        from vllm.config import get_current_vllm_config' + '\n'
        '' + '\n'
        '        parallel_config = get_current_vllm_config().parallel_config' + '\n'
        '        pp_size = parallel_config.pipeline_parallel_size' + '\n'
        '        if pp_size <= 1:' + '\n'
        '            return' + '\n'
        '' + '\n'
        '        num_layers = getattr(' + '\n'
        '            model_config.hf_text_config, "num_hidden_layers", None' + '\n'
        '        )' + '\n'
        '        if not num_layers:' + '\n'
        '            logger.warning_once(' + '\n'
        '                "PP weight filter disabled: unknown num_hidden_layers."' + '\n'
        '            )' + '\n'
        '            return' + '\n'
        '' + '\n'
        '        from vllm.distributed import get_pp_group' + '\n'
        '' + '\n'
        '        pp_rank = get_pp_group().rank_in_group' + '\n'
        '        self.local_layers = compute_local_layer_ids(' + '\n'
        '            int(num_layers), pp_size, pp_rank' + '\n'
        '        )' + '\n'
        '        layers = self.local_layers.layers' + '\n'
        '        logger.info_once(' + '\n'
        '            "PP weight filter: pp_rank=%d/%d keeps layers %d..%d "' + '\n'
        '            "(mtp=%s); foreign-layer tensors are skipped before they "' + '\n'
        '            "are read from disk",' + '\n'
        '            pp_rank,' + '\n'
        '            pp_size,' + '\n'
        '            layers.start,' + '\n'
        '            layers.stop - 1,' + '\n'
        '            self.local_layers.owns_mtp,' + '\n'
        '        )' + '\n'
        '',
    ),
    (
        'vllm/model_executor/model_loader/default_loader.py',
        # ---- before
"""
        self._init_ep_weight_filter(model_config)

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))""",
        # ---- after
"""
        self._init_ep_weight_filter(model_config)
        self._init_pp_weight_filter(model_config)

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))""",
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
