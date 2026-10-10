# diffs/ - the change set as unified diffs

One diff per file, each against a **clean base**: the pinned upstream tree with
the recipe patch `base/qwen38-ple-ssd.patch` applied and nothing else. That is
exactly the state `patches/01..` starts from, so the diffs and the scripts
describe the same change.

| diff | file in the vLLM tree | composed of |
|---|---|---|
| `01-ple_ssd.py.diff` | `vllm/models/qwen4_exp/nvidia/ple_ssd.py` | 01, 02, 03, 04, 05, 06 |
| `02-ngram_embedding.py.diff` | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | 10, 11, 12 |
| `03-model_state.py.diff` | `vllm/models/qwen4_exp/nvidia/model_state.py` | 07b |
| `04-model.py.diff` | `vllm/models/qwen4_exp/nvidia/model.py` | 14 |
| `05-ple_layer.py.diff` | `vllm/models/qwen4_exp/nvidia/ple_layer.py` | 14 |
| `06-mtp.py.diff` | `vllm/models/qwen4_exp/nvidia/mtp.py` | 09, 14 |
| `07-config.py.diff` | `vllm/model_executor/models/config.py` | 07 |
| `08-utils.py.diff` | `vllm/v1/worker/utils.py` | 08 |
| `09-moe_wna16.py.diff` | `vllm/model_executor/layers/quantization/moe_wna16.py` | 12b |
| `10-ep_weight_filter.py.diff` | `vllm/model_executor/model_loader/ep_weight_filter.py` | 05c |
| `11-default_loader.py.diff` | `vllm/model_executor/model_loader/default_loader.py` | 05b |
| `12-weight_utils.py.diff` | `vllm/model_executor/model_loader/weight_utils.py` | 05d |
| `13-model_loader-utils.py.diff` | `vllm/model_executor/model_loader/utils.py` | 13b |
| `14-envs.py.diff` | `vllm/envs.py` | 13 |
| `15-abliteration.py.diff` | `vllm/models/qwen4_exp/nvidia/abliteration.py` | 14 (a new file, so the diff is the whole file) |

Apply one of them to the clean base:

```bash
cd /path/to/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
patch -p1 < /path/to/release/diffs/07-config.py.diff
```

Two caveats:

* The diffs of `ple_ssd.py`, `ngram_embedding.py` and `mtp.py` are the *sum* of
  several patches, because those files were edited by several of them. Applying
  such a diff to a tree that already carries part of the set will conflict -
  use the scripts in `patches/` for partial states.
* The diffs cover the tree only. The launcher and the profile are shipped ready
  to use in `serve/` (see `PATCHES.md`, "Launcher and profile").

`overlay/` is the alternative to both: unpack it over the tree and the fifteen
files are byte-identical to the shipped ones (`manifest/files.sha256`).