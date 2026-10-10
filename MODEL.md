# MODEL.md - the checkpoint this build serves

## Which one

| | |
|---|---|
| checkpoint | `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (Hugging Face, public) |
| size | ~176 GB, 38 safetensors shards |
| contents | AWQ INT4 quantized backbone (experts included), everything else in the checkpoint's own dtype |
| PLE table | **BF16, 95.37 GiB**, in its own shard files (`...ngram_embedding.shard_*`) - this is the part that goes to NVMe |
| licence | Apache-2.0 for the OrcaRouter-derived weights, Qwen Community License for the base model |

Lineage: `Qwen/Qwen3.8-Flash-Next` (official) → abliterated fine-tune
`orcarouter/Qwen3.8-Flash-Next-Uncensored` → AutoRound quantization → AWQ-INT4.
The same lineage is described in the recipe's README (`base/README-recipe.md`).

Download it into one directory and point `MODEL` in `serve/awq-pp2.env` at it:

```bash
huggingface-cli download cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 \
  --local-dir /srv/models/vllm/flashnext-awq-int4
du -sh /srv/models/vllm/flashnext-awq-int4      # ~176 GB
```

No conversion, no metadata edit, no index repair: this checkpoint is served as
it comes from the hub. (`hf_transfer` speeds the download up substantially on a
fast link.)

## What the recipe expects from a checkpoint

* The PLE (n-gram embedding) tensors are **BF16** and live either in dedicated
  shard files or mixed into a tail shard - the recipe's iterator handles the
  dedicated case, patch 02 adds the mixed case. The table never enters VRAM:
  `PLESSDTable` reads rows from disk by `data_offsets` from
  `model.safetensors.index.json`.
* Mixed precision would break it: the recipe refuses a table that is not BF16
  (`PLE SSD offload currently requires BF16 embeddings`) because it derives the
  row stride from the dtype. Patches 03/04 lift that restriction for an FP8
  (F8_E4M3) table by dequantizing on the host - a reserve path, not the
  production one, because `sm_80` has no FP8 tensor cores.
* The quantized parts must be readable by the quantizer you select. This build
  serves the AWQ-INT4 checkpoint through compressed-tensors (patch 10 teaches the
  PLE method to accept it); the AutoRound/GPTQ variant (`moe_wna16`, patches
  12/12b) was measured as well but is slower on `sm_80`.
* Experts may carry `nn.Linear`-style modules that must **not** be quantized
  (norms, attention). Patch 12b makes the loader honour the checkpoint's own
  `modules_to_not_convert` list instead of forcing it empty.

## Quick health check of a checkpoint before you serve it

```bash
cd /srv/models/vllm/flashnext-awq-int4
python3 - <<'PY'
import json, collections
idx = json.load(open("model.safetensors.index.json"))["weight_map"]
ple = [k for k in idx if ".ngram_embedding.shard_" in k]
dtypes = collections.Counter()
import struct
for f in sorted({idx[k] for k in ple}):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k != "__metadata__":
            dtypes[v["dtype"]] += 1
print("shards:", len(idx), "| PLE tensors:", len(ple), "| PLE dtypes:", dict(dtypes))
PY
# expect: shards ~ 38 | PLE tensors 137 | PLE dtypes {'BF16': 137}
```

If the PLE dtypes are not `BF16`, either use an FP8-PLE checkpoint (with patches
03/04 and a table that fits in host RAM) or convert the table to BF16 first.

## Variants that were tried and are not shipped

| variant | why not |
|---|---|
| FP8-PLE checkpoint (47.68 GiB table) | works through patches 03/04 with host-side dequantization, but the table then competes for RAM and `sm_80` has no FP8 compute; kept as a reserve path |
| AutoRound 3 bpw + `moe_wna16` | the recipe's own target (one card, 15 GiB RAM); measured slower here than AWQ-INT4 with compressed-tensors |
| W4A16-FP8 PLE variants | the non-PLE weights do not fit in 64 GiB without the CPU-offload path; retired |

## VRAM budget on two 64 GiB cards

| item | size |
|---|---|
| weights per stage, split 26,22 | 42.49 / 42.52 GiB (the last stage also carries the MTP drafter) |
| KV cache at ctx 262 144, `GPU_MEMORY=0.90` | 744 431 tokens (about 3.6x concurrency at full context) |
| PLE table in VRAM | 0 - it is on NVMe, only a row cache (512 MiB) is resident |