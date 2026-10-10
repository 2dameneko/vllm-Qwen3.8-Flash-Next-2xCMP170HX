# serve/ - the launcher and the production profile

| file | what it is |
|---|---|
| `serve.sh` | the launcher: builds the `vllm serve` command line from a profile, prints a banner for every switch, then `exec`s the engine |
| `awq-pp2.env` | the profile the published benchmarks were taken with (PP2 + MTP2, ctx 262144, PLE on NVMe, port 8008) |
| `run.sh` | background wrapper: detached start, wait for `/health`, tail the log on failure, `--stop` to shut down |

## Start, watch, stop

```bash
# foreground: the launcher prints its banner and becomes the engine process
sudo bash serve/serve.sh serve/awq-pp2.env

# or detached, waiting until /health answers 200 (150-200 s on this hardware)
sudo bash serve/run.sh serve/awq-pp2.env
curl -s http://127.0.0.1:8008/health
sudo bash serve/run.sh --stop
```

`serve.sh` derives from the recipe launcher of the same name (Apache-2.0,
`github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx`) and adds what the patches in
`../patches` need: the PP flags and layer split, the MTP draft config, the
non-PLE loader switch, the abliteration switch in the export allowlist, and an
optional API key.

## The banner is the truth, not the profile

The profile is `source`d, which does **not** export anything, and the launcher
exports only an allowlist of variables to the engine. A key that is spelled
differently, or missing from that list, produces no error at all: the engine
starts, and the value silently does nothing. That is why every switch is echoed
before the `exec` and why the two lines below matter:

```
=== non-PLE weight loader: instanttensor (patch 06)     # not "safetensors/mmap"
=== pipeline layer split: 26,22 (PP=2)
```

If a line is missing or shows the default, the value never reached the process -
read the profile again, not the log. The same check for a running engine, from
the outside:

```bash
sudo tr '\0' '\n' < /proc/$(cat /run/flashnext-engine.pid)/environ | grep -E 'VLLM_|QWEN38_'
```

## Knobs worth knowing

| variable | effect |
|---|---|
| `GPU_MEMORY` | VRAM fraction. 0.90 in production; 0.95 is fine only when nothing else shares the card |
| `MAX_BATCH_TOKENS` | 4096 here: prefill@32768 8330 vs 7522 tok/s and decode 121 vs 97 tok/s against 2048, for 0.7 % of KV |
| `VLLM_PP_LAYER_PARTITION` | layer split between the two cards, `26,22`. The PLE layer must stay on rank 0 (patch 07 checks it) |
| `MAX_NUM_SEQS` | concurrent sequences; raise it only together with KV (see the engine's `Maximum concurrency` line) |
| `VLLM_PLE_NONPLE_LOADER` | `instanttensor` (production), `safetensors` (mmap), `fastsafetensors` |
| `MTP` | speculative tokens; `2` in production, `0` disables the draft model |
| `QWEN38_ABLITERATION` | path or name of a refusal direction (patch 14); empty = off |
| `API_KEY` | non-empty turns on Bearer auth on `/v1`, `/v2`, `/inference`, `/cohere` |

Anything else can be passed as a raw flag through `EXTRA_FLAGS`, or on the
command line: `bash serve.sh awq-pp2.env --enforce-eager`.