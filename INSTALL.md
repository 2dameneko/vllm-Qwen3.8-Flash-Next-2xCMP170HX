# INSTALL.md - build from upstream to a running engine

Everything below is one pass on a fresh machine. The result is a vLLM tree whose
fifteen patched files match `manifest/files.sha256` byte for byte, served by
`serve/serve.sh`.

## 0. Prerequisites

| need | why |
|---|---|
| Python 3.12 | the recipe's `install-runtime.sh` creates a venv with `uv`; the pins assume 3.12 |
| CUDA toolkit (any recent) | only for building the small native PLE helper; the CUDA extensions themselves come from a wheel |
| `git`, `patch`, `curl`, `build-essential` | upstream checkout, the recipe patch, the wheel, the native helper |
| Local storage | the checkpoint (~176 GB) **and** the PLE table (95.37 GiB) are read from disk at every start; NVMe strongly recommended |
| RAM | 32 GiB here. The loader patch (06) is what makes this enough; with less RAM keep `VLLM_PLE_NONPLE_LOADER=instanttensor` |
| 2 GPUs with 64 GiB each | `PP=2`; a single 64 GiB card also works with `PP_SIZE=1` and enough RAM |
| NVIDIA driver | 610.57.04 here, open kernel modules |

## 1. Upstream tree at the pinned commit

```bash
mkdir -p /srv/llm/src && cd /srv/llm/src
git clone --depth 1 https://github.com/vllm-project/vllm
cd vllm && git checkout a5a30471ff2bb7f0824f2da10e358af98d304472
```

The tarball works too and needs no `.git`:
`https://github.com/vllm-project/vllm/archive/a5a30471ff2bb7f0824f2da10e358af98d304472.tar.gz`.

Keep the tree where you like: the scripts never assume a location, they take the
tree root as an argument (or as `$VLLM_TREE`). Only the *defaults* in
`serve/awq-pp2.env` carry the paths of the machine this was measured on.

## 2. The recipe (PLE table on local storage)

```bash
cd /srv/llm/src/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
patch -p1 < /path/to/release/base/qwen38-ple-ssd.patch     # 13 files, 2 of them new
sha256sum /path/to/release/base/qwen38-ple-ssd.patch
# 09c4e13fbd2b77169a7f8cb28d36242cc6e3916fe9c5a3c25d8e90101a38ccb0
```

Then the recipe's own setup, which builds the venv (`uv`, python 3.12), installs
the precompiled wheel of the same commit plus the tree in editable mode, and
compiles the native helper `ple_ssd_io.so`:

```bash
bash scripts/install-runtime.sh          # from the recipe repository (base/README-recipe.md)
```

If the wheel URL (`https://wheels.vllm.ai/<revision>/...`) has expired, build the
CUDA extensions from the same commit instead - the rest of this document is
unchanged.

Record where `ple_ssd_io.so` ended up; the profile needs it as `PLE_SSD_LIB` (the
recipe normally puts it inside the venv, this machine keeps it under
`/srv/models/vllm/flashnext-opt/`).

## 3. Our patches

```bash
cd /path/to/release
bash patches/apply.sh /srv/llm/src/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
# ... or, identical result and no scripts: cp -r overlay/. <tree>/
```

Each script prints one line per file (`patched`, `already`, `ANCHOR?`). Twenty
scripts, 40 edits, fifteen files. If a script reports `ANCHOR?`, the tree is not
the pinned revision or is not clean - do not "fix" it by hand, compare against
`overlay/` instead.

## 4. Verify the tree

```bash
cd /path/to/release
bash verify.sh /srv/llm/src/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
```

`verify.sh` checks the fifteen files against `manifest/files.sha256`, compiles
the patched python files, looks for leftovers from other tools
(`*.orig`, `*.rej`, i.e. a recipe patch that did not apply cleanly) and greps for
the marker of every patch in the set - for example `VLLM_PLE_NONPLE_LOADER` in
`ple_ssd.py` and `abliteration.register` in `abliteration.py`.

## 5. Model

See `MODEL.md`. Short version: the production checkpoint is
`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (~176 GB, 38 shards, BF16 PLE table in
separate files), and it needs no metadata surgery.

## 6. First start

```bash
cd /path/to/release
# edit serve/awq-pp2.env: VLLM_BIN, MODEL, PLE_SSD_LIB; set PP_SIZE=1 for a
# single card; leave API_KEY empty unless you want Bearer auth
sudo bash serve/run.sh serve/awq-pp2.env     # waits for /health, tails the log
curl -s http://127.0.0.1:8008/v1/models | head -c 400
```

Expect 150-200 s to the first `200` on `/health` with the loader switch on. Read
the launcher banner: it echoes the layer split, the non-PLE loader, the
abliteration state and the API key state. A switch that is missing there did not
reach the engine, whatever the profile says (see `serve/README.md`).

Stop with `sudo bash serve/run.sh --stop`.

## Paths and overrides

| setting | default in `serve/awq-pp2.env` | meaning |
|---|---|---|
| `VLLM_BIN` | `<venv>/bin/vllm` | the `vllm` entry point of the patched tree |
| `MODEL` | `/srv/models/vllm/flashnext-awq-int4` | checkpoint directory |
| `PLE_SSD_LIB` | `<venv>/lib/ple_ssd_io.so` | the recipe's native helper |
| `PP_SIZE` / `VLLM_PP_LAYER_PARTITION` | `2` / `26,22` | pipeline stages and the layer split |
| `GPU_MEMORY` | `0.90` | VRAM fraction; raise to 0.95 only if nothing else uses the card |
| `MAX_BATCH_TOKENS` | `4096` | prefill/decode trade-off (`BENCHMARKS.md`) |
| `VLLM_PLE_NONPLE_LOADER` | `instanttensor` | `safetensors` (mmap) if instanttensor is not installed |

`serve.sh` takes the profile path as its first argument, and any value can be
overridden from the environment:

```bash
PP_SIZE=1 MAX_MODEL_LEN=32768 GPU_MEMORY=0.95 bash serve/serve.sh serve/awq-pp2.env
```

## Optional: runtime refusal abliteration (patch 14)

The code is installed by patch 14 but does nothing until `QWEN38_ABLITERATION`
points at a direction vector: a JSON file with a `refusal_direction` list, a
safetensors file with a `refusal_direction` tensor, or a bare name resolved
inside the `BUILTIN_DIR` constant of
`vllm/models/qwen4_exp/nvidia/abliteration.py`. The vector is not shipped - it
belongs to the Apache-2.0 release
`orcarouter/Qwen3.8-Flash-Next-Uncensored`. The launcher banner then shows
`=== refusal abliteration (patch 14): ON`. Unset means off, and with it off no
code runs and no weight or activation is touched.

## What "done" looks like

```
manifest/files.sha256 ......... 15/15 OK
/health ....................... 200 after 150-200 s
banner ........................ PLE on SSD, layer split 26,22, loader instanttensor
GPU KV cache size ............. ~744 000 tokens at ctx 262144 and GPU_MEMORY=0.90
```

The acceptance run used here (`BENCHMARKS.md`, "End-to-end acceptance"): a
65 536 token quota, thinking on, 45 507 tokens in 358 s, `finish=stop`, and zero
`NVRM: Xid` in `dmesg` before and after. A GPU Xid counter is worth having on a
fresh machine; see `TROUBLESHOOTING.md`.