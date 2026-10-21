# vLLM build: Qwen3.8-Flash-Next-AWQ-INT4 on 2x NVIDIA CMP 170HX

A reproducible, self-contained build that turns upstream vLLM
`a5a30471ff2bb7f0824f2da10e358af98d304472` into an engine serving
**Qwen3.8-Flash-Next-AWQ-INT4** on two **CMP 170HX** cards (GA100, `sm_80`,
64 GiB each): pipeline parallel across both cards, MTP speculative decoding,
262 144 token context, and the 95.37 GiB PLE (n-gram embedding) table offloaded
to local NVMe instead of VRAM.

Why this exists: these cards have no peer-to-peer link (Gen2 x4 through the
host), and the checkpoint's 95 GiB PLE table cannot live in 64 GiB of VRAM.
Together those two facts rule out the obvious layouts - tensor parallel across
the cards, or the table in memory - so the build has to pipeline the model and
stream the table from disk. The patches here are what that took, measured step by
step; the numbers are in [`BENCHMARKS.md`](BENCHMARKS.md) and every change is
listed in [`PATCHES.md`](PATCHES.md).

## What is in here

| path | content |
|---|---|
| `README.md` | this file: what the build is, where every piece came from, how to verify |
| `INSTALL.md` | end-to-end build: upstream tree, recipe, our patches, native helper, first start |
| `MODEL.md` | the checkpoint: where to get it, what it must contain, how the pieces are laid out |
| `PATCHES.md` | registry of every change against "upstream + recipe", with origin and marker |
| `BENCHMARKS.md` | the measurements this configuration is justified by, and how they were taken |
| `TROUBLESHOOTING.md` | the failures we actually hit and what they look like from the outside |
| `base/` | the build base: the recipe patch `qwen38-ple-ssd.patch` (third-party, Apache-2.0), its licence and README, the tested version pins |
| `patches/` | twenty-two idempotent `install / check / remove` scripts (one logical change per file group) |
| `diffs/` | the same changes as unified diffs against a clean base, one file each |
| `overlay/` | the resulting nineteen files, paths 1:1 from the vLLM tree root (an alternative to running the patches) |
| `serve/` | the English launcher derived from the recipe's `serve.sh`, the production profile, and a background runner |
| `manifest/` | `files.sha256` (the nineteen changed files), `tree.sha256` (every file of the built tree) and `pip-freeze.txt` |
| `verify.sh` | one command that checks a tree against `manifest/` |

## Quick start

```bash
# 0. prerequisites: python 3.12, an NVMe volume for the model, sudo
# 1. upstream tree at the pinned commit (INSTALL.md has the exact URL)
git clone --depth 1 https://github.com/vllm-project/vllm
cd vllm && git checkout a5a30471ff2bb7f0824f2da10e358af98d304472
# 2. recipe: PLE-SSD offload + the native async I/O helper
patch -p1 < ../base/qwen38-ple-ssd.patch
bash scripts/install-runtime.sh          # ships with the recipe, see base/README-recipe.md
# 3. our patches, in order
bash ../patches/apply.sh "$PWD"
# 4. verify: nineteen files must match byte for byte
sha256sum -c ../manifest/files.sha256
# 5. serve
sudo bash ../serve/run.sh ../serve/awq-pp2.env
curl -s http://127.0.0.1:8008/health
```

`overlay/` is the shortcut: unpack it over the tree instead of step 3 (identical
result, same hashes).

## Where every piece came from

Three layers, applied in this order:

```
vllm-project/vllm @ a5a30471f           upstream (Apache-2.0)
  + base/qwen38-ple-ssd.patch           the third-party recipe (Apache-2.0)
    + patches/ 01..14, 16                our changes (Apache-2.0)
      + serve/                          the launcher and the production profile
```

**Upstream.** `vllm-project/vllm`, commit
`a5a30471ff2bb7f0824f2da10e358af98d304472` (`0.29.1rc1.dev402+ga5a30471f`). This
revision already carries the `qwen4_exp` model and the n-gram PLE embedding;
everything below builds on it.

**The recipe** - `iIIusi0n/qwen3.8-flash-next-cmp170hx` (Apache-2.0), file
`patches/qwen38-ple-ssd.patch`, shipped verbatim in `base/` (sha256
`09c4e13f…`). It is the piece that makes the whole thing possible on these
cards: it keeps the 95.37 GiB BF16 PLE table on local storage, reads it with
asynchronous Linux AIO through a small native helper (`ple_ssd_io.c`), and loads
everything else normally. It was written for **one** 64 GiB SM80 card and tested
at 15 GiB RAM + 31 GiB swap; this package is the two-card variant. The recipe's
own README, licence and file listing are reproduced under `base/` with credit.

**The pipeline-parallel recipe** - three edits from upstream vLLM PR
[#54709](https://github.com/vllm-project/vllm/pull/54709) (reported by
*@hackxixi* on eight CMP 170HX cards) are the backbone of patches 07/07b/09:
allow `PP>1` for a PLE-style model by checking where the PLE layer *lands*
instead of banning pipeline parallelism; treat the MTP drafter as a module that
receives its input rather than as "the first rank"; skip the PLE layer when
allocating the KV cache. Our tree needed all three; they are adapted to it, not
copied.

**The abliteration overlay** - `DominikBucko/qwen38-flash-next-2x3090`
(Apache-2.0), `runtime/vllm-overlay/.../abliteration.py`: patch 14 is a port of
that module (renamed for this tree, reworked for pipeline parallel and for CUDA
graphs). The refusal direction itself is **not** shipped; it belongs to the
Apache-2.0 release `orcarouter/Qwen3.8-Flash-Next-Uncensored` (Arditi et al.,
2024, *Refusal in Language Models Is Mediated by a Single Direction*) and is
expected to be supplied at runtime.

**The INT8 KV cache overlay** - `DominikBucko/qwen38-flash-next-3090`
(Apache-2.0), the same checkpoint on `sm_86`: patch 16 is a port of its
per-(token, head) INT8 KV cache to this tree's QSA attention path. The storage
side it builds on - the quantizing store kernel and the `INT8_PER_TOKEN_HEAD`
quant mode - is already in the pinned upstream tree.

**The model lineage** - `Qwen/Qwen3.8-Flash-Next` (official, Qwen Community
License) → `orcarouter/Qwen3.8-Flash-Next-Uncensored` (abliterated fine-tune,
Apache-2.0) → AutoRound quantization → the AWQ-INT4 checkpoint this build
serves. Prefix-cache/scheduling advice we measured against comes from
`wtdcode/vllm-backport` (`--prefix-match-unit`); remaining MTP risks are tracked
in upstream issues #46994 and #52295.

**Our changes** - everything else: mixed shards, the FP8-PLE reserve path, the
pipeline weight filter, the non-PLE loader switch, the abliteration and INT8 KV
ports, and the launcher/profile plumbing. `PATCHES.md` gives the one-line origin of
each.

## Target hardware and software

| item | value |
|---|---|
| GPUs | 2x NVIDIA CMP 170HX (GA100, `sm_80`, 64 GiB HBM each), PCIe Gen2 x4, no peer-to-peer link |
| Parallelism | `PP=2` with the layer split `26,22`, expert parallel, async scheduling, MTP=2 |
| Host | x86-64, 32 GiB RAM plus swap, one NVMe namespace holding both the checkpoint and the PLE table |
| OS | Ubuntu Server, bare metal (no containers) |
| Driver | NVIDIA open kernel module 610.57.04 |
| Upstream | `vllm-project/vllm` @ `a5a30471ff2bb7f0824f2da10e358af98d304472` = `0.29.1rc1.dev402+ga5a30471f` |
| Python / torch | 3.12 / `2.13.0+cu130`, `triton 3.7.1`, `transformers 5.17.0`, `flashinfer-python 0.6.18.post1`; the full set is `manifest/pip-freeze.txt` |
| CUDA extensions | not compiled here: `vllm/_C*.abi3.so` and the flash-attention libraries come from the precompiled wheel of the same commit |
| Model | `Qwen3.8-Flash-Next-AWQ-INT4`, ~176 GB in 38 shards, not included - see `MODEL.md` |

## Measured effect of the patch set

| metric | before | after | made by |
|---|---|---|---|
| engine start, PP2 | 680 / 628 s, occasionally much worse | **150-166 s** | patch 06 |
| bytes read from disk during start | 124 GiB | **96 GiB** | patches 05, 05b-05d |
| prefill @32k prompt | 7522 tok/s (`MAX_BATCH_TOKENS=2048`) | **8330 tok/s** (4096) | profile |
| decode, single request | 97 tok/s | **121 tok/s** | profile |
| end-to-end acceptance (65 536 token quota, thinking on) | 567 s | **45 507 tokens in 358 s (~127 tok/s)**, `finish=stop` | all |
| KV cache at ctx 262 144 | 840 019 tok (split 24,24, BF16 KV) | **1 352 663 tok** (26,22, `GPU_MEMORY=0.90`; the same profile with BF16 KV gives 744 431) | profile + patch 16 |

Method, repeats and the counter-examples are in `BENCHMARKS.md`. A single
measurement on this host proves nothing: decode moves 10-20 % run to run, and
before patch 06 the start time was bimodal.

## Limitations

* `sm_80` has no FP8/FP4 tensor cores: patches 03/04 are a reserve path that
  dequantizes an FP8 PLE table on the CPU, and NVFP4/MXFP4 kernels are not
  available on these cards at all.
* The KV cache is INT8 with a scale per (token, head) - patch 16. FP8 KV does run
  here, but only as an `fp8e4b15` emulation and it costs about a third of decode
  (`BENCHMARKS.md`, "KV cache quantization on SM80").
* No CUDA source is compiled by this package: the `*.abi3.so` extensions and
  flash-attention come from the precompiled wheel of the pinned commit.
* The launcher and the profile default to the paths of the machine this was
  measured on (`/srv/models/vllm/...`); all of them can be overridden, see
  `INSTALL.md`.
* No model weights and no secrets are shipped: `serve/awq-pp2.env` carries an
  empty `API_KEY` and an empty `QWEN38_ABLITERATION`.
* Everything here is English, including the comments the patches insert into the
  tree.

## Licence

Apache-2.0, the same as upstream vLLM, the recipe and the abliteration overlay.
vLLM code is (c) its authors; the recipe in `base/` is (c) iIIusi0n (see
`base/recipe-LICENSE`); the abliteration module in `patches/14_abliteration.py`
is derived from DominikBucko's overlay. Model weights carry their own licence
(Qwen Community License for the base, Apache-2.0 for the OrcaRouter fine-tune).
See `LICENSE`.