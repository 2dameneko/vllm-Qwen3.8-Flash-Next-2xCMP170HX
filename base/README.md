# base/ - the build base (third-party recipe)

Everything in this directory comes from the recipe repository
**`iIIusi0n/qwen3.8-flash-next-cmp170hx`** (Apache-2.0) and is shipped verbatim.
It is not ours; see `recipe-LICENSE` and `NOTICE` in the release root.

| file | what it is |
|---|---|
| `qwen38-ple-ssd.patch` | the recipe: PLE table on local storage with asynchronous Linux AIO, plus the native helper `ple_ssd_io.c`. Applies to upstream vLLM commit `a5a30471ff2bb7f0824f2da10e358af98d304472` (13 files, 2 of them new) |
| `README-recipe.md` | the recipe's own README (renamed here; the text is unchanged): what it does, the model lineage, the measured results, the requirements |
| `recipe-LICENSE` | Apache-2.0, as published with the recipe |
| `RECIPE_LISTING.txt` | the recipe repository's file listing, so you can see what else it ships (`scripts/install-runtime.sh`, `serve.sh`, `docs/GUIDE.md`, benchmark scripts) |
| `tested-constraints.txt` | the version pins the recipe was measured with (torch, transformers, triton, flashinfer, ...) |
| `runtime-snapshot.txt` | the recipe's runtime snapshot (python/CUDA/driver lines) |

## Pinned

```
sha256  qwen38-ple-ssd.patch
        09c4e13fbd2b77169a7f8cb28d36242cc6e3916fe9c5a3c25d8e90101a38ccb0
size    55767 bytes
```

Check it after downloading, before applying - the whole build rests on this file
being the revision that was measured:

```bash
sha256sum base/qwen38-ple-ssd.patch
```

## How it is used

```bash
cd /path/to/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
patch -p1 < /path/to/release/base/qwen38-ple-ssd.patch
bash scripts/install-runtime.sh        # comes from the recipe repo, not from here
```

`patch -p1` must report every file as clean; a `.rej` file means a different
upstream revision, and `verify.sh` warns about leftovers of that kind. The
recipe's `install-runtime.sh` (not shipped here, see `RECIPE_LISTING.txt`) builds
the venv with `uv`, installs the precompiled wheel of the same commit plus the
tree in editable mode, and compiles `ple_ssd_io.so`.

Two notes from our own use of it:

* The wheel URL (`https://wheels.vllm.ai/<revision>/...`) expires after a while;
  when it 404s, build the CUDA extensions from the same commit instead.
* `ple_ssd_io.so` ends up wherever the script puts it - record the path, the
  launcher needs it as `PLE_SSD_LIB`.