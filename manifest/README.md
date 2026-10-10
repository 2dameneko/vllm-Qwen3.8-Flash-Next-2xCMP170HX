# manifest/ - hashes and versions of the build

| file | what it is |
|---|---|
| `files.sha256` | sha256 of the **fifteen files this release changes**, paths relative to the vLLM tree root. This is the file `verify.sh` and `sha256sum -c` use |
| `tree.sha256` | sha256 of **every file of the built tree** on the reference machine (10 602 files), generated with `LC_ALL=C find . -type f -print0 \| LC_ALL=C sort -z \| xargs -0 sha256sum` from the tree root |
| `pip-freeze.txt` | the engine venv's package set (`python -m pip freeze`), 89 lines; the versions that matter are also in `../base/tested-constraints.txt` and `../README.md` |

## Use

```bash
# the fifteen patched files, from the tree root
cd /path/to/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
sha256sum -c /path/to/release/manifest/files.sha256

# or the wrapper, which also compiles the files and greps for patch markers
bash /path/to/release/verify.sh "$PWD"

# a spot check of any other file of the tree
grep ' vllm/models/qwen4_exp/nvidia/abliteration.py$' /path/to/release/manifest/tree.sha256 \
  | sha256sum -c -
```

`tree.sha256` was taken on the machine this package was measured on, so it
contains the whole built tree - including the recipe's `ple_ssd_io.so` and any
artefact a build leaves behind. A tree built elsewhere will differ in those
files; compare the specific paths you care about rather than the whole list.

## What the hashes are worth

* The fifteen files in `files.sha256` are the ones that decide behaviour: they
  are what `patches/` produce and what `overlay/` contains.
* `tree.sha256` also pins the pieces this package does *not* ship: the CUDA
  extensions (`vllm/_C*.abi3.so`, the flash-attention libraries) and the model
  files, so you can tell a difference in those from a difference in our patches.
* The recipe patch itself is pinned separately:
  `sha256(qwen38-ple-ssd.patch) = 09c4e13fbd2b77169a7f8cb28d36242cc6e3916fe9c5a3c25d8e90101a38ccb0`.
* Anything under `manifest/` is generated, not hand-edited, so it can be
  regenerated from a tree at any time.