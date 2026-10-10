# patches/ - what this directory changes

Twenty scripts, applied in the order listed by `apply.sh`. Each one is a small
idempotent text edit: `install | check | remove`, a `<file>.bak-<patch>` copy
next to every file it touches, and an exact anchor - if the anchor is missing the
script says so and changes nothing. Nothing here compiles, downloads or runs
anything; the whole set is plain text rewriting.

```bash
# apply to the tree built from ../base (see ../INSTALL.md)
bash patches/apply.sh /path/to/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472
bash patches/apply.sh /path/to/vllm-a5a30471ff2bb7f0824f2da10e358af98d304472 --check

# one patch at a time
python3 patches/09_mtp_pp.py install /path/to/vllm
```

See `../PATCHES.md` for the registry (what each one does, why, and where the idea
came from) and `../diffs/` for the resulting change as a unified diff.

## Notes that save time

* **Order is not cosmetic.** `04_ple_ssd_fp8_iter` edits text that
  `02_ple_ssd_mixed` inserted, and `06_ple_ssd_nonple_loader` extends the tail of
  the iterator that `05_ple_ssd_ppfilter` reworked. `apply.sh` fixes the order;
  by hand, follow the file names.
* **The launcher is not patched here.** The recipe's launcher edits (PP flags,
  the export allowlist, `RUN_ENV`, the API key, the abliteration switch) are
  already inside `../serve/serve.sh`, which ships as an English file derived from
  the recipe's `serve.sh`.
* **LF only.** Anchors are compared byte for byte, so a tree converted to CRLF
  will not match; the scripts write LF back.
* **`check` before `install` on an unknown tree.** `check` classifies every edit
  as already-present, would-apply or anchor-missing and writes nothing.
* **A few scripts say `neither` on a fully patched tree, and that is correct.**
  Patches 02, 05 and 10 edit text that a *later* patch rewrites (the FP8 iterator
  reworks the mixed-shard branch, the loader switch extends the iterator tail,
  the AutoRound branch extends the compressed-tensors branch), so on the end
  state neither their "before" nor their "after" text exists verbatim. The line
  is a note, not a failure; the end state is checked by `../verify.sh` (hashes
  and markers), not by these scripts.
* **`remove` restores from the `.bak-<patch>` files**, so it only works where
  `install` ran. `git` (or the pristine base plus `apply.sh`) is a cleaner way
  back.
* `05b..05d` are the three files of one logical change, and so are
  `07`/`07b` and `13`/`13b`; the numbering keeps the apply order readable rather
  than inventing a patch count.

## Result

After `apply.sh` the fifteen files it touches are byte-identical to the tree this
package was built from:

```bash
# from the release root, on the tree you just patched
sha256sum -c manifest/files.sha256      # 15 files, all OK
```

The same result without running the scripts: unpack `overlay/` over the tree.