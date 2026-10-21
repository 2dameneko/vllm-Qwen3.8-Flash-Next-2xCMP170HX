# TROUBLESHOOTING.md - the failures we actually hit

Every entry below happened on this build, and each one is written so you can
recognise it from the outside. Symptoms first, then the cause.

## The engine looks healthy but a setting does nothing

**Symptom:** the profile sets `VLLM_PLE_NONPLE_LOADER=instanttensor`, the engine
starts, no error anywhere, but the start still takes 600+ s and the log has no
`PLE non-PLE loader=` line.

**Cause:** the profile is `source`d, which does not export, and `serve.sh`
exports only the variables in its allowlist. A key that is not in that list (or
spelled differently) never reaches the process - silently.
**Check:** the launcher banner prints the state of each switch, and for a running
engine: `sudo tr '\0' '\n' < /proc/$(cat /run/flashnext-engine.pid)/environ | grep VLLM_`.

## Slow start, and it is not the drive

**Symptom:** start takes 600-760 s instead of 150-200 s; `iostat` shows high
`%util` and seconds-long `await`; the host shows heavy swap activity.

**Cause:** loading a 176 GB checkpoint through mmap keeps tens of gigabytes of
page cache resident on a 32 GiB host, so the working set goes to swap.
**Fix:** keep `VLLM_PLE_NONPLE_LOADER=instanttensor` (patch 06). Do **not** "fix"
it with `--safetensors-load-strategy=eager`: that was measured at 491 s versus
343 s on the same day, with 63 GiB swapped.
**Diagnose:** sample `/proc/diskstats` and `vmstat` during the start; the loader
line in the log, page-cache peak and swap samples tell the two modes apart
immediately.

## `PLE SSD offload currently requires BF16 embeddings`

The table is not BF16 (an FP8-PLE checkpoint served through the SSD path). Either
serve a BF16-PLE checkpoint or convert the table; patches 03/04 have a reserve
path for FP8 tables that dequantizes on the host, but it costs RAM and `sm_80`
has no FP8 compute.

## `PLE SSD requires PLE tensors in dedicated shards`

Your checkpoint interleaves PLE tensors with ordinary weights in a tail shard and
patch 02 is not applied. Apply the whole `patches/` set; patch 02 is what accepts
mixed shards.

## `NotImplementedError: Qwen4Exp PLE embedding does not support quantization config CompressedTensorsConfig`

Patch 10 is missing. AWQ-INT4 checkpoints are marked compressed-tensors, and
their PLE tensors are not quantized at all, so the PLE method must map that
config to the unquantized path.

## `RuntimeError: N-gram PLE embedding currently requires pipeline_parallel_size=1`

Patches 07/07b are missing, or they are applied but the PLE layer lands on a rank
that does not own it: `VLLM_PP_LAYER_PARTITION` must place `ple_layer_ids`
(typically layer 2) on **rank 0**. The error text tells you which layers are
outside rank 0 - raise rank 0's share (for two cards, `26,22`).

## `StopIteration` in `allocate_kv_cache`

Patch 08 is missing. The PLE layer has no KV cache at all, and with PP>1 it ends
up in the tensor list without belonging to any KV cache group.

## `NotImplementedError: Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache`

The profile asks for `KV_DTYPE=int8_per_token_head` but patch 16 is not in the
tree: the unpatched QSA backend rejects the per-token-head dtypes (so does the
cache-owner check, a few lines later). Apply `patches/16_kv_int8_qsa.py`, or drop
`KV_DTYPE` from the profile to run BF16 - the pool then holds 744 431 tokens
instead of 1 352 663 at ctx 262 144 (`BENCHMARKS.md`).

## `AssertionError` in the MTP drafter under PP

Patch 09 is missing: the drafter is built on the last stage but the runner calls
it with `hidden_states` and no intermediate tensors, so the "am I the first rank"
test is the wrong question.

## `400` on every `/v1` request after enabling the API key

The request is missing `Authorization: Bearer <key>`. When `API_KEY` is non-empty
the engine starts with `--api-key` and `/v1`, `/v2`, `/inference` and `/cohere`
require the header (`/health` and `/metrics` stay open). A key change needs an
engine restart - vLLM does not pick it up live.

## The model "is not loading" and the supervisor killed it

**Symptom:** the loader reports the engine dead a few seconds after start, while
the log shows it was still loading; a framework that manages the process then
kills the whole process group.

**Cause:** liveness decided by a single `pgrep -f "vllm serve"` right after the
fork. The process with that command line only exists after the `exec` at the end
of `serve.sh`, so on a cold page cache the probe runs too early.
**Fix:** `run.sh`/`serve.sh` write `/run/flashnext-engine.pid` before the exec
and treat "no pid file yet" as normal for a grace period (60 s here), then decide
by `kill -0` on that pid. If you write your own supervisor, do the same: the pid
file plus `kill -0`, never a single `pgrep`.

## A 16 384-token request returns `content: null`

Not a bug: with thinking mode on, the quota is spent on `reasoning` tokens
first. Ask for 65 536 and read `message.reasoning`.

## `dmesg` shows Xid and the engine died

Count GPU Xid correctly: search for `NVRM: Xid`, not for `Xid` - a network
driver prints its own XID and it has nothing to do with the GPU. On this build
three Xid events were seen in ~45 hours on one card, all with an older
checkpoint/loader combination; the current configuration has been clean through
long acceptance runs, but keep an eye on it and record the counter before and
after every long run.

## `pip install` broke the venv

The recipe's venv is managed by `uv` from a manifest; `uv sync` (or the recipe's
install script) will remove packages that are not in that manifest. If you add a
package by hand, reinstall it after any `sync`.

## The status of this package's files is unclear

`git` never sees the vLLM tree (a flat copy, no repository), so "did I change
anything?" is answered by hashes: `manifest/files.sha256` for the nineteen patched
files, `manifest/tree.sha256` for the whole tree, and `verify.sh` as the one
command that checks both plus the patch markers.