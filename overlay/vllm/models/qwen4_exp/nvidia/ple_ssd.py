# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous PLE row reads from the original safetensors checkpoint."""

import bisect
import copy
import ctypes
import json
import os
import struct
import threading
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from .ngram_embedding import Qwen4ExpNGramEmbedding, Qwen4ExpPLEEmbedding

logger = init_logger(__name__)


def ple_ssd_weights_iterator(
    files, use_tqdm, strategy, local_expert_ids=None, local_layers=None
):
    """Avoid mmap's commit charge for checkpoint files containing only PLE.

    *local_layers* is handed over to ``safetensors_weights_iterator``
    so tensors of layers owned by another pipeline stage are never
    read at all (see ep_weight_filter.should_skip_weight).
    """
    from vllm.model_executor.model_loader.ep_weight_filter import (
        should_skip_weight,
    )
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
    )

    regular_files = []
    for path in files:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(f.read(header_len))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if tensors and all(".ngram_embedding.shard_" in k for k in tensors):
            for name, info in tensors.items():
                if info["dtype"] not in ("BF16", "F8_E4M3"):
                    raise ValueError("PLE SSD requires BF16/F8_E4M3 checkpoint tensors")
                ple_dtype = (
                    torch.bfloat16
                    if info["dtype"] == "BF16"
                    else torch.float8_e4m3fn
                )
                yield (
                    name,
                    torch.empty(info["shape"], dtype=ple_dtype, device="meta"),
                )
        else:
            # PATCH: mixed shard (PLE + regular weights) is allowed.
            # PLE is yielded as meta - PLESSDTable reads it from disk
            # by data_offsets. The other tensors are read with plain safe_open,
            # otherwise they would be lost or land in VRAM together with PLE.
            if any(".ngram_embedding.shard_" in k for k in tensors):
                from safetensors import safe_open

                with safe_open(path, framework="pt") as fp:
                    for name, info in tensors.items():
                        if ".ngram_embedding.shard_" in name:
                            if info["dtype"] not in ("BF16", "F8_E4M3"):
                                raise ValueError(
                                    "PLE SSD requires BF16/F8_E4M3 checkpoint tensors"
                                )
                            ple_dtype = (
                                torch.bfloat16
                                if info["dtype"] == "BF16"
                                else torch.float8_e4m3fn
                            )
                            yield (
                                name,
                                torch.empty(
                                    info["shape"],
                                    dtype=ple_dtype,
                                    device="meta",
                                ),
                            )
                        elif not should_skip_weight(
                            name, local_expert_ids, local_layers
                        ):
                            yield (name, fp.get_tensor(name))
            else:
                regular_files.append(path)
    import os as _os

    _nonple = (
        _os.environ.get("VLLM_PLE_NONPLE_LOADER", "safetensors").strip() or "safetensors"
    )
    if _nonple == "safetensors":
        yield from safetensors_weights_iterator(
            regular_files, use_tqdm, strategy, local_expert_ids, local_layers
        )
        return

    # --- PATCH #13: direct I/O instead of mmap for non-PLE files --------------
    # mmap (page faults) starves the NVMe, see docs/benchmarks.md section 4d.
    # We run SINGLE-RANK (process_group=None / SingleGroup): with PP2 the list
    # of files differs per rank, the libraries' TP mode does not apply here.
    import logging as _logging

    _log = _logging.getLogger(__name__)
    _kept = []
    for _path in regular_files:
        with open(_path, "rb") as _f:
            _hlen = struct.unpack("<Q", _f.read(8))[0]
            _header = json.loads(_f.read(_hlen))
        if any(
            not should_skip_weight(_n, local_expert_ids, local_layers)
            for _n in _header
            if _n != "__metadata__"
        ):
            _kept.append(_path)
    _log.info(
        "PLE non-PLE loader=%s: reading %d of %d files (PP/EP filter by header)",
        _nonple,
        len(_kept),
        len(regular_files),
    )
    if not _kept:
        _log.info("PLE non-PLE loader=%s: no non-PLE files of ours - skipping", _nonple)
        return

    if _nonple == "instanttensor":
        import instanttensor

        _device = torch.cuda.current_device()
        with instanttensor.safe_open(
            _kept,
            framework="pt",
            device=_device,
            process_group=None,
            copy=True,
        ) as _fp:
            for _name, _tensor in _fp.tensors():
                if should_skip_weight(_name, local_expert_ids, local_layers):
                    continue
                yield (_name, _tensor)
    elif _nonple == "fastsafetensors":
        from fastsafetensors import SingleGroup
        from fastsafetensors.parallel_loader import ParallelLoader

        _device = torch.device(f"cuda:{torch.cuda.current_device()}")
        _pl = ParallelLoader(
            pg=SingleGroup(),
            hf_weights_files=sorted(_kept),
            use_tqdm_on_load=bool(use_tqdm),
            device=str(_device),
            nogds=True,
        )
        try:
            for _name, _tensor in _pl.iterate_weights():
                if should_skip_weight(_name, local_expert_ids, local_layers):
                    continue
                yield (_name, _tensor)
        finally:
            _pl.close()
    else:
        raise ValueError(
            "VLLM_PLE_NONPLE_LOADER must be safetensors|instanttensor|"
            f"fastsafetensors, got {_nonple!r}"
        )


class PLESSDNativeReader:
    """Bounded Linux direct AIO; the native call releases the Python GIL."""

    def __init__(self, table, library: str, depth: int) -> None:
        if not 1 <= depth <= 4096 or not 1 <= table.row_bytes <= 4096:
            raise ValueError("PLE AIO requires depth and row bytes in [1, 4096]")
        self._lib = ctypes.CDLL(library, use_errno=True)
        self._lib.rows_open.argtypes = [ctypes.c_uint]
        self._lib.rows_open.restype = ctypes.c_void_p
        self._lib.rows_close.argtypes = [ctypes.c_void_p]
        self._lib.rows_close.restype = None
        self._lib.rows_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self._lib.rows_read.restype = ctypes.c_int
        self._reader = self._lib.rows_open(depth)
        self._fds: dict[int, int] = {}
        if not self._reader:
            raise OSError(ctypes.get_errno(), "PLE io_setup failed")
        try:
            for fd in table._fds.values():
                self._fds[fd] = os.open(
                    f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_DIRECT
                )
            self._parts = np.asarray(table._parts, dtype=np.int64)
            self._parts[:, 2] = [self._fds[fd] for fd in self._parts[:, 2]]
            self.row_bytes = table.row_bytes
        except BaseException:
            self.close()
            raise

    def read(self, rows: list[int]) -> list[tuple[int, bytes]]:
        ids = np.asarray(rows, dtype=np.int64)
        parts = self._parts[np.searchsorted(self._parts[:, 0], ids, side="right") - 1]
        fds = np.ascontiguousarray(parts[:, 2], dtype=np.int32)
        offsets = np.ascontiguousarray(
            parts[:, 3] + (ids - parts[:, 0]) * self.row_bytes, dtype=np.uint64
        )
        out = np.empty((len(ids), self.row_bytes), dtype=np.uint8)
        status = self._lib.rows_read(
            self._reader,
            fds.ctypes.data,
            offsets.ctypes.data,
            len(ids),
            self.row_bytes,
            out.ctypes.data,
        )
        if status:
            raise OSError(-status, "PLE asynchronous SSD read failed")
        return [(row, out[i].tobytes()) for i, row in enumerate(rows)]

    def close(self) -> None:
        if self._reader:
            self._lib.rows_close(self._reader)
            self._reader = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class PLESSDTable:
    """Read immutable BF16 rows with bounded concurrent I/O and an LRU cache.

    Native AIO or disk workers release the GIL while reading. Neither the
    full table nor whole checkpoint shards are loaded into RAM.
    """

    def __init__(
        self,
        model_path: str,
        prefix: str,
        rows: int,
        dim: int,
        workers: int = 8,
        cache_mb: int = 128,
        native_library: str | None = None,
        io_depth: int = 256,
    ) -> None:
        if workers < 1 or cache_mb < 0:
            raise ValueError(
                "PLE SSD workers must be positive and cache_mb nonnegative"
            )
        # PATCH: the dtype comes from the shard headers (done below),
        # itemsize: BF16 -> 2, F8_E4M3 -> 1.
        self.dtype: str | None = None
        self.is_fp8 = False
        self.weight_scale = 1.0
        self.row_bytes = dim * 2
        self.rows = rows
        self.workers = workers
        self.cache_limit = cache_mb * 1024 * 1024 // (self.row_bytes + 128)
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.hits = self.reads = 0
        self._condition = threading.Condition()
        self._reading = False
        self._waiting = 0
        self._fds: dict[str, int] = {}
        self._parts: list[tuple[int, int, int, int]] = []
        root = Path(model_path)
        with open(root / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        # HF and vLLM use different wrappers around the same decoder layers.
        suffix = prefix[prefix.index("layers.") :] + ".shard_"
        keys = [k for k in weight_map if suffix in k and k.endswith(".weight")]
        keys.sort(key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
        if not keys:
            raise ValueError(f"No checkpoint PLE shards match {prefix}")
        headers = {}
        start = 0
        try:
            for i, key in enumerate(keys):
                if int(key.rsplit("shard_", 1)[1].split(".")[0]) != i:
                    raise ValueError("PLE checkpoint shard indices must be contiguous")
                filename = weight_map[key]
                if filename not in headers:
                    path = root / filename
                    with open(path, "rb") as f:
                        header_len = struct.unpack("<Q", f.read(8))[0]
                        if header_len > 100_000_000:
                            raise ValueError(f"Invalid safetensors header: {path}")
                        headers[filename] = (
                            json.loads(f.read(header_len)),
                            8 + header_len,
                        )
                    fd = os.open(path, os.O_RDONLY)
                    self._fds[filename] = fd
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                header, data_start = headers[filename]
                info = header[key]
                shape = info["shape"]
                if info["dtype"] not in ("BF16", "F8_E4M3") or len(shape) != 2 or shape[1] != dim:
                    raise ValueError(
                        f"PLE SSD requires BF16/F8_E4M3 [rows, {dim}]: {key}"
                    )
                if self.dtype is None:
                    self.dtype = info["dtype"]
                    self.is_fp8 = info["dtype"] == "F8_E4M3"
                    self.row_bytes = dim * (2 if not self.is_fp8 else 1)
                    self.cache_limit = cache_mb * 1024 * 1024 // (
                        self.row_bytes + 128
                    )
                elif info["dtype"] != self.dtype:
                    raise ValueError(
                        f"PLE shards mix checkpoint dtypes: {key}"
                    )
                begin, end = info["data_offsets"]
                fd = self._fds[filename]
                if end - begin != shape[0] * self.row_bytes:
                    raise ValueError(f"Invalid PLE tensor byte range: {key}")
                if data_start + end > os.fstat(fd).st_size:
                    raise ValueError(f"Incomplete PLE checkpoint: {filename}")
                self._parts.append((start, start + shape[0], fd, data_start + begin))
                start += shape[0]
            if start != rows:
                raise ValueError(f"PLE table has {start} rows, expected {rows}")
            if self.is_fp8:
                self.weight_scale = self._read_weight_scale(
                    weight_map, prefix, headers, root
                )
        except BaseException:
            for fd in self._fds.values():
                os.close(fd)
            raise
        self._starts = [p[0] for p in self._parts]
        self._native = None
        self._pool = None
        try:
            if native_library:
                self._native = PLESSDNativeReader(self, native_library, io_depth)
            else:
                self._pool = ThreadPoolExecutor(
                    workers, thread_name_prefix="ple-ssd-io"
                )
        except BaseException:
            for fd in self._fds.values():
                os.close(fd)
            raise

    @staticmethod
    def _bf16_bytes_to_float(data: bytes) -> float:
        """Little-endian BF16 scalar to Python float (no numpy/torch)."""
        u16 = int.from_bytes(data, "little")
        sign = -1.0 if (u16 >> 15) & 1 else 1.0
        exp = (u16 >> 7) & 0xFF
        man = u16 & 0x7F
        if exp == 0:
            return sign * (man / 128.0) * (2.0 ** -126)
        if exp == 0xFF:
            return sign * float("inf") if man == 0 else float("nan")
        return sign * (1.0 + man / 128.0) * (2.0 ** (exp - 127))

    def _read_weight_scale(self, weight_map, prefix, headers, root) -> float:
        """Read the global per-tensor FP8 weight_scale from the checkpoint."""
        scale_suffix = prefix[prefix.index("layers."):] + ".weight_scale"
        keys = [k for k in weight_map if k.endswith(scale_suffix)]
        if len(keys) != 1:
            raise ValueError(
                f"FP8 PLE checkpoint must have exactly one {scale_suffix}"
            )
        key = keys[0]
        filename = weight_map[key]
        if filename not in headers:
            path = root / filename
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                if header_len > 100_000_000:
                    raise ValueError(f"Invalid safetensors header: {path}")
                headers[filename] = (
                    json.loads(f.read(header_len)),
                    8 + header_len,
                )
            fd = os.open(path, os.O_RDONLY)
            self._fds[filename] = fd
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        header, data_start = headers[filename]
        info = header[key]
        begin, end = info["data_offsets"]
        if info["dtype"] != "BF16" or end - begin != 2:
            raise ValueError(f"PLE weight_scale must be a BF16 scalar: {key}")
        fd = self._fds[filename]
        data = os.pread(fd, end - begin, data_start + begin)
        if len(data) != 2:
            raise OSError(f"Short PLE weight_scale read: {key}")
        return self._bf16_bytes_to_float(data)

    def _read_rows(self, ids: list[int]) -> list[tuple[int, bytes]]:
        result = []
        for row in ids:
            start, _, fd, offset = self._parts[
                bisect.bisect_right(self._starts, row) - 1
            ]
            data = os.pread(fd, self.row_bytes, offset + (row - start) * self.row_bytes)
            if len(data) != self.row_bytes:
                raise OSError(f"Short PLE SSD read for row {row}")
            result.append((row, data))
        return result

    def read(
        self, ids: np.ndarray, output: np.ndarray, *, prefetch: bool = False
    ) -> None:
        """Serialize cache/I/O access, giving demanded rows priority over read-ahead."""
        with self._condition:
            if not prefetch:
                self._waiting += 1
            try:
                while self._reading or (prefetch and self._waiting):
                    self._condition.wait()
                self._reading = True
            finally:
                if not prefetch:
                    self._waiting -= 1
        try:
            self._read(ids, output)
        finally:
            with self._condition:
                self._reading = False
                self._condition.notify_all()

    def _read(self, ids: np.ndarray, output: np.ndarray) -> None:
        """Fill a byte buffer in request order, preserving duplicate rows."""
        flat = ids.reshape(-1)
        if output.shape != (flat.size, self.row_bytes) or output.dtype != np.uint8:
            raise ValueError("PLE SSD output must be uint8 [number of IDs, row bytes]")
        unique = np.unique(flat).tolist()
        if unique and (unique[0] < 0 or unique[-1] >= self.rows):
            raise IndexError("PLE row ID outside checkpoint table")
        found = {}
        missing = []
        for row in unique:
            cached = self.cache.get(row)
            if cached is None:
                missing.append(row)
            else:
                found[row] = cached
                self.cache.move_to_end(row)
        self.hits += len(found)
        self.reads += len(missing)
        if missing:
            # Keep only workers tasks queued, even for a large prefill.
            active_workers = min(self.workers, len(missing))
            if len(missing) < 1024:
                active_workers = min(active_workers, 8)
            results: Iterable[list[tuple[int, bytes]]]
            if self._native is not None:
                results = [self._native.read(missing)]
            else:
                assert self._pool is not None
                chunks = [missing[i::active_workers] for i in range(active_workers)]
                results = self._pool.map(self._read_rows, chunks)
            for result in results:
                for row, data in result:
                    found[row] = data
                    if self.cache_limit:
                        self.cache[row] = data
                        if len(self.cache) > self.cache_limit:
                            self.cache.popitem(last=False)
        data = b"".join(found[row] for row in flat.tolist())
        output.reshape(-1)[:] = np.frombuffer(data, dtype=np.uint8)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._native is not None:
            self._native.close()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class PLEPromptPrefetcher:
    """Warm a bounded prefix of known prompt rows while CUDA executes chunks."""

    def __init__(self, ngram: Qwen4ExpNGramEmbedding, limit: int) -> None:
        self.table = ngram.ngram_embedding._table
        self.ngram = copy.copy(ngram)
        self.ngram._modules = {}
        self.ngram._parameters = {}
        self.ngram._buffers = {k: v.detach().cpu() for k, v in ngram._buffers.items()}
        self.limit = min(limit, self.table.cache_limit // (2 * ngram.ngram_heads))
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ple-prompt")
        self._future: Future | None = None
        self._stop = threading.Event()
        self._request_id: str | None = None

    def submit(self, request_id: str, tokens: list[int], start: int) -> None:
        if min(self.limit, len(tokens) - start) < 2048:
            return
        if self._future is not None:
            if not self._future.done():
                return
            self._future.result()
        history = min(start, self.ngram.ngram_size - 1)
        tokens = tokens[start - history : start + self.limit]
        self._request_id = request_id
        self._stop.clear()
        self._future = self._pool.submit(self._prefetch, tokens, history)

    @torch.inference_mode()
    def _prefetch(self, tokens: list[int], history: int) -> None:
        ngram = self.ngram
        context = [ngram.eos_token_id] * (ngram.ngram_size - 1 - history)
        context += tokens[:history]
        ids = ngram.compute_ngram_ids(
            torch.tensor(tokens[history:]),
            torch.tensor([0, len(tokens) - history]),
            torch.tensor([context]),
        ).numpy()
        for start in range(0, len(ids), 256):
            if self._stop.is_set():
                return
            batch = ids[start : start + 256]
            output = np.empty((batch.size, self.table.row_bytes), dtype=np.uint8)
            self.table.read(batch, output, prefetch=True)

    def cancel(self, request_id: str) -> None:
        if request_id == self._request_id:
            self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=True)
        if self._future is not None:
            self._future.result()


class Qwen4ExpPLESSDEmbedding(Qwen4ExpPLEEmbedding):
    """Prefetch SSD rows while the preceding decoder layer executes on CUDA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        config = get_current_vllm_config()
        # PATCH: TP>1 is allowed - every rank keeps its own PLESSDTable.
        # DP=1 is required: with DP>1 the ranks would end up with independent copies.
        if config.parallel_config.data_parallel_size != 1:
            raise ValueError("PLE SSD offload requires DP=1")
        super().__init__(*args, **kwargs)
        if self.weight.dtype != torch.bfloat16:
            raise ValueError("PLE SSD offload currently requires BF16 embeddings")
        options = config.additional_config
        self._table = PLESSDTable(
            config.model_config.model,
            kwargs["prefix"],
            self.org_vocab_size,
            self.embedding_dim,
            int(options.get("ple_ssd_workers", 8)),
            int(options.get("ple_ssd_cache_mb", 128)),
            options.get("ple_ssd_native_library"),
            int(options.get("ple_ssd_io_depth", 256)),
        )
        self.prompt_prefetcher: PLEPromptPrefetcher | None = None
        self._device = torch.device("cuda", torch.cuda.current_device())
        shape = (kwargs["max_total_tokens"], kwargs["num_ngram_heads"])
        self._ids = torch.empty(shape, dtype=torch.int64, device="cpu", pin_memory=True)
        self._device_ids = torch.empty_like(self._ids, device=self._device)
        self._host = torch.empty(
            (*shape, self.embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._gpu = torch.empty_like(self._host, device=self._device)
        self._stream = torch.cuda.Stream(device=self._device)
        self._ids_ready = torch.cuda.Event()
        self._copy_ready = torch.cuda.Event()
        self._previous_use = torch.cuda.Event()
        self._previous_use.record()
        self._pending: Future | None = None
        self._coordinator = ThreadPoolExecutor(1, thread_name_prefix="ple-ssd-prefetch")
        self._loaded_starts: set[int] = set()
        logger.info(
            "PLE SSD: %.2f GiB on disk, %d I/O workers, %d MiB row cache",
            self.org_vocab_size * self._table.row_bytes / 2**30,
            self._table.workers,
            int(options.get("ple_ssd_cache_mb", 128)),
        )
        if self._table._native is not None:
            logger.info(
                "PLE SSD uses native Linux AIO, queue depth %d",
                int(options.get("ple_ssd_io_depth", 256)),
            )

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")

    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        if checkpoint_start is None or loaded_weight.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise ValueError("PLE SSD requires checkpoint-split BF16/FP8 weights")
        self._loaded_starts.add(checkpoint_start)

    def _read_and_copy(self, tokens: int) -> None:
        self._ids_ready.synchronize()
        ids = self._ids[:tokens].numpy()
        if self._table.is_fp8:
            # Raw F8_E4M3 bytes -> temp buffer, dequantized on the CPU.
            raw = np.empty((ids.size, self._table.row_bytes), dtype=np.uint8)
            self._table.read(ids, raw)
            f8 = (
                torch.from_numpy(raw)
                .view(torch.float8_e4m3fn)
                .view(ids.size, self.embedding_dim)
            )
            scale = torch.tensor(self._table.weight_scale, dtype=torch.bfloat16)
            bf = f8.to(torch.bfloat16) * scale
            self._host[:tokens] = bf.view(tokens, -1, self.embedding_dim)
        else:
            self._table.read(
                ids,
                self._host[:tokens]
                .view(torch.uint8)
                .numpy()
                .reshape(-1, self._table.row_bytes),
            )
        self._copy_to_gpu(tokens)

    def _copy_to_gpu(self, tokens: int) -> None:
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            self._gpu[:tokens].copy_(self._host[:tokens], non_blocking=True)
            self._copy_ready.record(self._stream)

    @partial(eager_break_during_capture, always=True)
    def start_prefetch(self, hidden_states, ngram_ids) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE SSD requires eager or breakable CUDA graphs")
        if self._pending is not None:
            raise RuntimeError("Previous PLE SSD lookup has not been consumed")
        if len(self._loaded_starts) != len(self._table._parts):
            raise RuntimeError("PLE checkpoint shards have not all been loaded")
        tokens = ngram_ids.shape[0]
        if tokens > self._ids.shape[0]:
            raise ValueError("PLE SSD prefetch exceeds configured token capacity")
        if BreakableCUDAGraphCapture.current() is not None:
            # Captured ID-producing kernels have not executed yet. Actual
            # lookups start on replay, when those kernels have run.
            return
        # Host staging must outlive H2D; device staging must outlive its consumer.
        self._copy_ready.synchronize()
        self._stream.wait_event(self._previous_use)
        # Preserve IDs before the next graph segment reuses their pool slot.
        self._device_ids[:tokens].copy_(ngram_ids)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._ids[:tokens].copy_(self._device_ids[:tokens], non_blocking=True)
            self._ids_ready.record(self._stream)
        self._pending = self._coordinator.submit(self._read_and_copy, tokens)

    @partial(eager_break_during_capture, always=True)
    def _finalize_prefetch(self, output: torch.Tensor) -> None:
        if BreakableCUDAGraphCapture.current() is not None:
            output.zero_()
            return
        if self._pending is None:
            raise RuntimeError("PLE SSD lookup was not prefetched")
        self._pending.result()
        self._pending = None
        torch.cuda.current_stream().wait_event(self._copy_ready)
        output.copy_(self._gpu[: output.shape[0]].flatten(-2))
        self._previous_use.record()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (hidden_states.shape[0], self._gpu.shape[1] * self.embedding_dim),
            dtype=self.weight.dtype,
            device=hidden_states.device,
        )
        self._finalize_prefetch(output)
        return output

    def close(self) -> None:
        self._coordinator.shutdown(wait=True)
        try:
            if self.prompt_prefetcher is not None:
                self.prompt_prefetcher.close()
            self._stream.synchronize()
        finally:
            self._table.close()
