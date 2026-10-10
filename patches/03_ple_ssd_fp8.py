#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FP8 PLE table (reserve path)

Lets PLESSDTable hold an F8_E4M3 table: item size from the shard dtype, the
per-tensor weight scale read from the checkpoint, dequantization to BF16 on
the host. Dormant unless an FP8-PLE checkpoint is served.

usage: python 03_ple_ssd_fp8.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-03_ple_ssd_fp8 next to every change.
"""

import shutil
import sys
from pathlib import Path

TAG = "03_ple_ssd_fp8"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                "PLE SSD workers must be positive and cache_mb nonnegative"
            )
        self.row_bytes = dim * 2
        self.rows = rows""",
        # ---- after
"""                "PLE SSD workers must be positive and cache_mb nonnegative"
            )
        # PATCH: the dtype comes from the shard headers (done below),
        # itemsize: BF16 -> 2, F8_E4M3 -> 1.
        self.dtype: str | None = None
        self.is_fp8 = False
        self.weight_scale = 1.0
        self.row_bytes = dim * 2
        self.rows = rows""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                info = header[key]
                shape = info["shape"]
                if info["dtype"] != "BF16" or len(shape) != 2 or shape[1] != dim:
                    raise ValueError(f"PLE SSD requires BF16 [rows, {dim}]: {key}")
                begin, end = info["data_offsets"]
                fd = self._fds[filename]""",
        # ---- after
"""                info = header[key]
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
                fd = self._fds[filename]""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""            if start != rows:
                raise ValueError(f"PLE table has {start} rows, expected {rows}")
        except BaseException:
            for fd in self._fds.values():""",
        # ---- after
"""            if start != rows:
                raise ValueError(f"PLE table has {start} rows, expected {rows}")
            if self.is_fp8:
                self.weight_scale = self._read_weight_scale(
                    weight_map, prefix, headers, root
                )
        except BaseException:
            for fd in self._fds.values():""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""                os.close(fd)
            raise

    def _read_rows(self, ids: list[int]) -> list[tuple[int, bytes]]:""",
        # ---- after
        '                os.close(fd)' + '\n'
        '            raise' + '\n'
        '' + '\n'
        '    @staticmethod' + '\n'
        '    def _bf16_bytes_to_float(data: bytes) -> float:' + '\n'
        '        """Little-endian BF16 scalar to Python float (no numpy/torch)."""' + '\n'
        '        u16 = int.from_bytes(data, "little")' + '\n'
        '        sign = -1.0 if (u16 >> 15) & 1 else 1.0' + '\n'
        '        exp = (u16 >> 7) & 0xFF' + '\n'
        '        man = u16 & 0x7F' + '\n'
        '        if exp == 0:' + '\n'
        '            return sign * (man / 128.0) * (2.0 ** -126)' + '\n'
        '        if exp == 0xFF:' + '\n'
        '            return sign * float("inf") if man == 0 else float("nan")' + '\n'
        '        return sign * (1.0 + man / 128.0) * (2.0 ** (exp - 127))' + '\n'
        '' + '\n'
        '    def _read_weight_scale(self, weight_map, prefix, headers, root) -> float:' + '\n'
        '        """Read the global per-tensor FP8 weight_scale from the checkpoint."""' + '\n'
        '        scale_suffix = prefix[prefix.index("layers."):] + ".weight_scale"' + '\n'
        '        keys = [k for k in weight_map if k.endswith(scale_suffix)]' + '\n'
        '        if len(keys) != 1:' + '\n'
        '            raise ValueError(' + '\n'
        '                f"FP8 PLE checkpoint must have exactly one {scale_suffix}"' + '\n'
        '            )' + '\n'
        '        key = keys[0]' + '\n'
        '        filename = weight_map[key]' + '\n'
        '        if filename not in headers:' + '\n'
        '            path = root / filename' + '\n'
        '            with open(path, "rb") as f:' + '\n'
        '                header_len = struct.unpack("<Q", f.read(8))[0]' + '\n'
        '                if header_len > 100_000_000:' + '\n'
        '                    raise ValueError(f"Invalid safetensors header: {path}")' + '\n'
        '                headers[filename] = (' + '\n'
        '                    json.loads(f.read(header_len)),' + '\n'
        '                    8 + header_len,' + '\n'
        '                )' + '\n'
        '            fd = os.open(path, os.O_RDONLY)' + '\n'
        '            self._fds[filename] = fd' + '\n'
        '            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)' + '\n'
        '        header, data_start = headers[filename]' + '\n'
        '        info = header[key]' + '\n'
        '        begin, end = info["data_offsets"]' + '\n'
        '        if info["dtype"] != "BF16" or end - begin != 2:' + '\n'
        '            raise ValueError(f"PLE weight_scale must be a BF16 scalar: {key}")' + '\n'
        '        fd = self._fds[filename]' + '\n'
        '        data = os.pread(fd, end - begin, data_start + begin)' + '\n'
        '        if len(data) != 2:' + '\n'
        '            raise OSError(f"Short PLE weight_scale read: {key}")' + '\n'
        '        return self._bf16_bytes_to_float(data)' + '\n'
        '' + '\n'
        '    def _read_rows(self, ids: list[int]) -> list[tuple[int, bytes]]:',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""
    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        if checkpoint_start is None or loaded_weight.dtype != torch.bfloat16:
            raise ValueError("PLE SSD requires checkpoint-split BF16 weights")
        self._loaded_starts.add(checkpoint_start)
""",
        # ---- after
"""
    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        if checkpoint_start is None or loaded_weight.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise ValueError("PLE SSD requires checkpoint-split BF16/FP8 weights")
        self._loaded_starts.add(checkpoint_start)
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_ssd.py',
        # ---- before
"""    def _read_and_copy(self, tokens: int) -> None:
        self._ids_ready.synchronize()
        self._table.read(
            self._ids[:tokens].numpy(),
            self._host[:tokens]
            .view(torch.uint8)
            .numpy()
            .reshape(-1, self._table.row_bytes),
        )
        self._copy_to_gpu(tokens)
""",
        # ---- after
"""    def _read_and_copy(self, tokens: int) -> None:
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
""",
    ),
)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args and args[0] in ("install", "check", "remove") else "check"
    rest = args[1:] if args and args[0] in ("install", "check", "remove") else args
    root = Path(rest[0]).resolve() if rest else Path(__file__).resolve().parent.parent
    if not (root / "vllm").is_dir():
        print(f"not a vLLM tree: {root} (pass its root as an argument)", file=sys.stderr)
        return 2

    bad, soft = [], []
    for rel, old, new in EDITS:
        path = root / rel
        if not path.is_file():
            print(f"  MISSING  {rel}")
            bad.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"  already  {rel}")
            continue
        if old not in text:
            # A later patch in the chain may have rewritten this very region, so
            # neither the before nor the after text is here any more.
            print(f"  neither  {rel} - before and after text are both absent")
            soft.append(rel)
            continue
        if cmd == "check":
            print(f"  would    {rel}")
            continue
        if cmd == "remove":
            print(f"  skip     {rel} (nothing installed here)")
            continue
        bak = path.with_name(path.name + ".bak-" + TAG)
        shutil.copy2(path, bak)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
        print(f"  patched  {rel}  (backup {bak.name})")

    if cmd == "remove":
        for rel, _old, _new in EDITS:
            path = root / rel
            bak = path.with_name(path.name + ".bak-" + TAG)
            if bak.is_file():
                shutil.copy2(bak, path)
                print(f"  restored {rel} <- {bak.name}")
    if soft:
        print(f"note: {len(soft)} region(s) were rewritten by a later patch of the set;"
              f" check the end state with ../verify.sh")
    if bad and cmd == "install":
        print(f"WARNING: {len(bad)} file(s) were not patched", file=sys.stderr)
        return 1
    if cmd == "check":
        print("check only, nothing written" if not bad
              else f"check: {len(bad)} file(s) do not match this patch")
        return 1 if bad else 0
    print(f"{TAG}: {len(EDITS)} edit(s) in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
