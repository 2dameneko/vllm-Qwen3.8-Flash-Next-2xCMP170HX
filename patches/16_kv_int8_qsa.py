#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""INT8 (per-token, per-head) KV cache for the QSA attention path

SM80 has no `fp8e4nv` in Triton, so an FP8 KV cache in this tree needs a
bit-hack and costs about 36 % of decode. INT8 has no such barrier: int8 -> bf16
is a native widening on Ampere, and per-(token, head) scales give better quality
than one fp8 scale per tensor, for 16 bytes per token per layer (two fp32 scales
for two KV heads, about +1.6 % of the pool).

Everything the storage side needs is already upstream in this tree -
`triton_reshape_and_cache_flash_per_token_head_quant()` in
`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`, the
`INT8_PER_TOKEN_HEAD` quant mode in `vllm/v1/kv_cache_interface.py`, and the
worker that clears `layer.impl._k_scale_cache` on teardown. What is missing is
the QSA wiring, which this patch adds:

  * `nvidia/qsa.py`: the backend accepts the two per-token-head dtypes, the impl
    keeps its own `_k_scale_cache`/`_v_scale_cache` ([blocks, block_size, heads]
    fp32, outside the KV page, so the hybrid block-size arithmetic is unchanged),
    `do_kv_cache_update()` routes the store through the core quantizing kernel,
    and the attention path hands the scale tensors to the kernel;
  * `nvidia/ops/qsa.py`: a `QUANT` branch in the attention kernel (widening loads
    plus per-token K/V scales), scale pointers and strides in the signature, and
    a scale stub in the warmup.

Enable it with `--kv-cache-dtype int8_per_token_head` (`KV_DTYPE` in
`serve/awq-pp2.env`). Without that flag the new branch is never taken, so the
patch is safe to keep installed.

Measured on 2x CMP 170HX (see ../BENCHMARKS.md): KV pool 744 431 -> 1 352 663
tokens at ctx 262 144 (5.16x the full context), decode 125 -> 123 tok/s, prefill
within noise, needle-in-a-haystack 3/3 at 32k.

usage: python 16_kv_int8_qsa.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-16_kv_int8_qsa next to every
       file it changes.
"""

import shutil
import sys
from pathlib import Path

TAG = "16_kv_int8_qsa"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 1)
        '''    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
''',
        # ---- after
        '''    softmax_scale,
    output_scale,
    output_gate_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_q_row,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 2)
        '''    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
''',
        # ---- after
        '''    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_indices_row,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 3)
        '''    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
''',
        # ---- after
        '''    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
    QUANT: tl.constexpr,
) -> None:
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 4)
        '''        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if IS_FP8:
            # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
            # softmax) so fp16 has the range, its wider mantissa is more
            # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
            values = values.to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
''',
        # ---- after
        '''        if QUANT:
            # PATCH #16 (kv_int8_qsa): per-(token, head) int8/fp8 cache. Exact
            # int8/fp8 -> BF16 widening (native on sm80), K scale applied to the
            # scores and V scale to the probabilities (both per selected token).
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0,
            ).to(query.dtype)
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0,
            ).to(query.dtype)
            scale_offsets = (
                safe_page * stride_s_block
                + page_offset * stride_s_token
                + kv_head * stride_s_head
            )
            key_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0)
            value_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if IS_FP8:
                # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
                # measured slower here and less accurate).
                keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        if QUANT:
            scores *= key_scale[None, :]
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if QUANT:
            weighted = (probabilities * value_scale[None, :]).to(values.dtype)
        else:
            if IS_FP8:
                # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
                # softmax) so fp16 has the range, its wider mantissa is more
                # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
                values = values.to(tl.float16)
            weighted = probabilities.to(values.dtype)
        accumulator = tl.dot(
            weighted,
            values,
            acc=accumulator * alpha[:, None],
        )
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 5)
        '''    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
''',
        # ---- after
        '''    assert k_cache.dtype == v_cache.dtype
    # PATCH #16 (kv_int8_qsa): k_scale/v_scale are the per-(token, head) float32
    # scale caches for the int8/fp8-per-token-head mode, or host floats/None for
    # the legacy per-tensor path.
    quant = isinstance(k_scale, torch.Tensor)
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if quant:
        assert isinstance(v_scale, torch.Tensor)
        assert k_cache.dtype in (torch.int8, torch.float8_e4m3fn)
        assert (
            k_scale.shape == v_scale.shape == k_cache.shape[:3]
            and k_scale.dtype == torch.float32
        )
        # The per-token K scale is applied in-kernel; keep unit host scales.
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    elif is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 6)
        '''        output_gate_view,
        q.stride(0),
''',
        # ---- after
        '''        output_gate_view,
        k_scale if quant else out,
        v_scale if quant else out,
        q.stride(0),
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 7)
        '''        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
''',
        # ---- after
        '''        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_scale.stride(0) if quant else 0,
        k_scale.stride(1) if quant else 0,
        k_scale.stride(2) if quant else 0,
        logical_indices.stride(0),
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 8)
        '''        IS_FP8=is_fp8,
        num_warps=partial_warps,
''',
        # ---- after
        '''        IS_FP8=is_fp8,
        QUANT=quant,
        num_warps=partial_warps,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 9)
        '''    # An fp8 cache is allocated as uint8 and viewed as e4m3 at attention time.
    is_fp8 = kv_cache.dtype == torch.uint8
    cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
''',
        # ---- after
        '''    # An fp8 cache is allocated as uint8 and viewed as e4m3 at attention time;
    # a per-token-head int8 cache is stored as int8 (PATCH #16).
    is_fp8 = kv_cache.dtype == torch.uint8
    quant = kv_cache.dtype == torch.int8
    if quant:
        cache_dtype = torch.int8
    else:
        cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 10)
        '''    v_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    # +1: the packed buffer's trailing count column.
''',
        # ---- after
        '''    v_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    # PATCH #16: per-(token, head) float32 scale cache for the quantized mode.
    scale_num_blocks = key_cache.shape[0]
    scale_block = key_cache.shape[1]
    scale_heads = key_cache.shape[2]
    scale_ptr = TritonWarmupTensor(
        torch.float32,
        shape=(scale_num_blocks, scale_block, scale_heads),
        strides=(scale_block * scale_heads, scale_heads, 1),
    )
    # +1: the packed buffer's trailing count column.
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 11)
        '''            output_gate_ptr,
            row_stride,
            head_stride,
''',
        # ---- after
        '''            output_gate_ptr,
            scale_ptr,
            scale_ptr,
            row_stride,
            head_stride,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 12)
        '''            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            selection_width + 1,
''',
        # ---- after
        '''            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            scale_block * scale_heads,
            scale_heads,
            1,
            selection_width + 1,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ops/qsa.py',
        # ---- before (qsa.py edit 13)
        '''            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            num_warps=warps,
''',
        # ---- after
        '''            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            QUANT=quant,
            num_warps=warps,
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 14)
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
''',
        # ---- after
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 15)
        '''        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            kv_cache_dtype = "auto"
''',
        # ---- after
        '''        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in (
            "fp8",
            "fp8_e4m3",
            "int8_per_token_head",
            "fp8_per_token_head",
        ):
            kv_cache_dtype = "auto"
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 16)
        '''        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        self.supports_quant_query_input = False
''',
        # ---- after
        '''        if self.kv_cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8_per_token_head",
            "fp8_per_token_head",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or per-token-head "
                "int8/fp8 main KV cache"
            )
        self.supports_quant_query_input = False
        # PATCH #16 (kv_int8_qsa): per-(token, head) quantized cache.
        self._pth = self.kv_cache_dtype in (
            "int8_per_token_head",
            "fp8_per_token_head",
        )
        self._pth_key = None
        self._k_scale_cache = None
        self._v_scale_cache = None
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 17)
        '''        self._v_scale_cache = None

    def forward_qsa(
''',
        # ---- after
        '''        self._v_scale_cache = None

    def _pth_caches(self, kv_cache: torch.Tensor):
        """Quantized K/V views [blocks, heads, block_size, head_size] plus their
        per-(token, head) float32 scale caches [blocks, block_size, heads].

        The scales live in separate tensors (8 bits per token per layer), so the
        KV page stays exactly the quantized data and the hybrid block-size math
        is unchanged (the core also nulls layer.impl._k_scale_cache on teardown).
        """
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if self.kv_cache_dtype == "fp8_per_token_head":
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        ident = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        if self._pth_key != ident:
            num_blocks, num_heads, block_size = (
                kv_cache.shape[0],
                kv_cache.shape[1],
                kv_cache.shape[2],
            )
            self._k_scale_cache = torch.ones(
                num_blocks,
                block_size,
                num_heads,
                dtype=torch.float32,
                device=kv_cache.device,
            )
            self._v_scale_cache = torch.ones_like(self._k_scale_cache)
            self._pth_key = ident
        return key_cache, value_cache, self._k_scale_cache, self._v_scale_cache

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # PATCH #16 (kv_int8_qsa): store per-(token, head) quantized K/V and its
        # scales through the core kernel; the QSA kernel reads them back.
        if not self._pth:
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_per_token_head_quant,
        )

        key_cache, value_cache, k_scale, v_scale = self._pth_caches(kv_cache)
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            slot_mapping,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def forward_qsa(
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 18)
        '''        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
            )
''',
        # ---- after
        '''        k_scale = v_scale = None
        if self._pth:
            key_cache, value_cache, k_scale, v_scale = self._pth_caches(kv_cache)
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
                # The cache is allocated as uint8; reinterpret the e4m3 bytes
                # (same itemsize, so shape and strides are preserved).
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)
                # Host-side per-tensor dequant scales (Python floats), as used by
                # other host-scale backends; folded into the kernel's scales.
                k_scale = layer._k_scale_float
                v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or (
            not self._pth
            and key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16, FP8-e4m3 or quantized K/V"
            )
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 19)
        '''        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
''',
        # ---- after
        '''        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8_per_token_head",
            "fp8_per_token_head",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or per-token-head "
                "int8/fp8 main KV cache"
            )
''',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/qsa.py',
        # ---- before (qsa.py edit 20)
        '''        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or FP8-e4m3 (uint8) cache storage"
            )
''',
        # ---- after
        '''        if self.kv_cache_torch_dtype not in (
            torch.bfloat16,
            torch.uint8,
            torch.int8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-e4m3 (uint8) or int8 cache storage"
            )
''',
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

    bad = []
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
            print(f"  ANCHOR?  {rel} - the before text is not there")
            bad.append(rel)
            continue
        if cmd == "check":
            print(f"  would    {rel}")
            continue
        # One backup per file, not per edit: twenty edits go into two files, and a
        # backup taken after the fifth edit is not the state `remove` must return.
        bak = path.with_name(path.name + ".bak-" + TAG)
        if not bak.is_file():
            shutil.copy2(path, bak)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
        print(f"  patched  {rel}")

    if cmd == "remove":
        # One restore per file, from the backup of the first edit.
        done = set()
        for rel, _old, _new in EDITS:
            if rel in done:
                continue
            done.add(rel)
            path = root / rel
            bak = path.with_name(path.name + ".bak-" + TAG)
            if bak.is_file():
                shutil.copy2(bak, path)
                print(f"  restored {rel} <- {bak.name}")
            else:
                print(f"  no backup for {rel} - was this ever installed here?")
                bad.append(rel)
        return 1 if bad else 0

    if bad and cmd == "install":
        print(f"WARNING: {len(bad)} edit(s) did not apply", file=sys.stderr)
        return 1
    if cmd == "check":
        print("check only, nothing written" if not bad
              else f"check: {len(bad)} edit(s) do not match this patch")
        return 1 if bad else 0
    print(f"{TAG}: {len(EDITS)} edit(s) in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
