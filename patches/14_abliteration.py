#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runtime refusal abliteration

The direction vector is not shipped: it belongs to the Apache-2.0 release
orcarouter/Qwen3.8-Flash-Next-Uncensored (revision e096800036ec) by way of the
2x3090 overlay github.com/DominikBucko/qwen38-flash-next-2x3090 (Apache-2.0).
Pass its path in QWEN38_ABLITERATION (a JSON file with a refusal_direction list,
a safetensors file with a refusal_direction tensor, or a bare name resolved
against vllm/models/qwen4_exp/nvidia.BUILTIN_DIR).

With the switch off (unset or 0) nothing is loaded and no work is added, so the
patch is safe to keep installed.

usage: python 14_abliteration.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-14_abliteration next to
       every file it changes.
"""

import shutil
import sys
from pathlib import Path

TAG = "14_abliteration"

# one new module (path, whole file)
MODULE = (
    "vllm/models/qwen4_exp/nvidia/abliteration.py",
            '# SPDX-License-Identifier: Apache-2.0' + '\n'
        '"""Optional refusal-direction removal ("abliteration") applied at runtime.' + '\n'
        '' + '\n'
        'Ported from the 2x3090 Qwen3.8-Flash-Next overlay' + '\n'
        '(github.com/DominikBucko/qwen38-flash-next-2x3090, runtime/vllm-overlay,' + '\n'
        'Apache-2.0). The direction is recovered from the Apache-2.0 release' + '\n'
        'orcarouter/Qwen3.8-Flash-Next-Uncensored (Arditi et al., 2024).' + '\n'
        '' + '\n'
        '``QWEN38_ABLITERATION`` names a JSON file with a ``refusal_direction`` list, or' + '\n'
        'a safetensors file with a ``refusal_direction`` tensor, of length' + '\n'
        '``hidden_size``. A bare name such as ``orcarouter`` resolves to' + '\n'
        '``/srv/models/vllm/abliteration/<name>.json``. With it set, every' + '\n'
        'residual-stream write loses its component along the unit direction r:' + '\n'
        '' + '\n'
        '    x <- x - r (r . x)' + '\n'
        '' + '\n'
        'applied to the attention and linear-attention block outputs, the MoE block' + '\n'
        'output (routed plus shared experts), the PLE value projection and the text' + '\n'
        'token embeddings. For one direction this equals the weight edit' + '\n'
        "``W' = W - r (r^T W)`` on ``self_attn.o_proj``, ``linear_attn.out_proj``," + '\n'
        "every expert's and the shared expert's ``down_proj``, ``ple.value_proj`` and" + '\n'
        'the rows of ``embed_tokens``. All weights stay untouched. Unset or ``0``,' + '\n'
        'nothing is loaded and no work is added.' + '\n'
        '"""' + '\n'
        '' + '\n'
        'import json' + '\n'
        'import math' + '\n'
        'import os' + '\n'
        '' + '\n'
        'import torch' + '\n'
        '' + '\n'
        'ENV = "QWEN38_ABLITERATION"' + '\n'
        'BUILTIN_DIR = "/srv/models/vllm/abliteration"' + '\n'
        '' + '\n'
        '' + '\n'
        'def direction_path() -> str | None:' + '\n'
        '    """Resolve the switch into a file path, or None when it is off."""' + '\n'
        '    value = os.environ.get(ENV, "").strip()' + '\n'
        '    if value in ("", "0"):' + '\n'
        '        return None' + '\n'
        '    if os.sep not in value and not value.endswith((".json", ".safetensors")):' + '\n'
        '        value = os.path.join(BUILTIN_DIR, f"{value}.json")' + '\n'
        '    return value' + '\n'
        '' + '\n'
        '' + '\n'
        'def load_direction(hidden_size: int) -> torch.Tensor | None:' + '\n'
        '    """The unit refusal direction as a float32 CPU tensor, or None when off."""' + '\n'
        '    path = direction_path()' + '\n'
        '    if path is None:' + '\n'
        '        return None' + '\n'
        '    if path.endswith(".json"):' + '\n'
        '        with open(path) as f:' + '\n'
        '            r = torch.tensor(json.load(f)["refusal_direction"], dtype=torch.float32)' + '\n'
        '    else:' + '\n'
        '        from safetensors import safe_open' + '\n'
        '' + '\n'
        '        with safe_open(path, framework="pt", device="cpu") as f:' + '\n'
        '            r = f.get_tensor("refusal_direction")' + '\n'
        '    r = r.to(torch.float32).flatten()' + '\n'
        '    if r.numel() != hidden_size:' + '\n'
        '        raise ValueError(' + '\n'
        '            f"{ENV}: {path} holds {r.numel()} values; the model\'s hidden "' + '\n'
        '            f"size is {hidden_size}"' + '\n'
        '        )' + '\n'
        '    norm = float(r.norm())' + '\n'
        '    if not math.isfinite(norm) or norm == 0.0:' + '\n'
        '        raise ValueError(f"{ENV}: {path} holds a zero or non-finite direction")' + '\n'
        '    return r / norm' + '\n'
        '' + '\n'
        '' + '\n'
        'def _offload_process() -> bool:' + '\n'
        '    """True only in a CPU-only PLE-offload worker, if this tree has one.' + '\n'
        '' + '\n'
        "    Upstream's overlay calls ``ple_offload_layer.is_offload_process``; this" + '\n'
        '    build has no such module (the PLE SSD offload runs the layer on the GPU),' + '\n'
        '    so the probe is best-effort and defaults to False.' + '\n'
        '    """' + '\n'
        '    try:' + '\n'
        '        from vllm.model_executor.layers.ple_offload_layer import is_offload_process' + '\n'
        '    except Exception:' + '\n'
        '        return False' + '\n'
        '    try:' + '\n'
        '        return bool(is_offload_process())' + '\n'
        '    except Exception:' + '\n'
        '        return False' + '\n'
        '' + '\n'
        '' + '\n'
        'def register(module: torch.nn.Module, hidden_size: int, dtype: torch.dtype) -> None:' + '\n'
        '    """Attach the direction as a non-persistent buffer ``_abliteration_r``.' + '\n'
        '' + '\n'
        "    Call inside a module constructor so the buffer lands on the module's device" + '\n'
        '    before CUDA-graph capture. With the switch off the attribute is set to' + '\n'
        '    None, so ``project_`` stays a no-op and nothing is loaded.' + '\n'
        '    """' + '\n'
        '    r = None if _offload_process() else load_direction(hidden_size)' + '\n'
        '    if r is None:' + '\n'
        '        module._abliteration_r = None' + '\n'
        '        return' + '\n'
        '    buf = torch.empty(hidden_size, dtype=dtype)' + '\n'
        '    buf.copy_(r)' + '\n'
        '    module.register_buffer("_abliteration_r", buf, persistent=False)' + '\n'
        '' + '\n'
        '' + '\n'
        'def project_(x: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:' + '\n'
        '    """In place: drop the component along r from x\'s last dim. No-op if None."""' + '\n'
        '    if r is None:' + '\n'
        '        return x' + '\n'
        '    width = x.shape[-1]' + '\n'
        '    if x.is_contiguous():' + '\n'
        '        flat = x.view(-1, width)' + '\n'
        '        flat.addr_(flat @ r, r, alpha=-1.0)  # rank-1 update, no full-size temp' + '\n'
        '    else:' + '\n'
        '        x.sub_(torch.outer(x.reshape(-1, width) @ r, r).view(x.shape))' + '\n'
        '    return x' + '\n'
        '',
)

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""from ..config import ATTENTION_LAYER_TYPES, QSA_LAYER_TYPE, Qwen4ExpConfig
from .hyperconnection import GatedResidual, HyperConnectionConfig
""",
        # ---- after
"""from ..config import ATTENTION_LAYER_TYPES, QSA_LAYER_TYPE, Qwen4ExpConfig
from . import abliteration
from .hyperconnection import GatedResidual, HyperConnectionConfig
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
""",
        # ---- after
"""        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )
        # PATCH (abliteration): the direction buffer is placed ON EVERY layer,
        # so it exists exactly on the PP ranks where the layer really lives
        # (see docs/abliteration.md, the PP section).
        abliteration.register(self, config.hidden_size, model_config.dtype)

    def forward(
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
""",
        # ---- after
"""        else:
            raise ValueError("Invalid layer_type")
        # PATCH (abliteration): o_proj / out_proj write straight into the residual.
        abliteration.project_(attn_out, self._abliteration_r)

        mlp_hc = self.mlp_hyper_connection
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
""",
        # ---- after
"""        mlp_out = self.mlp(block_input)
        # PATCH (abliteration): the routed+shared expert sum before it is written
        # residual - the projection of the sum equals patching every down_proj.
        abliteration.project_(mlp_out, self._abliteration_r)
        return hidden_states, mlp_out, injection
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
""",
        # ---- after
"""        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        # PATCH (abliteration): buffer on the model itself - for text embeddings.
        abliteration.register(self, config.hidden_size, vllm_config.model_config.dtype)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        # ---- before
"""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
""",
        # ---- after
"""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PATCH (abliteration): embed_tokens rows write into the residual. The projection
        # runs BEFORE the multimodal merge (images are left untouched)
        # and BEFORE the repeat under hc_count - that is, along hidden_size.
        return abliteration.project_(
            self.embed_tokens(input_ids), self._abliteration_r
        )
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/mtp.py',
        # ---- before
"""from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
""",
        # ---- after
"""from . import abliteration
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/mtp.py',
        # ---- before
"""        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
""",
        # ---- after
"""        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        # PATCH (abliteration): the draft embedding branch. Draft layers are the same
        # Qwen4ExpDecoderLayer, so their residual writers are already covered.
        abliteration.register(self, self.hidden_size, model_config.dtype)
        draft_vllm_config = _make_draft_vllm_config(
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/mtp.py',
        # ---- before
"""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
""",
        # ---- after
"""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PATCH (abliteration): the same projection for the draft - otherwise spec-decode
        # would diverge from the target (see docs/abliteration.md, pt.5).
        return abliteration.project_(
            self.embed_tokens(input_ids), self._abliteration_r
        )
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_layer.py',
        # ---- before
"""from .ngram_embedding import Qwen4ExpNGramEmbedding
from .ops.ple import ple_conv, ple_gate
""",
        # ---- after
"""from . import abliteration
from .ngram_embedding import Qwen4ExpNGramEmbedding
from .ops.ple import ple_conv, ple_gate
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_layer.py',
        # ---- before
"""            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
""",
        # ---- after
"""            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        # PATCH (abliteration): buffer on the PLE layer itself. The value projection goes
        # forward BEFORE the gate/norm: the norm is non-linear, so patching the weight
        # value_proj acts exactly there (docs/abliteration.md, pt.3).
        abliteration.register(self, self.hidden_size, model_config.dtype)
        norm_args = (
""",
    ),
    (
        'vllm/models/qwen4_exp/nvidia/ple_layer.py',
        # ---- before
"""        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
""",
        # ---- after
"""        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        # PATCH (abliteration): part of value writes into the residual through the gate.
        # value is a non-contiguous slice, project_ accounts for that.
        abliteration.project_(value, self._abliteration_r)
        gated_output, conv_input = ple_gate(
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

    rel, content = MODULE
    path = root / rel
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if current == content:
        print(f"  already  {rel}")
    elif cmd == "check":
        print(f"  would    {rel}")
    elif cmd == "install":
        if current is not None:
            shutil.copy2(path, path.with_name(path.name + ".bak-" + TAG))
        path.write_text(content, encoding="utf-8", newline="")
        print(f"  written  {rel}")
    elif current is not None:
        bak = path.with_name(path.name + ".bak-" + TAG)
        if bak.is_file():
            shutil.copy2(bak, path)
            print(f"  restored {rel} <- {bak.name}")

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
    print(f"{TAG}: module + {len(EDITS)} edit(s) in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
