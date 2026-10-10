# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filter out non-local expert weights during loading to avoid redundant I/O.

In DP+EP deployments each rank only needs its own expert shard.  Skipping
non-local expert tensors *before* they are read from disk eliminates the
majority of storage I/O for MoE models (experts typically account for
~85-90 % of total weight bytes).
"""

import regex as re
from typing import NamedTuple

# Matches per-expert weight names like ".experts.42.gate_proj.weight".
# Does NOT match 3D fused-expert names like ".experts.gate_proj.weight"
# (no numeric id) — those are intentionally left unfiltered so the full
# tensor is loaded and sliced later by RoutedExperts.weight_loader.
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")

# Matches the transformer-layer index in names such as
# ".layers.17.mlp.experts.3.gate_proj.weight". Attention, MoE, norm
# and PLE sub-modules all carry it, which is what makes pipeline
# filtering possible (see compute_local_layer_ids).
_LAYER_ID_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_expert_id(weight_name: str) -> int | None:
    """Return the expert id embedded in *weight_name*, or ``None`` if it is
    not an per-expert weight.

    Returns ``None`` for dense weights (attention, layernorm, embedding),
    shared experts, and 3D fused-expert tensors where all experts are stored
    in a single tensor without a numeric expert id in the name."""
    m = _EXPERT_ID_RE.search(weight_name)
    return int(m.group(1)) if m else None


def compute_local_expert_ids(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    placement: str = "linear",
) -> set[int] | None:
    """Compute the set of global expert ids owned by *ep_rank*.

    Returns ``None`` when EP is not active (``ep_size <= 1``), meaning all
    experts are local and no filtering should be performed.

    The distribution logic mirrors
    :func:`vllm.model_executor.layers.fused_moe.layer.determine_expert_map`.

    Args:
        num_experts: Total number of experts in the layer.
        ep_size: Expert parallel world size.
        ep_rank: Rank whose local expert ids are computed.
        placement: ``"linear"`` for contiguous assignment,
            ``"round_robin"`` for interleaved assignment.

    """
    if ep_size <= 1:
        return None

    if placement == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        start = ep_rank * base + min(ep_rank, remainder)
        local_count = base + (1 if ep_rank < remainder else 0)
        return set(range(start, start + local_count))
    elif placement == "round_robin":
        return set(range(ep_rank, num_experts, ep_size))
    else:
        raise ValueError(f"Unknown expert placement strategy: {placement}")


class LocalLayerIds(NamedTuple):
    """Hidden layers owned by one pipeline stage, plus MTP ownership."""

    layers: range
    owns_mtp: bool


def compute_local_layer_ids(
    num_hidden_layers: int,
    pp_size: int,
    pp_rank: int,
) -> LocalLayerIds | None:
    """Return the hidden layers owned by *pp_rank* (``None`` when PP==1).

    Every pipeline stage builds only its own slice of the transformer,
    yet the loader walks the whole checkpoint and the model silently
    drops foreign layers: on a 76 GiB checkpoint with PP=2 each stage
    reads about twice what it needs, and both stages compete for the
    same NVMe.  Knowing the slice upfront lets the iterator skip those
    tensors *before* they are read (see should_skip_weight).

    The range comes from vLLM's own :func:`get_pp_indices`, i.e. it
    honours VLLM_PP_LAYER_PARTITION and cannot drift from the actual
    placement - models/qwen4_exp/nvidia/model_state.py validates PLE
    the same way.  ``mtp.*`` tensors are NOT part of that range: the
    multi-token-prediction draft sits on the last stage, which is
    reported separately as :attr:`LocalLayerIds.owns_mtp`.
    """
    if pp_size <= 1:
        return None
    from vllm.distributed.utils import get_pp_indices

    start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    return LocalLayerIds(range(start, end), pp_rank == pp_size - 1)


def should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
    local_layers: LocalLayerIds | None = None,
) -> bool:
    """Return ``True`` if *weight_name* does not belong to the local rank
    and should be skipped *before* it is read from disk.

    Two independent filters apply, both of them pure I/O savings:
    pipeline parallelism (layers owned by another stage) and expert
    parallelism (experts owned by another rank).  A skipped tensor is
    by construction one the local model would never consume.
    """
    if local_layers is not None:
        if weight_name.startswith("mtp."):
            if not local_layers.owns_mtp:
                return True
        else:
            m = _LAYER_ID_RE.search(weight_name)
            if m is not None and int(m.group(1)) not in local_layers.layers:
                return True
    if local_expert_ids is None:
        return False
    eid = parse_expert_id(weight_name)
    if eid is None:
        # Not an expert weight (dense / shared-expert / embedding) → keep.
        return False
    # Only skip heavy weight tensors, never scale/metadata tensors.
    # Scale tensors are tiny and some backends need them from ALL experts
    # (e.g. FlashInfer NVFP4 computes a global max of activation scales).
    if not weight_name.endswith((".weight", ".weight_packed")):
        return False
    return eid not in local_expert_ids
