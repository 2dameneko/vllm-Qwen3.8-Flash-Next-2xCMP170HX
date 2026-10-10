# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.ple_prompt_prefetchers = []
        prefetch_tokens = int(
            vllm_config.additional_config.get("ple_ssd_prefetch_tokens", 0)
        )
        if prefetch_tokens:
            from .ngram_embedding import Qwen4ExpNGramEmbedding
            from .ple_ssd import PLEPromptPrefetcher, Qwen4ExpPLESSDEmbedding

            for module in model.modules():
                if isinstance(module, Qwen4ExpNGramEmbedding) and isinstance(
                    module.ngram_embedding, Qwen4ExpPLESSDEmbedding
                ):
                    prefetcher = PLEPromptPrefetcher(module, prefetch_tokens)
                    module.ngram_embedding.prompt_prefetcher = prefetcher
                    self.ple_prompt_prefetchers.append(prefetcher)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        # PATCH (vllm#54709): the check is on PLACEMENT, not on PP>1 itself.
        # Here pp_group is already initialized, so the rank is known.
        from vllm.distributed import get_pp_group
        from vllm.distributed.utils import get_pp_indices

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            _pp_size = vllm_config.parallel_config.pipeline_parallel_size
            _pp_rank = get_pp_group().rank_in_group
            _num_layers = int(config.num_hidden_layers)
            _start, _end = get_pp_indices(_num_layers, _pp_rank, _pp_size)
            _ple_idx = sorted(int(i) for i in config.ple_layer_ids)
            _here = [i for i in _ple_idx if _start <= i < _end]
            if _pp_rank == 0 and len(_here) != len(_ple_idx):
                raise RuntimeError(
                    f"Qwen4Exp PLE layers {sorted(set(_ple_idx) - set(_here))} "
                    f"fall outside pipeline rank 0 (layers {_start}..{_end - 1}). "
                    f"Give rank 0 more layers via VLLM_PP_LAYER_PARTITION, "
                    f"or run with PP=1."
                )
            if _pp_rank != 0 and _here:
                raise RuntimeError(
                    f"Qwen4Exp PLE layers {_here} landed on pipeline rank "
                    f"{_pp_rank} (layers {_start}..{_end - 1}), but PLE must "
                    f"live on rank 0. Adjust VLLM_PP_LAYER_PARTITION."
                )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if new_req_data.prefill_token_ids is not None:
            for prefetcher in self.ple_prompt_prefetchers:
                prefetcher.submit(
                    new_req_data.req_id,
                    new_req_data.prefill_token_ids,
                    new_req_data.num_computed_tokens,
                )

    def remove_request(self, req_id: str) -> None:
        for prefetcher in self.ple_prompt_prefetchers:
            prefetcher.cancel(req_id)
        super().remove_request(req_id)

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
