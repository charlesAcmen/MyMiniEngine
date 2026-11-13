"""Manual CUDA graph replay for fixed-size paged decode buckets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from miniengine.paged_attention import PagedDecode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miniengine.model import CausalLM

logger = logging.getLogger(__name__)


def select_graph_bucket(batch_size: int, buckets: Sequence[int]) -> int:
    """Return the smallest captured bucket that can hold ``batch_size`` rows."""
    for bucket in buckets:
        if batch_size <= bucket:
            return bucket
    raise ValueError(
        f"decode batch size {batch_size} exceeds largest CUDA graph bucket "
        f"{buckets[-1]}"
    )


@dataclass
class _CapturedDecode:
    #token id
    input_ids: torch.Tensor
    #position for each token
    position_ids: torch.Tensor
    #kv len for each sequence in the batch
    cache_seqlens: torch.Tensor
    #logic->physical block mapping for each sequence in the batch
    block_table: torch.Tensor
    #padding
    dummy_positions: torch.Tensor
    dummy_cache_seqlens: torch.Tensor
    #captured CUDA graph
    graph: torch.cuda.CUDAGraph
    #sampling is excluded from the graph, so we need to return the logits for each sequence in the batch
    logits: torch.Tensor


class CudaGraphRunner:
    """Capture and replay paged model decode while sampling stays eager."""

    def __init__(
        self,
        model: CausalLM,
        kv_caches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        batch_sizes: tuple[int, ...],
        page_size: int,
        max_model_len: int,
        device: str,
        warmup_steps: int = 3,
    ) -> None:
        #max bucket = 32,page size = 256
        if max(batch_sizes) > page_size:
            raise ValueError(
                "largest CUDA graph bucket must fit in the reserved scratch page: "
                f"bucket={max(batch_sizes)}, page_size={page_size}"
            )
        self.model = model
        #pre allocated,address stable
        self.kv_caches = kv_caches
        self.batch_sizes = batch_sizes
        self.max_model_len = max_model_len
        self.max_blocks = (max_model_len + page_size - 1) // page_size
        self.device = torch.device(device)
        self._captured: dict[int, _CapturedDecode] = {}

        # Graph replay must never trigger a RoPE cache replacement because the
        # captured kernels retain the original cos/sin storage addresses.
        position = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.model.model.rotary_emb(position, max_model_len)
        for batch_size in batch_sizes:
            self._captured[batch_size] = self._capture(batch_size, warmup_steps)

    def _capture(self, batch_size: int, warmup_steps: int) -> _CapturedDecode:
        #fixed address
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            batch_size, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        cache_seqlens = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )
        block_table = torch.zeros(
            (batch_size, self.max_blocks), dtype=torch.int32, device=self.device
        )
        dummy_positions = torch.arange(
            batch_size, dtype=torch.long, device=self.device
        )
        dummy_cache_seqlens = dummy_positions.to(torch.int32)

        def forward() -> torch.Tensor:
            #no sampling, only complete forward
            return self.model.forward_paged_decode(
                input_ids,
                position_ids,
                self.kv_caches,
                cache_seqlens,
                block_table,
                self.max_model_len,
            )

        with torch.inference_mode():
            #warm up,lazy load etc
            #torch.compile
            #Triton kernel generation
            #CUDA module load
            #lazy workspace allocation
            #allocator initialization
            #kernel autotuning
            for _ in range(warmup_steps):
                logits = forward()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = forward()
        logger.info("Captured paged decode CUDA graph for batch size %d", batch_size)
        return _CapturedDecode(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            dummy_positions=dummy_positions,
            dummy_cache_seqlens=dummy_cache_seqlens,
            graph=graph,
            logits=logits,
        )

    def __call__(self, decode: PagedDecode) -> torch.Tensor:
        actual_batch = decode.input_ids.shape[0]
        bucket = select_graph_bucket(actual_batch, self.batch_sizes)
        captured = self._captured[bucket]

        #reuse address
        captured.input_ids[:actual_batch].copy_(decode.input_ids)
        captured.position_ids[:actual_batch].copy_(decode.position_ids)
        captured.cache_seqlens[:actual_batch].copy_(decode.cache_seqlens)
        block_columns = decode.block_table.shape[1]
        if block_columns > self.max_blocks:
            raise ValueError(
                f"decode block table has {block_columns} columns, but the model "
                f"context permits only {self.max_blocks}"
            )
        captured.block_table[:actual_batch, :block_columns].copy_(decode.block_table)

        if actual_batch < bucket:
            padding = bucket - actual_batch
            dummy = captured.dummy_positions[:padding]
            captured.input_ids[actual_batch:].zero_()
            captured.position_ids[actual_batch:, 0].copy_(dummy)
            captured.cache_seqlens[actual_batch:].copy_(
                captured.dummy_cache_seqlens[:padding]
            )
            # Page zero is reserved as the scratch sink. Each dummy row uses a
            # distinct cache offset, so its in-place K/V append cannot race.
            captured.block_table[actual_batch:].zero_()

        #CPU commits instanized graph to GPU, GPU executes graph
        captured.graph.replay()
        return captured.logits[:actual_batch]#
