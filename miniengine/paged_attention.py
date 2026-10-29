"""Paged-KV request metadata and FlashAttention runtime boundary.

This module owns no model state. It translates request-local logical token
positions into physical slots in ``KVMemoryPool``. The later model attention
path consumes its packed metadata and is the only code allowed to write K/V
vectors into those slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool

if TYPE_CHECKING:
    from collections.abc import Sequence

#frozen=True:__init__：__repr__：__eq__： is automatically generated,
# and the instance is immutable after creation.
# but not indepth readonly,packed.input_ids[0] = 999 might be possible
@dataclass(frozen=True)
class PackedPrefill:
    """One ragged prompt batch, packed without padding.

    ``input_ids`` and ``position_ids`` have ``sum(prompt_lengths)`` entries.
    ``cu_seqlens`` records each request's half-open (半开区间[start,end)) segment 
    in that packed token axis. ``slot_mapping`` gives the physical flattened 
    KV-pool slot to which every newly computed K/V vector must be written.
    """
    # input_ids=[
    #     10, 11, 12,
    #     20, 21, 22, 23, 24,
    # ],

    # position_ids=[
    #     0, 1, 2,
    #     0, 1, 2, 3, 4,
    # ],

    # cu_seqlens=[0, 3, 8],

    # slot_mapping=[
    #     4, 5, 6,
    #     8, 9, 10, 11, 12,
    # ],

    # max_seqlen=5,
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    slot_mapping: torch.Tensor
    max_seqlen: int


@dataclass(frozen=True)
class PagedDecode:
    """One-token decode metadata for a heterogeneous request batch.

    ``cache_seqlens`` contains the committed length before the new token is
    appended. ``block_table`` maps each logical page to a physical pool page;
    unused columns point at scratch page 0 and are never read because
    ``cache_seqlens`` bounds every sequence.
    """
# page_size = 4

# req A：
# prompt tokens = [10, 11, 12]
# kv_seq_len = 3
# page_table = [1]
# next decode token = 30

# req B：
# prompt tokens = [20, 21, 22, 23, 24]
# kv_seq_len = 5
# page_table = [2, 3]
# next decode token = 40

# input_ids=[
#         [30],
#         [40],
#     ],
#     position_ids=[
#         [3],
#         [5],
#     ],
#     cache_seqlens=[
#         3,
#         5,
#     ],Do not confuse these two ,in sliding window kv cache,pos id != cache_seqlens
#     block_table=[
#         [1, 0],(0 is padded for req A, because it only has 1 page)
#         [2, 3],
#     ],

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor


def allocate_prefill_pages(requests: "Sequence[Request]", pool: KVMemoryPool) -> None:
    """Assign prompt-sized pages atomically, or leave every request unchanged."""
    allocated: list[Request] = []
    try:
        for request in requests:
            if request.page_table or request.kv_seq_len:
                raise ValueError(
                    f"request {request.request_id} already owns paged KV state"
                )
            if request.num_input_tokens < 1:
                raise ValueError(f"request {request.request_id} has an empty prompt")
            request.page_table = pool.allocate(pool.pages_needed(request.num_input_tokens))
            allocated.append(request)
    except Exception:
        for request in allocated:
            pool.free(request.page_table)
            request.page_table = []
        raise


def build_packed_prefill(
    requests: "Sequence[Request]", pool: KVMemoryPool
) -> PackedPrefill:
    """Build FlashAttention-style packed prompt metadata without padding."""
    if not requests:
        raise ValueError("packed prefill requires at least one request")

    packed_ids: list[int] = []
    packed_positions: list[int] = []
    slots: list[int] = []
    cu_seqlens = [0]
    max_seqlen = 0

    for request in requests:
        prompt_len = request.num_input_tokens
        expected_pages = pool.pages_needed(prompt_len)
        if len(request.page_table) != expected_pages:
            raise ValueError(
                f"request {request.request_id} needs {expected_pages} prompt pages, "
                f"but owns {len(request.page_table)}"
            )
        if request.kv_seq_len != 0:
            raise ValueError(
                f"request {request.request_id} has {request.kv_seq_len} committed KV tokens"
            )

        packed_ids.extend(request.input_ids)
        packed_positions.extend(range(prompt_len))
        slots.extend(
            _slot_for_position(request.page_table, position, pool.page_size)
            for position in range(prompt_len)
        )
        cu_seqlens.append(cu_seqlens[-1] + prompt_len)
        max_seqlen = max(max_seqlen, prompt_len)

    return PackedPrefill(
        input_ids=torch.tensor(packed_ids, dtype=torch.long, device=pool.device),
        position_ids=torch.tensor(packed_positions, dtype=torch.long, device=pool.device),
        # flash_attn_varlen_func requires cumulative lengths in int32.
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=pool.device),
        # index_copy_ uses int64 indices, so retain that type for pool writes.
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=pool.device),
        max_seqlen=max_seqlen,
    )


def commit_prefill(requests: "Sequence[Request]") -> None:
    """Mark prompt slots valid only after attention has written them."""
    for request in requests:
    #mark by updating kv_seq_len
        request.kv_seq_len = request.num_input_tokens


def build_paged_decode(
    requests: "Sequence[Request]", pool: KVMemoryPool
) -> PagedDecode:
    """Reserve one append position per request and build FA block metadata.

    Page growth is transactional for the whole batch. If any allocation fails,
    pages added for earlier requests in this call are returned and every page
    table is restored to its original length.
    """
    if not requests:
        raise ValueError("paged decode requires at least one request")

    original_page_counts = [len(request.page_table) for request in requests]
    try:
        for request in requests:
            if not request.output_ids:
                raise ValueError(
                    f"request {request.request_id} has no sampled token to decode"
                )
            reserve_decode_slot(request, pool)
    except Exception:
        for request, original_count in zip(requests, original_page_counts):
            newly_allocated = request.page_table[original_count:]
            pool.free(newly_allocated)
            del request.page_table[original_count:]
        raise

    max_pages = max(len(request.page_table) for request in requests)
    block_table = torch.full(
        (len(requests), max_pages),
        fill_value=pool.SCRATCH_PAGE,
        dtype=torch.int32,
        device=pool.device,
    )
    for row, request in enumerate(requests):
        block_table[row, : len(request.page_table)] = torch.tensor(
            request.page_table, dtype=torch.int32, device=pool.device
        )

    return PagedDecode(
        input_ids=torch.tensor(
            [[request.output_ids[-1]] for request in requests],
            dtype=torch.long,
            device=pool.device,
        ),
        position_ids=torch.tensor(
            [[request.kv_seq_len] for request in requests],
            dtype=torch.long,
            device=pool.device,
        ),
        cache_seqlens=torch.tensor(
            [request.kv_seq_len for request in requests],
            dtype=torch.int32,
            device=pool.device,
        ),
        block_table=block_table,
    )


def reserve_decode_slot(request: Request, pool: KVMemoryPool) -> int:
    """Reserve and return the physical slot for one next-token K/V write."""
    if request.kv_seq_len < 1:
        raise ValueError(f"request {request.request_id} has no committed prompt KV")

    logical_position = request.kv_seq_len
    required_pages = pool.pages_needed(logical_position + 1)
    if required_pages > len(request.page_table):
        # Allocate new pages for the request if it doesn't have enough to hold the next token.
        request.page_table.extend(pool.allocate(required_pages - len(request.page_table)))
    return _slot_for_position(request.page_table, logical_position, pool.page_size)


def commit_decode_token(request: Request) -> None:
    """Advance ``kv_seq_len`` after the reserved decode slot has been written."""
    #mark by updating kv_seq_len
    request.kv_seq_len += 1


def commit_decode_batch(requests: "Sequence[Request]") -> None:
    """Publish one successfully written KV token for every batch member."""
    for request in requests:
        commit_decode_token(request)


def release_request_pages(request: Request, pool: KVMemoryPool) -> None:
    """Return all physical pages owned by a finished or aborted request."""
    pool.free(request.page_table)
    request.page_table = []
    request.kv_seq_len = 0


def require_flash_attention() -> tuple[object, object]:
    """Lazily load the optional FlashAttention-2 functions for paged mode."""
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except ImportError as exc:
        raise RuntimeError(
            "paged attention requires the optional 'paged' dependency; "
            "install FlashAttention with `pip install --no-build-isolation '.[paged]'`"
        ) from exc
    return flash_attn_varlen_func, flash_attn_with_kvcache

# Helper function to calculate the physical slot for a given logical position.
def _slot_for_position(page_table: list[int], logical_position: int, page_size: int) -> int:
    page_offset, offset_in_page = divmod(logical_position, page_size)
    return page_table[page_offset] * page_size + offset_in_page
