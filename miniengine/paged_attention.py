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
class PrefillSuffix:
    """One uncached prompt suffix following a request's committed KV prefix."""

    request: Request
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    @property
    def completes_prompt(self) -> bool:
        return self.end == self.request.num_input_tokens

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

@dataclass(frozen=True)
class PackedPrefillSuffix:
    """Ragged query suffixes attending to their full committed paged prefixes.

    ``input_ids`` and ``position_ids`` cover only newly scheduled query tokens.
    ``cu_seqlens_k`` describes each sequence after its new suffix is appended,
    while ``block_table`` maps those logical histories into the stable KV pool.
    """

    # Example with page_size = 4 and two requests:
    #
    # request A:
    #   input_ids = [10, 11, 12, 13, 14]
    #   committed KV prefix = [10, 11]        (kv_seq_len = 2)
    #   scheduled suffix = [12, 13, 14]       ([start, end) = [2, 5))
    #   page_table = [1, 4]
    #
    # request B:
    #   input_ids = [20, 21, 22, 23, 24, 25, 26, 27]
    #   committed KV prefix = [20, 21, 22, 23] (kv_seq_len = 4)
    #   scheduled suffix = [24, 25]            ([start, end) = [4, 6))
    #   page_table = [2, 3]
    #
    # input_ids = [12, 13, 14, 24, 25]
    # position_ids = [2, 3, 4, 4, 5]
    #
    # Q contains only this step's new suffix tokens: lengths [3, 2].
    # cu_seqlens_q = [0, 3, 5]
    #
    # K contains the full visible history after appending each suffix:
    # lengths [5, 6], so cu_seqlens_k = [0, 5, 11].
    #
    # For A's logical position 2:
    #   logical_page = 2 // page_size = 2 // 4 = 0
    #   offset_in_page = 2 % page_size = 2 % 4 = 2
    #   physical_page = page_table[logical_page] = page_table[0] = 1
    #   physical_slot = physical_page * page_size + offset_in_page
    #                 = 1 * 4 + 2 = 6
    # Therefore, A's logical positions 3 and 4 map to slots 7 and 16.
    #
    # Physical slot = physical_page * page_size + offset_in_page.
    # slot_mapping = [6, 7, 16, 12, 13]
    #   A positions [2, 3, 4] -> page 1 offsets [2, 3], page 4 offset [0]
    #   B positions [4, 5]    -> page 3 offsets [0, 1]
    #
    # block_table = [
    #     [1, 4],
    #     [2, 3],
    # ]
    #
    # A reaches the end of its prompt, but B still has tokens [26, 27] left.
    # sample_indices = [2] selects A's final new-token hidden state (token 14).
    # B is not sampled.
    # max_seqlen_q = 3
    # max_seqlen_k = 6

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    sample_indices: torch.Tensor
    #used by fa to select appropriate CUDA kernel,tile size, and shared memory allocation etc
    max_seqlen_q: int
    max_seqlen_k: int


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


def allocate_prefill_suffix_pages(
    suffixes: "Sequence[PrefillSuffix]", pool: KVMemoryPool
) -> list[int]:
    """Grow suffix page tables atomically and return their original lengths."""
    if not suffixes:
        raise ValueError("prefix-continuation prefill requires at least one suffix")

    original_page_counts = [len(suffix.request.page_table) for suffix in suffixes]
    seen_requests: set[int] = set()
    try:
        for suffix in suffixes:
            request = suffix.request
            request_identity = id(request)
            if request_identity in seen_requests:
                raise ValueError(
                    f"request {request.request_id} appears more than once in one suffix batch"
                )
            seen_requests.add(request_identity)
            if suffix.start != request.kv_seq_len:
                raise ValueError(
                    f"request {request.request_id} suffix starts at {suffix.start}, "
                    f"but committed KV length is {request.kv_seq_len}"
                )
            if not 0 <= suffix.start < suffix.end <= request.num_input_tokens:
                raise ValueError(
                    f"request {request.request_id} has invalid prefill suffix "
                    f"[{suffix.start}, {suffix.end})"
                )
            expected_existing_pages = pool.pages_needed(suffix.start)
            if len(request.page_table) != expected_existing_pages:
                raise ValueError(
                    f"request {request.request_id} owns {len(request.page_table)} pages "
                    f"for {suffix.start} committed tokens; expected {expected_existing_pages}"
                )

            required_pages = pool.pages_needed(suffix.end)
            missing_pages = required_pages - len(request.page_table)
            request.page_table.extend(pool.allocate(missing_pages))
    except Exception:
        rollback_prefill_suffix_pages(suffixes, original_page_counts, pool)
        raise
    return original_page_counts


def rollback_prefill_suffix_pages(
    suffixes: "Sequence[PrefillSuffix]",
    original_page_counts: "Sequence[int]",
    pool: KVMemoryPool,
) -> None:
    """Return pages added by an uncommitted suffix batch."""
    for suffix, original_count in zip(suffixes, original_page_counts):
        request = suffix.request
        newly_allocated = request.page_table[original_count:]
        pool.free(newly_allocated)
        del request.page_table[original_count:]


def build_packed_prefill_suffix(
    suffixes: "Sequence[PrefillSuffix]", pool: KVMemoryPool
) -> PackedPrefillSuffix:
    """Build packed query metadata against full paged KV histories."""
    if not suffixes:
        raise ValueError("prefix-continuation prefill requires at least one suffix")

    packed_ids: list[int] = []
    packed_positions: list[int] = []
    slots: list[int] = []
    sample_indices: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0

    for suffix in suffixes:
        request = suffix.request
        if suffix.start != request.kv_seq_len:
            raise ValueError(
                f"request {request.request_id} suffix does not start at committed KV"
            )
        expected_pages = pool.pages_needed(suffix.end)
        if len(request.page_table) != expected_pages:
            raise ValueError(
                f"request {request.request_id} needs {expected_pages} pages through "
                f"token {suffix.end}, but owns {len(request.page_table)}"
            )
        #for fa
        packed_ids.extend(request.input_ids[suffix.start : suffix.end])
        packed_positions.extend(range(suffix.start, suffix.end))
        #slot_mapping is used to map logical position to physical slot in KVMemoryPool
        slots.extend(
            _slot_for_position(request.page_table, position, pool.page_size)
            for position in range(suffix.start, suffix.end)
        )
        cu_seqlens_q.append(cu_seqlens_q[-1] + suffix.num_tokens)
        cu_seqlens_k.append(cu_seqlens_k[-1] + suffix.end)
        max_seqlen_q = max(max_seqlen_q, suffix.num_tokens)
        max_seqlen_k = max(max_seqlen_k, suffix.end)
        if suffix.completes_prompt:
            #sample only when completes
            sample_indices.append(cu_seqlens_q[-1] - 1)

    max_pages = max(len(suffix.request.page_table) for suffix in suffixes)
    #non ragged square
    block_table = torch.full(
        #len(suffixes)*max_pages
        (len(suffixes), max_pages),
        fill_value=pool.SCRATCH_PAGE,
        dtype=torch.int32,
        device=pool.device,
    )
    for row, suffix in enumerate(suffixes):
        page_table = suffix.request.page_table
        block_table[row, : len(page_table)] = torch.tensor(
            page_table, dtype=torch.int32, device=pool.device
        )

    return PackedPrefillSuffix(
        input_ids=torch.tensor(packed_ids, dtype=torch.long, device=pool.device),
        position_ids=torch.tensor(
            packed_positions, dtype=torch.long, device=pool.device
        ),
        cu_seqlens_q=torch.tensor(
            cu_seqlens_q, dtype=torch.int32, device=pool.device
        ),
        cu_seqlens_k=torch.tensor(
            cu_seqlens_k, dtype=torch.int32, device=pool.device
        ),
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=pool.device),
        block_table=block_table,
        sample_indices=torch.tensor(
            sample_indices, dtype=torch.long, device=pool.device
        ),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def commit_prefill_suffixes(suffixes: "Sequence[PrefillSuffix]") -> None:
    """Publish every successfully written suffix end position."""
    for suffix in suffixes:
        suffix.request.kv_seq_len = suffix.end


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
