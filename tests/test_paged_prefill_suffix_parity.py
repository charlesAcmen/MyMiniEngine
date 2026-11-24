"""CUDA parity test between M2 full prefill and M3 prefix-continuation prefill."""

from __future__ import annotations

import unittest

import torch

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, PagedAttentionOps
from miniengine.paged_attention import (
    PrefillSuffix,
    allocate_prefill_suffix_pages,
    allocate_prefill_pages,
    build_packed_prefill_suffix,
    build_packed_prefill,
    commit_prefill_suffixes,
    require_flash_attention,
)


def _flash_attention_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        require_flash_attention()
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(
    _flash_attention_available(),
    "prefix-continuation prefill parity requires CUDA and FlashAttention",
)
class PrefillSuffixParityTests(unittest.TestCase):
    def test_suffix_logits_match_single_shot_packed_prefill(self) -> None:
        torch.manual_seed(7)
        config = ModelConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        model = CausalLM(config).to(device="cuda", dtype=torch.bfloat16)
        model.eval()
        prefill_op, decode_op = require_flash_attention()
        model.set_paged_attention_ops(PagedAttentionOps(prefill_op, decode_op))

        def make_pool() -> KVMemoryPool:
            return KVMemoryPool(
                num_pages=5,
                page_size=256,
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )

        a_ids = torch.randint(0, config.vocab_size, (17,)).tolist()
        b_ids = torch.randint(0, config.vocab_size, (23,)).tolist()

        full_pool = make_pool()
        full_requests = [
            Request(request_id="a-full", input_ids=a_ids),
            Request(request_id="b-full", input_ids=b_ids),
        ]
        allocate_prefill_pages(full_requests, full_pool)
        full = build_packed_prefill(full_requests, full_pool)
        with torch.inference_mode():
            full_logits = model.forward_paged_prefill(
                full.input_ids,
                full.position_ids,
                full_pool.kv_caches,
                full.slot_mapping,
                full.cu_seqlens,
                full.max_seqlen,
            )

        suffix_pool = make_pool()
        a = Request(request_id="a", input_ids=a_ids)
        b = Request(request_id="b", input_ids=b_ids)

        def run_suffixes(suffixes: list[PrefillSuffix]) -> torch.Tensor:
            allocate_prefill_suffix_pages(suffixes, suffix_pool)
            packed = build_packed_prefill_suffix(suffixes, suffix_pool)
            with torch.inference_mode():
                logits = model.forward_paged_prefill_suffix(
                    packed.input_ids,
                    packed.position_ids,
                    suffix_pool.kv_caches,
                    packed.slot_mapping,
                    packed.cu_seqlens_q,
                    packed.cu_seqlens_k,
                    packed.block_table,
                    packed.sample_indices,
                    packed.max_seqlen_q,
                    packed.max_seqlen_k,
                )
            commit_prefill_suffixes(suffixes)
            return logits

        first_logits = run_suffixes([PrefillSuffix(a, 0, 8)])
        self.assertEqual(tuple(first_logits.shape), (0, config.vocab_size))

        a_logits = run_suffixes(
            [PrefillSuffix(a, 8, 17), PrefillSuffix(b, 0, 7)]
        )
        b_logits = run_suffixes([PrefillSuffix(b, 7, 23)])
        suffix_logits = torch.cat([a_logits, b_logits], dim=0)

        torch.testing.assert_close(
            suffix_logits,
            full_logits,
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertEqual((a.kv_seq_len, b.kv_seq_len), (17, 23))


if __name__ == "__main__":
    unittest.main()

