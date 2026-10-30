"""Token sampling utilities."""

from collections.abc import Sequence

import torch
from miniengine.core import SamplingParams


def apply_repetition_penalty(
    logits: torch.Tensor,
    output_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Penalize tokens that already appeared in the output."""
    if penalty == 1.0 or not output_ids:
        return logits
    token_ids = torch.tensor(output_ids, device=logits.device)
    scores = torch.gather(logits, -1, token_ids.unsqueeze(0))
    # Reduce probability of already-seen tokens
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits.scatter_(-1, token_ids.unsqueeze(0), scores)
    return logits


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    output_ids: list[int] | None = None,
) -> int:
    """
    Sample one token from logits using the given SamplingParams.

    Args:
        logits: shape (1, vocab_size), raw model output on the last position
        params: sampling configuration
        output_ids: previously generated tokens (for repetition penalty)

    Returns:
        A single token id (int).
    """
    logits = logits.clone()

    # Repetition penalty
    if output_ids:
        logits = apply_repetition_penalty(logits, output_ids, params.repetition_penalty)

    # Greedy
    if params.temperature == 0:
        return logits.argmax(dim=-1).item()

    # Temperature scaling
    logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus) filtering
    if 0 < params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Mask tokens with cumulative prob above top_p (keep at least one)
        mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= params.top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_batch(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    output_ids: Sequence[list[int]],
) -> list[int]:
    """Choose one token per row while preserving per-request sampling rules.

    Pure greedy requests share one GPU argmax and one device-to-host transfer.
    A heterogeneous batch falls back to the existing single-request sampler so
    temperature, top-k/top-p, and repetition penalties keep their semantics.
    """

    pure_greedy = all(
        item.temperature == 0
        and (item.repetition_penalty == 1.0 or not history)
        for item, history in zip(params, output_ids)
    )
    if pure_greedy:
        # tolist() is the single required synchronization that hands the next
        # token ids to the CPU scheduler; avoid one .item() per request.
        return logits.argmax(dim=-1).tolist()

    return [
        #reserve batch dim,ith request
        sample_token(logits[index : index + 1], item, history)
        for index, (item, history) in enumerate(zip(params, output_ids))
    ]
