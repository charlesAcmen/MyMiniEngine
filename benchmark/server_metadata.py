"""HTTP metadata shared by benchmark clients."""

from __future__ import annotations


def fetch_server_model_metadata(base_url: str) -> dict:
    """Return the first model descriptor from a MiniEngine server."""
    import requests

    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        model = response.json()["data"][0]
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError("could not read server model metadata") from exc
    if not isinstance(model, dict):
        raise RuntimeError("server model metadata must be an object")
    return model
