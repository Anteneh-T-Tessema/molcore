"""
Domain events emitted by molcore skills.
Import and call emit() to fire structured events to your event bus.
Noop by default — set MOLCORE_EVENTS_ENDPOINT to enable delivery.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


ENDPOINT = os.environ.get("MOLCORE_EVENTS_ENDPOINT", "")


@dataclass
class FeaturizeEvent:
    skill: str = "fingerprint"
    smiles_count: int = 0
    backend: str = "rust"
    radius: int = 2
    nbits: int = 2048
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class SimilarityEvent:
    skill: str = "similarity_search"
    query_count: int = 0
    library_count: int = 0
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class AgenticRagEvent:
    skill: str = "agentic_rag"
    query: str = ""
    sources: list = None
    iterations: int = 0
    relevant: bool = True
    duration_ms: float = 0.0

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


def emit(event) -> None:
    """Fire a domain event. Noop if MOLCORE_EVENTS_ENDPOINT is not set."""
    if not ENDPOINT:
        return
    payload = json.dumps({
        "timestamp": time.time(),
        "event": event.skill,
        **asdict(event),
    })
    try:
        import urllib.request
        req = urllib.request.Request(
            ENDPOINT,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass  # observability must never break the hot path
