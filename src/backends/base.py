from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import BackendDescriptor, Transcript, TranscriptionRequest


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Common interface for host-side transcription backends."""

    def descriptor(self) -> BackendDescriptor:
        """Return a static description of the backend and its partitioning."""
        ...

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Execute a transcription request and return a transcript result."""
        ...
