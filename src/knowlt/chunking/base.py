from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, cast, Literal, Callable
from knowlt.project import ProjectManager
from .models import AbstractChunker
from .recursive import RecursiveChunker


ChunkerType = Literal["recursive"]


def create_chunker(
    max_tokens: int,
    min_tokens: int,
    chunker_type: ChunkerType,
    token_counter: Callable[[str], int],
) -> AbstractChunker:
    """
    Factory to construct a text chunker.

    Parameters
    ----------
    chunker_type : Chunking implementation to use.
    max_tokens : The maximum number of tokens a leaf chunk may contain.
    token_counter : Used to count tokens in text.

    Returns
    -------
    An instance of an AbstractChunker.
    """
    if chunker_type == "recursive":
        return RecursiveChunker(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            token_counter=token_counter,
        )

    raise ValueError(f"Unknown chunker type: {chunker_type}")


def create_project_chunker(pm: ProjectManager) -> AbstractChunker:
    """
    Helper to create a chunker based on project settings.
    """

    chunking_settings = pm.settings.chunking

    if pm.embeddings:
        token_counter = pm.embeddings.get_token_count
        max_tokens = pm.embeddings.get_max_context_length()
        min_tokens = chunking_settings.min_tokens
    else:
        token_counter = lambda text: len(text.split())
        max_tokens = chunking_settings.max_tokens
        min_tokens = chunking_settings.min_tokens

    chunker_type = chunking_settings.chunker_type

    return create_chunker(
        chunker_type=cast(Literal["recursive"], chunker_type),
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        token_counter=token_counter,
    )
