from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


# Tree node
@dataclass
class Chunk:
    start: int  # byte offset in the original string
    end: int  # exclusive
    text: str

    def __repr__(self) -> str:  # compact / readable when printing
        t = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return f"Chunk({self.start}:{self.end}, {len(self.text)} ch, {t!r} "


class AbstractChunker(ABC):
    """
    Abstract base class for text chunkers.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Return a list of top-level Chunk objects.
        """
        ...
