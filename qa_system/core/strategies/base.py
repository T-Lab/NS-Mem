"""Retrieval strategy base class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any


class RetrievalStrategy(ABC):
    """Retrieval strategy base class."""

    @abstractmethod
    def search(
        self,
        video_graph: Any,
        query: str,
        current_context: List,
        topk: int,
        threshold: float,
        before_clip: Optional[int] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], List, Optional[Dict]]:
        """Execute retrieval, returning (memories, updated_context, metadata)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    def reset(self):
        """Reset strategy state (call at the start of each new question)."""
        pass
