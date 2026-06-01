"""Clip-level retrieval strategy wrapping mmagent.retrieve.search()."""

from typing import Dict, List, Tuple, Optional, Any
from .base import RetrievalStrategy


class ClipLevelStrategy(RetrievalStrategy):
    """Clip-level retrieval strategy using mmagent baseline search."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "clip_level"

    def search(
        self,
        video_graph: Any,
        query: str,
        current_context: List,
        topk: int,
        threshold: float,
        before_clip: Optional[int] = None,
        **kwargs
    ) -> Tuple[Dict, List, Optional[Dict]]:
        """Clip-level retrieval via original mmagent.retrieve.search()."""
        from mmagent.retrieve import search as mmagent_search

        memories, updated_clips, scores = mmagent_search(
            video_graph,
            query,
            current_context,
            topk=topk,
            threshold=threshold,
            before_clip=before_clip
        )

        metadata = {
            'strategy': self.name,
            'clip_scores': scores,
            'num_clips': len(memories),
        }

        return memories, updated_clips, metadata
