"""Episodic link builder with verification and auto-discovery."""

from typing import Dict, List, Set, Optional
import numpy as np
from .utils import get_normalized_embedding, batch_get_normalized_embeddings


class EpisodicLinker:
    """Build verified episodic links via embedding similarity."""

    def __init__(
        self,
        verify_threshold: float = 0.35,
        discover_threshold: float = 0.50,
        max_links_per_proc: int = 10,
        debug: bool = False
    ):
        self.verify_threshold = verify_threshold
        self.discover_threshold = discover_threshold
        self.max_links_per_proc = max_links_per_proc
        self.debug = debug
        self._content_emb_cache: Dict[int, np.ndarray] = {}

    def _ensure_content_embeddings(self, all_episodic_contents: List[Dict]):
        """Batch pre-compute and cache all content embeddings."""
        if self._content_emb_cache:
            return

        texts = []
        clip_ids = []
        for item in all_episodic_contents:
            clip_id = item['clip_id']
            content = item.get('content', '')
            if content and clip_id not in self._content_emb_cache:
                texts.append(content)
                clip_ids.append(clip_id)

        if not texts:
            return

        all_embs = batch_get_normalized_embeddings(texts)

        for clip_id, emb in zip(clip_ids, all_embs):
            self._content_emb_cache[clip_id] = emb

    def build_verified_links(
        self,
        procedure: Dict,
        all_episodic_contents: List[Dict],
    ) -> List[Dict]:
        """Build verified episodic links by similarity-based verification and discovery."""
        links = []

        llm_clips = set()
        for clip in procedure.get('source_clips', []):
            if isinstance(clip, int):
                llm_clips.add(clip)
            elif isinstance(clip, str):
                import re
                match = re.search(r'(\d+)', clip)
                if match:
                    llm_clips.add(int(match.group(1)))

        self._ensure_content_embeddings(all_episodic_contents)

        goal = procedure.get('goal', '')
        description = procedure.get('description', '')
        proc_text = f"{goal}. {description}"
        proc_emb = get_normalized_embedding(proc_text)

        if proc_emb is None:
            return links

        candidates = []

        for item in all_episodic_contents:
            clip_id = item['clip_id']
            content = item.get('content', '')

            if not content:
                continue

            content_emb = self._content_emb_cache.get(clip_id)
            if content_emb is None:
                continue

            sim = float(np.dot(proc_emb, content_emb))

            if clip_id in llm_clips:
                if sim >= self.verify_threshold:
                    candidates.append({
                        'clip_id': clip_id,
                        'relevance': 'source',
                        'similarity': round(sim, 4),
                        'content_preview': content[:100] if content else None
                    })

            elif sim >= self.discover_threshold:
                candidates.append({
                    'clip_id': clip_id,
                    'relevance': 'discovered',
                    'similarity': round(sim, 4),
                    'content_preview': content[:100] if content else None
                })

        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        links = candidates[:self.max_links_per_proc]

        return links

    def clear_cache(self):
        """Clear embedding cache."""
        self._content_emb_cache.clear()


def create_episodic_linker(
    verify_threshold: float = 0.35,
    discover_threshold: float = 0.50,
    debug: bool = False
) -> EpisodicLinker:
    """Factory function."""
    return EpisodicLinker(
        verify_threshold=verify_threshold,
        discover_threshold=discover_threshold,
        debug=debug
    )
