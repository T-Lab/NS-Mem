"""Shared embedding cache and similarity utilities."""

from typing import List, Dict, Optional
import numpy as np

_embedding_cache: Dict[str, np.ndarray] = {}


def get_normalized_embedding(text: str, use_cache: bool = True) -> np.ndarray:
    """Get a normalized embedding vector with optional caching."""
    if use_cache and text in _embedding_cache:
        return _embedding_cache[text]

    from mmagent.utils.chat_api import parallel_get_embedding
    embs, _ = parallel_get_embedding("text-embedding-3-large", [text])
    vec = np.array(embs[0])
    normalized = vec / (np.linalg.norm(vec) + 1e-8)

    if use_cache:
        _embedding_cache[text] = normalized
    return normalized


def batch_get_normalized_embeddings(texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
    """Batch compute normalized embeddings, using cache where available."""
    results: List[Optional[np.ndarray]] = [None] * len(texts)
    texts_to_fetch: List[str] = []
    indices_to_fetch: List[int] = []

    for i, text in enumerate(texts):
        if use_cache and text in _embedding_cache:
            results[i] = _embedding_cache[text]
        else:
            texts_to_fetch.append(text)
            indices_to_fetch.append(i)

    if texts_to_fetch:
        from mmagent.utils.chat_api import parallel_get_embedding
        embs, _ = parallel_get_embedding("text-embedding-3-large", texts_to_fetch)
        for idx, text, emb in zip(indices_to_fetch, texts_to_fetch, embs):
            vec = np.array(emb)
            normalized = vec / (np.linalg.norm(vec) + 1e-8)
            if use_cache:
                _embedding_cache[text] = normalized
            results[idx] = normalized

    return results


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return {
        'cached_embeddings': len(_embedding_cache),
        'memory_estimate_mb': len(_embedding_cache) * 3072 * 8 / (1024 * 1024)
    }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(vec1, vec2))
