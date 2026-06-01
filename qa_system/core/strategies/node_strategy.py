"""Node-level retrieval strategy using per-node embedding similarity."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from .base import RetrievalStrategy


class NodeLevelStrategy(RetrievalStrategy):
    """Node-level retrieval strategy (independent implementation)."""

    def __init__(
        self,
        include_timestamp: bool = True,
        group_by_clip: bool = True,
        include_semantic: bool = False,
        preserve_clip_order: bool = True,
    ):
        self.include_timestamp = include_timestamp
        self.group_by_clip = group_by_clip
        self.include_semantic = include_semantic
        self.preserve_clip_order = preserve_clip_order
        self._retrieved_node_ids: Set[int] = set()

    @property
    def name(self) -> str:
        return "node_level"

    def reset(self):
        """Reset dedup state for each new question."""
        self._retrieved_node_ids.clear()

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
        """Node-level retrieval with embedding similarity, filtering, and grouping."""
        import random
        from mmagent.retrieve import back_translate, translate
        from mmagent.utils.chat_api import parallel_get_embedding

        queries = back_translate(video_graph, [query])
        if len(queries) > 100:
            queries = random.sample(queries, 100)

        model = "text-embedding-3-large"
        query_embeddings = parallel_get_embedding(model, queries)[0]
        query_embeddings = np.array(query_embeddings)

        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_embeddings = query_embeddings / norms

        nodes_with_scores = []
        for node_id, node in video_graph.nodes.items():
            if node.type in ['img', 'voice']:
                continue
            if not self.include_semantic and node.type == 'semantic':
                continue

            node_emb = getattr(node, 'embeddings', None)
            if node_emb is None:
                continue

            try:
                if isinstance(node_emb, list):
                    if len(node_emb) == 0:
                        continue
                    if isinstance(node_emb[0], list):
                        vec = np.array(node_emb[0], dtype=float)
                    elif isinstance(node_emb[0], (int, float)):
                        vec = np.array(node_emb, dtype=float)
                    else:
                        continue
                else:
                    vec = np.array(node_emb, dtype=float).flatten()

                if vec.size < 64:
                    continue

                query_dim = query_embeddings.shape[1]
                if vec.size > query_dim:
                    vec = vec[:query_dim]
                elif vec.size < query_dim:
                    vec = np.pad(vec, (0, query_dim - vec.size))

                node_embedding = vec.reshape(1, -1)
            except Exception:
                continue

            node_norm = np.linalg.norm(node_embedding)
            if node_norm == 0:
                continue
            node_embedding = node_embedding / node_norm

            similarities = np.dot(query_embeddings, node_embedding.T)
            score = float(np.max(similarities))

            nodes_with_scores.append((node_id, score))

        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)

        filtered_nodes = []
        for node_id, score in nodes_with_scores:
            if node_id in self._retrieved_node_ids:
                continue

            node = video_graph.nodes[node_id]
            clip_id = node.metadata.get('timestamp', -1)

            if before_clip is not None and clip_id > before_clip:
                continue

            if score < threshold:
                continue

            content = node.metadata.get('contents', [''])[0]

            filtered_nodes.append({
                'node_id': node_id,
                'score': score,
                'clip_id': clip_id,
                'content': content,
                'type': node.type,
            })

            if len(filtered_nodes) >= topk:
                break

        for n in filtered_nodes:
            self._retrieved_node_ids.add(n['node_id'])

        if self.group_by_clip:
            memories = self._format_grouped(video_graph, filtered_nodes)
        else:
            memories = self._format_flat(video_graph, filtered_nodes)

        updated_context = current_context + [n['content'] for n in filtered_nodes]

        metadata = {
            'strategy': self.name,
            'num_nodes': len(filtered_nodes),
            'node_scores': [(n['node_id'], n['score']) for n in filtered_nodes],
            'total_retrieved': len(self._retrieved_node_ids),
        }

        return memories, updated_context, metadata

    def _format_flat(self, video_graph, nodes: List[Dict]) -> Dict:
        """Flat format: nodes sorted by relevance."""
        from mmagent.retrieve import translate

        memories = {"RETRIEVED_NODES": []}
        for n in nodes:
            content = translate(video_graph, [n['content']])[0]
            if self.include_timestamp:
                entry = f"[CLIP_{n['clip_id']}] {content}"
            else:
                entry = content
            memories["RETRIEVED_NODES"].append(entry)

        return memories

    def _format_grouped(self, video_graph, nodes: List[Dict]) -> Dict:
        """Grouped format: grouped by clip (baseline format compatible)."""
        from mmagent.retrieve import translate

        grouped = defaultdict(list)
        for n in nodes:
            grouped[n['clip_id']].append(n)

        if self.preserve_clip_order:
            for clip_id in grouped:
                original_order = getattr(video_graph, 'text_nodes_by_clip', {}).get(clip_id, [])
                order_map = {nid: i for i, nid in enumerate(original_order)}
                grouped[clip_id].sort(key=lambda x: order_map.get(x['node_id'], float('inf')))

        memories = {}
        for clip_id in sorted(grouped.keys()):
            contents = [translate(video_graph, [n['content']])[0] for n in grouped[clip_id]]
            memories[f"CLIP_{clip_id}"] = contents

        return memories
