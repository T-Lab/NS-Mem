import os
import re
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from env_setup import setup_all
setup_all()

from mmagent.retrieve import search as baseline_search
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import parallel_get_embedding

from .cache_manager import cache
from .query_classifier import QueryClassifier, QueryType, ClassificationResult
from .name_resolver import NameResolver, create_resolver
from .symbolic_functions import (
    SymbolicFunctions,
    ProcedureResult,
    StepQueryResult,
    CharacterResult
)


@dataclass
class RetrievalConfig:
    """Retrieval config base class."""
    threshold: float = 0.3
    topk: int = 10
    include_episodic_evidence: bool = True


@dataclass
class BaselineConfig(RetrievalConfig):
    """Baseline mode config."""
    threshold: float = 0.3
    topk: int = 10
    mem_wise: bool = False


@dataclass
class NSTFConfig(RetrievalConfig):
    """NSTF mode config."""
    threshold: float = 0.35
    min_confidence: float = 0.30
    max_procedures: int = 3
    topk_baseline: int = 10
    threshold_baseline: float = 0.3
    include_episodic_evidence: bool = True
    use_reranking: bool = False
    use_dag_paths: bool = True
    factual_hybrid: bool = True


@dataclass
class RetrievalResult:
    """Retrieval result container."""
    memories: Dict[str, Any]
    clips: List[int]
    metadata: Dict[str, Any]


class HybridRetriever:
    """Hybrid retriever managing baseline, NSTF, and ablation modes."""

    def __init__(
        self,
        mode: str = "baseline",
        baseline_config: BaselineConfig = None,
        nstf_config: NSTFConfig = None,
    ):
        self.mode = mode
        self.baseline_config = baseline_config or BaselineConfig()
        self.nstf_config = nstf_config or NSTFConfig()

        self.query_classifier = QueryClassifier()
        self.symbolic_functions = SymbolicFunctions()
        self.name_resolver: Optional[NameResolver] = None

        self._current_video_graph = None
        self._current_nstf_graph = None

    def search(
        self,
        mem_path: str,
        query: str,
        current_clips: List = None,
        nstf_path: Optional[str] = None,
        before_clip: Optional[int] = None,
    ) -> RetrievalResult:
        """Execute hybrid retrieval."""
        if current_clips is None:
            current_clips = []

        video_graph = self._load_video_graph(mem_path)
        if before_clip is not None:
            video_graph.truncate_memory_by_clip(before_clip, False)
        video_graph.refresh_equivalences()

        self._current_video_graph = video_graph
        self.name_resolver = NameResolver(video_graph)

        if self._contains_entity_id(query):
            return self._handle_entity_id_query(video_graph, query, before_clip)

        if self.mode == "baseline":
            return self._search_baseline(video_graph, query, current_clips, before_clip)

        elif self.mode == "nstf_full":
            nstf_graph = self._load_nstf_graph(nstf_path) if nstf_path else None
            return self._search_nstf_full(
                video_graph, nstf_graph, query, current_clips, before_clip
            )

        elif self.mode.startswith("ablation_"):
            nstf_graph = self._load_nstf_graph(nstf_path) if nstf_path else None
            ablation_type = self.mode.replace("ablation_", "")
            return self._search_ablation(
                video_graph, nstf_graph, query, current_clips,
                ablation_type, before_clip
            )

        else:
            return self._search_baseline(video_graph, query, current_clips, before_clip)

    def _search_baseline(
        self,
        video_graph,
        query: str,
        current_clips: List,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Baseline retrieval using M3-Agent search."""
        config = self.baseline_config

        memories, current_clips, _ = baseline_search(
            video_graph, query, current_clips,
            threshold=config.threshold,
            topk=config.topk,
            before_clip=before_clip
        )

        return RetrievalResult(
            memories=memories,
            clips=current_clips,
            metadata={
                'mode': 'baseline',
                'retriever': 'M3-Agent',
            }
        )

    def _search_nstf_full(
        self,
        video_graph,
        nstf_graph: Optional[Dict],
        query: str,
        current_clips: List,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Full NSTF retrieval with query classification and symbolic functions."""
        config = self.nstf_config
        metadata = {'mode': 'nstf_full', 'retriever': 'HybridRetriever'}

        if nstf_graph is None:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason="No NSTF graph available"
            )

        self._current_nstf_graph = nstf_graph
        self.symbolic_functions.set_graphs(video_graph, nstf_graph)
        self.name_resolver = NameResolver(video_graph, nstf_graph)

        classification = self.query_classifier.classify(query)
        metadata['query_type'] = classification.query_type.value
        metadata['classification_confidence'] = classification.confidence
        metadata['classification_method'] = classification.method

        proc_embeddings = self._get_procedure_embeddings(nstf_graph)

        if not proc_embeddings:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason="No procedure embeddings"
            )

        matched_procs = self._search_procedures(query, proc_embeddings)

        if not matched_procs or matched_procs[0]['similarity'] < config.min_confidence:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason=f"Top sim {matched_procs[0]['similarity'] if matched_procs else 0:.3f} < min_confidence"
            )

        if config.use_reranking:
            matched_procs = self._apply_reranking(
                matched_procs, classification.query_type, nstf_graph
            )

        metadata['matched_procedures'] = [
            {
                'proc_id': p['proc_id'],
                'similarity': p['similarity'],
                'match_type': p['match_type']
            }
            for p in matched_procs[:config.max_procedures]
        ]

        memories = {}

        if classification.query_type == QueryType.PROCEDURAL:
            step_result = self.symbolic_functions.query_step_sequence(
                matched_procs[0]['proc_id'],
                query,
                use_dag=config.use_dag_paths
            )
            memories['NSTF_StepQuery'] = self.symbolic_functions.format_step_query_result(step_result)
            metadata['symbolic_function'] = 'query_step_sequence'

        elif getattr(classification, 'involves_character', False):
            character_id = self._extract_character_from_query(query, video_graph)
            if character_id:
                char_result = self.symbolic_functions.aggregate_character_behaviors(
                    character_id, self.name_resolver
                )
                memories['NSTF_CharacterAnalysis'] = self.symbolic_functions.format_character_result(char_result)
                metadata['symbolic_function'] = 'aggregate_character_behaviors'
                metadata['character_id'] = character_id
            else:
                self._add_procedure_memories(memories, matched_procs, nstf_graph, config, video_graph)
                metadata['symbolic_function'] = 'get_procedure_with_evidence'

        elif classification.query_type == QueryType.CONSTRAINT:
            self._add_procedure_memories(memories, matched_procs, nstf_graph, config, video_graph)
            metadata['symbolic_function'] = 'query_step_sequence'

            from qa_system.core.symbolic_functions import extract_constraints_from_query
            constraints = extract_constraints_from_query(query)
            metadata['constraints'] = constraints

            for proc in matched_procs[:config.max_procedures]:
                step_result = self.symbolic_functions.query_step_sequence(
                    proc['proc_id'], query, constraints=constraints, use_dag=True
                )
                if step_result.paths and len(step_result.paths) > 1:
                    memories[f'NSTF_Paths_{proc["proc_id"]}'] = (
                        self.symbolic_functions.format_step_query_result(step_result)
                    )
                elif step_result.full_sequence:
                    memories[f'NSTF_StepQuery_{proc["proc_id"]}'] = (
                        self.symbolic_functions.format_step_query_result(step_result)
                    )
        else:
            if config.factual_hybrid:
                baseline_memories, current_clips, _ = baseline_search(
                    video_graph, query, current_clips,
                    threshold=config.threshold_baseline,
                    topk=config.topk_baseline,
                    before_clip=before_clip
                )
                memories.update(baseline_memories)
                metadata['symbolic_function'] = 'hybrid_factual'
                metadata['baseline_clips'] = len(baseline_memories)

            if matched_procs and matched_procs[0]['similarity'] >= config.threshold:
                self._add_procedure_memories(memories, matched_procs, nstf_graph, config, video_graph)
                if 'symbolic_function' not in metadata or metadata['symbolic_function'] == 'hybrid_factual':
                    metadata['symbolic_function'] = 'hybrid_factual_with_proc'
            elif not config.factual_hybrid:
                self._add_procedure_memories(memories, matched_procs, nstf_graph, config, video_graph)
                metadata['symbolic_function'] = 'get_procedure_with_evidence'

        evidence_clips = {}
        if config.include_episodic_evidence:
            evidence_clips = self._extract_episodic_evidence(
                matched_procs[:config.max_procedures],
                nstf_graph,
                video_graph
            )
            for clip_id, clip_content in evidence_clips.items():
                resolved_content = self.name_resolver.resolve_text(clip_content)
                memories[f'clip_{clip_id}'] = resolved_content
                if clip_id not in current_clips:
                    current_clips.append(clip_id)

        metadata['num_evidence_clips'] = len(evidence_clips)
        metadata['decision'] = 'use_nstf'

        return RetrievalResult(
            memories=memories,
            clips=current_clips,
            metadata=metadata
        )

    def _search_ablation(
        self,
        video_graph,
        nstf_graph: Optional[Dict],
        query: str,
        current_clips: List,
        ablation_type: str,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Ablation experiment retrieval."""
        metadata = {'mode': f'ablation_{ablation_type}', 'retriever': 'HybridRetriever'}

        if nstf_graph is None:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason="No NSTF graph for ablation"
            )

        if ablation_type == "prototype":
            return self._ablation_prototype(video_graph, nstf_graph, query, current_clips, before_clip)

        elif ablation_type == "structure":
            return self._ablation_structure(video_graph, nstf_graph, query, current_clips, before_clip)

        else:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason=f"Unknown ablation type: {ablation_type}"
            )

    def _ablation_prototype(
        self,
        video_graph,
        nstf_graph: Dict,
        query: str,
        current_clips: List,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Ablation A: vector-only retrieval without procedure structure."""
        config = self.nstf_config

        memories, current_clips, _ = baseline_search(
            video_graph, query, current_clips,
            threshold=config.threshold,
            topk=config.topk_baseline,
            before_clip=before_clip
        )

        return RetrievalResult(
            memories=memories,
            clips=current_clips,
            metadata={
                'mode': 'ablation_prototype',
                'decision': 'prototype_only',
            }
        )

    def _ablation_structure(
        self,
        video_graph,
        nstf_graph: Dict,
        query: str,
        current_clips: List,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Ablation B: structure without symbolic reasoning."""
        config = self.nstf_config

        proc_embeddings = self._get_procedure_embeddings(nstf_graph)

        if not proc_embeddings:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason="No procedure embeddings for structure ablation"
            )

        matched_procs = self._search_procedures(query, proc_embeddings)

        if not matched_procs:
            return self._fallback_to_baseline(
                video_graph, query, current_clips, before_clip,
                reason="No procedures matched"
            )

        memories = {}
        self._add_procedure_memories(memories, matched_procs, nstf_graph, config, video_graph)

        return RetrievalResult(
            memories=memories,
            clips=current_clips,
            metadata={
                'mode': 'ablation_structure',
                'decision': 'structure_only',
                'matched_procedures': [p['proc_id'] for p in matched_procs[:config.max_procedures]],
            }
        )

    def _apply_reranking(
        self,
        matched_procs: List[Dict],
        query_type: QueryType,
        nstf_graph: Dict
    ) -> List[Dict]:
        """Type-aware re-ranking of matched procedures."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})

        def get_rerank_score(proc_match: Dict) -> float:
            base_score = proc_match['similarity']
            proc = proc_nodes.get(proc_match['proc_id'], {})

            boost = 0.0

            if query_type == QueryType.PROCEDURAL:
                num_steps = len(proc.get('steps', []))
                if num_steps >= 3:
                    boost += 0.05
                if num_steps >= 5:
                    boost += 0.03

            elif query_type == QueryType.CONSTRAINT:
                proc_type = proc.get('proc_type', '')
                if 'alternative' in proc_type.lower():
                    boost += 0.08

            return base_score + boost

        for proc in matched_procs:
            proc['rerank_score'] = get_rerank_score(proc)

        matched_procs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return matched_procs

    def _contains_entity_id(self, query: str) -> bool:
        """Check if query contains entity ID reference."""
        return "character id" in query

    def _handle_entity_id_query(
        self,
        video_graph,
        query: str,
        before_clip: Optional[int]
    ) -> RetrievalResult:
        """Handle query containing explicit entity ID."""
        memories, _, _ = baseline_search(
            video_graph, query, [],
            mem_wise=True, topk=20,
            before_clip=before_clip
        )
        return RetrievalResult(
            memories=memories,
            clips=[],
            metadata={'mode': 'entity_id_query'}
        )

    def _fallback_to_baseline(
        self,
        video_graph,
        query: str,
        current_clips: List,
        before_clip: Optional[int],
        reason: str
    ) -> RetrievalResult:
        """Fallback to baseline retrieval."""
        config = self.nstf_config

        fallback_threshold = min(config.threshold_baseline, 0.25)

        memories, current_clips, _ = baseline_search(
            video_graph, query, current_clips,
            threshold=fallback_threshold,
            topk=config.topk_baseline,
            before_clip=before_clip
        )

        return RetrievalResult(
            memories=memories,
            clips=current_clips,
            metadata={
                'mode': self.mode,
                'decision': 'fallback',
                'fallback_reason': reason,
                'use_baseline_prompt': True,
            }
        )

    def _load_video_graph(self, mem_path: str):
        """Load video graph via cache manager."""
        return cache.get_video_graph(mem_path, load_video_graph)

    def _load_nstf_graph(self, nstf_path: str) -> Optional[Dict]:
        """Load NSTF graph via cache manager."""
        def loader(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return cache.get_nstf_graph(nstf_path, loader)

    def _get_procedure_embeddings(self, nstf_graph: Dict) -> Dict:
        """Get procedure embeddings via cache manager."""
        graph_id = id(nstf_graph)

        cached = cache.get_embeddings(f"proc_emb_{graph_id}")
        if cached:
            return cached

        proc_nodes = nstf_graph.get('procedure_nodes', {})
        if not proc_nodes:
            return {}

        texts = []
        text_info = []

        for proc_id, proc in proc_nodes.items():
            goal = proc.get('goal', '')
            if goal and goal.strip():
                texts.append(goal)
                text_info.append((proc_id, 'goal'))

            steps = proc.get('steps', [])
            if steps:
                actions = [s.get('action', '') for s in steps
                          if isinstance(s, dict) and s.get('action')]
                combined = '. '.join(actions)
                if combined.strip():
                    texts.append(combined)
                    text_info.append((proc_id, 'steps'))

        if not texts:
            return {}

        all_embs, _ = parallel_get_embedding("text-embedding-3-large", texts)

        result = {}
        for i, emb in enumerate(all_embs):
            proc_id, emb_type = text_info[i]
            if proc_id not in result:
                result[proc_id] = {}

            vec = np.array(emb)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            result[proc_id][f'{emb_type}_emb'] = vec

        cache.set_embeddings(f"proc_emb_{graph_id}", result)
        return result

    def _search_procedures(
        self,
        query: str,
        proc_embeddings: Dict,
        alpha: float = 0.3
    ) -> List[Dict]:
        """Score = alpha*sim(goal) + (1-alpha)*sim(steps)."""
        query_embs, _ = parallel_get_embedding("text-embedding-3-large", [query])
        query_vec = np.array(query_embs[0])
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        results = []
        config = self.nstf_config

        for proc_id, embs in proc_embeddings.items():
            goal_sim = 0.0
            steps_sim = 0.0
            has_goal = 'goal_emb' in embs
            has_steps = 'steps_emb' in embs

            if has_goal:
                goal_sim = float(np.dot(query_vec, embs['goal_emb']))
            if has_steps:
                steps_sim = float(np.dot(query_vec, embs['steps_emb']))

            if has_goal and has_steps:
                combined_score = alpha * goal_sim + (1 - alpha) * steps_sim
            elif has_goal:
                combined_score = goal_sim
            elif has_steps:
                combined_score = steps_sim
            else:
                combined_score = 0.0

            if has_goal and has_steps:
                match_type = 'goal' if goal_sim >= steps_sim else 'steps'
            elif has_goal:
                match_type = 'goal_only'
            elif has_steps:
                match_type = 'steps_only'
            else:
                match_type = 'none'

            if combined_score >= config.threshold:
                results.append({
                    'proc_id': proc_id,
                    'similarity': combined_score,
                    'goal_sim': goal_sim,
                    'steps_sim': steps_sim,
                    'match_type': match_type,
                })
            elif combined_score >= config.min_confidence:
                results.append({
                    'proc_id': proc_id,
                    'similarity': combined_score,
                    'goal_sim': goal_sim,
                    'steps_sim': steps_sim,
                    'match_type': 'combined',
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _add_procedure_memories(
        self,
        memories: Dict,
        matched_procs: List[Dict],
        nstf_graph: Dict,
        config: NSTFConfig,
        video_graph = None
    ):
        """Add procedure information to memories with name resolution."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})
        lines = []

        has_real_names = False
        if self.name_resolver:
            mappings = self.name_resolver.get_all_mappings()
            has_real_names = any(
                not v.startswith(('character_', 'person_', 'face_', 'voice_'))
                for v in mappings.values()
            )

        if not has_real_names:
            lines.append("[Note: Real names are not available in this video. 'character_X' refers to distinct individuals identified by face/voice recognition. Do not ask about their real names.]")
            lines.append("")

        for i, match in enumerate(matched_procs[:config.max_procedures], 1):
            proc_id = match['proc_id']
            proc = proc_nodes.get(proc_id, {})

            lines.append(f"--- Procedure {i} (Relevance: {match['similarity']:.2f}, matched by {match['match_type']}) ---")

            goal = proc.get('goal', 'Unknown')
            if self.name_resolver:
                goal = self.name_resolver.resolve_text(goal)
            lines.append(f"Goal: {goal}")

            description = proc.get('description', '')
            if description:
                if self.name_resolver:
                    description = self.name_resolver.resolve_text(description)
                lines.append(f"Context: {description}")

            steps = proc.get('steps', [])
            for j, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    action = step.get('action', '')
                    if action:
                        if self.name_resolver:
                            action = self.name_resolver.resolve_text(action)
                        lines.append(f"Step {j}: {action}")

            episodic_links = proc.get('episodic_links', [])
            if episodic_links and video_graph:
                lines.append("")
                lines.append("=== Video Evidence (from linked episodic memories) ===")

                clips_added = 0
                max_clips = 2
                max_content_len = 800

                for link in episodic_links[:5]:
                    if clips_added >= max_clips:
                        break

                    clip_id = link.get('clip_id')
                    if clip_id is None:
                        continue

                    try:
                        clip_id = int(clip_id)
                    except (ValueError, TypeError):
                        continue

                    full_content = self._get_clip_full_content(video_graph, clip_id)
                    if full_content:
                        if len(full_content) > max_content_len:
                            full_content = full_content[:max_content_len] + "...[truncated]"

                        if self.name_resolver:
                            full_content = self.name_resolver.resolve_text(full_content)
                        sim = link.get('similarity', 0)
                        lines.append(f"[CLIP_{clip_id}] (relevance: {sim:.2f})")
                        lines.append(full_content)
                        lines.append("")
                        clips_added += 1

                if clips_added == 0:
                    for link in episodic_links[:2]:
                        preview = link.get('content_preview', '')
                        if preview:
                            if self.name_resolver:
                                preview = self.name_resolver.resolve_text(preview)
                            lines.append(f"  - {preview}")

            elif episodic_links:
                lines.append("Evidence from video:")
                for link in episodic_links[:2]:
                    preview = link.get('content_preview', '')
                    if preview:
                        if self.name_resolver:
                            preview = self.name_resolver.resolve_text(preview)
                        lines.append(f"  - {preview}")

            lines.append("")

        memories['NSTF_Procedures'] = '\n'.join(lines)

    def _get_clip_full_content(self, video_graph, clip_id: int) -> str:
        """Get full content of a single clip from video_graph."""
        if not hasattr(video_graph, 'text_nodes_by_clip'):
            return ''

        try:
            clip_id = int(clip_id)
        except (ValueError, TypeError):
            return ''

        if clip_id not in video_graph.text_nodes_by_clip:
            return ''

        node_ids = video_graph.text_nodes_by_clip[clip_id]
        contents = []

        for nid in node_ids:
            node = video_graph.nodes.get(nid)
            if node and hasattr(node, 'metadata'):
                node_contents = node.metadata.get('contents', [])
                contents.extend(str(c) for c in node_contents)

        return ' '.join(contents)

    def _extract_episodic_evidence(
        self,
        matched_procs: List[Dict],
        nstf_graph: Dict,
        video_graph
    ) -> Dict[int, str]:
        """Extract episodic evidence from matched procedures."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})
        clip_ids = set()

        for match in matched_procs:
            proc = proc_nodes.get(match['proc_id'], {})
            for link in proc.get('episodic_links', []):
                clip_id = link.get('clip_id')
                if clip_id is not None:
                    clip_ids.add(clip_id)

        evidence = {}

        if hasattr(video_graph, 'text_nodes_by_clip'):
            for clip_id in clip_ids:
                if clip_id not in video_graph.text_nodes_by_clip:
                    continue

                node_ids = video_graph.text_nodes_by_clip[clip_id]
                contents = []

                for nid in node_ids:
                    node = video_graph.nodes.get(nid)
                    if node and hasattr(node, 'metadata'):
                        node_contents = node.metadata.get('contents', [])
                        contents.extend(str(c) for c in node_contents)

                if contents:
                    evidence[clip_id] = ' '.join(contents)

        return evidence

    def _extract_character_from_query(self, query: str, video_graph) -> Optional[str]:
        """Extract character ID from query text."""
        match = re.search(r'character_\d+', query, re.IGNORECASE)
        if match:
            return match.group(0)

        match = re.search(r'(face|voice)_\d+', query, re.IGNORECASE)
        if match:
            tag = match.group(0)
            if hasattr(video_graph, 'reverse_character_mappings'):
                return video_graph.reverse_character_mappings.get(tag)

        return None

    def clear_cache(self):
        """Clear all caches."""
        cache.clear_all()

    @property
    def mode_name(self) -> str:
        """Current mode name."""
        return self.mode


def create_retriever(
    mode: str = "baseline",
    **kwargs
) -> HybridRetriever:
    """Factory function to create a HybridRetriever instance."""
    baseline_config = BaselineConfig(
        threshold=kwargs.get('threshold_baseline', 0.3),
        topk=kwargs.get('topk', 10),
    )

    nstf_config = NSTFConfig(
        threshold=kwargs.get('threshold', 0.40),
        min_confidence=kwargs.get('min_confidence', 0.35),
        max_procedures=kwargs.get('max_procedures', 3),
        topk_baseline=kwargs.get('topk', 10),
        threshold_baseline=kwargs.get('threshold_baseline', 0.3),
        use_reranking=kwargs.get('use_reranking', True),
        use_dag_paths=kwargs.get('use_dag_paths', True),
        factual_hybrid=kwargs.get('factual_hybrid', True),
    )

    return HybridRetriever(
        mode=mode,
        baseline_config=baseline_config,
        nstf_config=nstf_config,
    )
