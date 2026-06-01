import re
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from env_setup import setup_all
setup_all()

from mmagent.utils.chat_api import parallel_get_embedding
from mmagent.retrieve import search as baseline_search
from mmagent.utils.general import load_video_graph


class NSTFRetriever:
    """Procedure-based NSTF retriever with multi-granularity matching and symbolic functions."""

    def __init__(
        self,
        threshold: float = 0.35,
        min_confidence: float = 0.30,
        max_procedures: int = 3,
        topk_baseline: int = 10,
        threshold_baseline: float = 0.3,
        include_episodic_evidence: bool = True,
    ):
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.max_procedures = max_procedures
        self.topk_baseline = topk_baseline
        self.threshold_baseline = threshold_baseline
        self.include_episodic_evidence = include_episodic_evidence

        self._graph_cache: Dict[str, Any] = {}
        self._nstf_cache: Dict[str, Any] = {}
        self._embedding_cache: Dict[str, Dict] = {}

    def search(
        self,
        mem_path: str,
        query: str,
        current_clips: List = None,
        nstf_path: Optional[str] = None,
        before_clip: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List, Dict]:
        """Execute NSTF retrieval with fallback to baseline."""
        if current_clips is None:
            current_clips = []

        video_graph = self._load_video_graph(mem_path)
        if before_clip is not None:
            video_graph.truncate_memory_by_clip(before_clip, False)
        video_graph.refresh_equivalences()

        nstf_graph = self._load_nstf_graph(nstf_path) if nstf_path else None

        metadata = {
            'decision': 'unknown',
            'retriever': 'nstf_level',
        }

        if nstf_graph is None:
            memories, current_clips, _ = baseline_search(
                video_graph, query, current_clips,
                threshold=self.threshold_baseline,
                topk=self.topk_baseline,
                before_clip=before_clip
            )
            metadata['decision'] = 'fallback'
            metadata['fallback_reason'] = 'No NSTF graph available'
            return memories, current_clips, metadata

        proc_embeddings = self._get_procedure_embeddings(nstf_graph, nstf_path)

        if not proc_embeddings:
            memories, current_clips, _ = baseline_search(
                video_graph, query, current_clips,
                threshold=self.threshold_baseline,
                topk=self.topk_baseline,
                before_clip=before_clip
            )
            metadata['decision'] = 'fallback'
            metadata['fallback_reason'] = 'No procedure embeddings'
            return memories, current_clips, metadata

        matched_procs = self._search_procedures(query, proc_embeddings)

        if not matched_procs or matched_procs[0]['similarity'] < self.min_confidence:
            memories, current_clips, _ = baseline_search(
                video_graph, query, current_clips,
                threshold=self.threshold_baseline,
                topk=self.topk_baseline,
                before_clip=before_clip
            )
            metadata['decision'] = 'fallback'
            if not matched_procs:
                metadata['fallback_reason'] = 'No procedures above threshold'
            else:
                metadata['fallback_reason'] = f'Top sim {matched_procs[0]["similarity"]:.3f} < min_confidence {self.min_confidence}'
            return memories, current_clips, metadata

        metadata['decision'] = 'use_nstf'
        metadata['matched_procedures'] = [
            {'proc_id': p['proc_id'], 'similarity': p['similarity'], 'match_type': p['match_type']}
            for p in matched_procs[:self.max_procedures]
        ]

        query_type = self._classify_query(query)
        metadata['query_type'] = query_type

        memories = {}

        if query_type == 'temporal':
            step_result = self.query_step_sequence(
                matched_procs[0]['proc_id'],
                nstf_graph,
                query
            )
            memories['NSTF_StepQuery'] = self._format_step_query_result(step_result)
            metadata['symbolic_function'] = 'query_step_sequence'

        elif query_type == 'character':
            character_id = self._extract_character_from_query(query, video_graph)
            if character_id:
                char_result = self.aggregate_character_behaviors(
                    character_id,
                    nstf_graph,
                    video_graph
                )
                memories['NSTF_CharacterAnalysis'] = self._format_character_result(char_result)
                metadata['symbolic_function'] = 'aggregate_character_behaviors'
                metadata['character_id'] = character_id
            else:
                proc_info = self._format_procedures_for_prompt(
                    matched_procs[:self.max_procedures],
                    nstf_graph
                )
                memories['NSTF_Procedures'] = proc_info
                metadata['symbolic_function'] = 'get_procedure_with_evidence'
        else:
            proc_info = self._format_procedures_for_prompt(
                matched_procs[:self.max_procedures],
                nstf_graph
            )
            memories['NSTF_Procedures'] = proc_info
            metadata['symbolic_function'] = 'get_procedure_with_evidence'

        if self.include_episodic_evidence:
            evidence_clips = self._extract_episodic_evidence(
                matched_procs[:self.max_procedures],
                nstf_graph,
                video_graph
            )
            for clip_id, clip_content in evidence_clips.items():
                memories[f'clip_{clip_id}'] = clip_content
                if clip_id not in current_clips:
                    current_clips.append(clip_id)

        metadata['num_evidence_clips'] = len(memories) - 1

        return memories, current_clips, metadata

    def query_step_sequence(
        self,
        proc_id: str,
        nstf_graph: Dict,
        query: str
    ) -> Dict[str, Any]:
        """Temporal/step sequence query over a procedure."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})
        proc = proc_nodes.get(proc_id, {})

        steps = proc.get('steps', [])
        step_actions = []
        for s in steps:
            if isinstance(s, dict):
                action = s.get('action', '')
                if action:
                    step_actions.append(action)

        result = {
            'proc_id': proc_id,
            'goal': proc.get('goal', 'Unknown'),
            'total_steps': len(step_actions),
            'full_sequence': step_actions,
            'query_type': 'all',
            'result': None,
        }

        if not step_actions:
            result['result'] = 'No steps found'
            return result

        query_lower = query.lower()

        if any(kw in query_lower for kw in ['how many', 'count']):
            result['query_type'] = 'count'
            result['result'] = f'{len(step_actions)} steps'

        elif any(kw in query_lower for kw in ['first', 'begin', 'start']):
            result['query_type'] = 'first'
            result['result'] = step_actions[0]

        elif any(kw in query_lower for kw in ['last', 'final', 'end']):
            result['query_type'] = 'last'
            result['result'] = step_actions[-1]

        elif any(kw in query_lower for kw in ['after', 'then', 'next']):
            result['query_type'] = 'after'
            ref_action = self._find_reference_action(query, step_actions)
            if ref_action and ref_action['index'] < len(step_actions) - 1:
                result['result'] = step_actions[ref_action['index'] + 1]
                result['reference'] = ref_action['action']
            else:
                result['result'] = step_actions[-1] if step_actions else 'Unknown'

        elif any(kw in query_lower for kw in ['before', 'previous']):
            result['query_type'] = 'before'
            ref_action = self._find_reference_action(query, step_actions)
            if ref_action and ref_action['index'] > 0:
                result['result'] = step_actions[ref_action['index'] - 1]
                result['reference'] = ref_action['action']
            else:
                result['result'] = step_actions[0] if step_actions else 'Unknown'
        else:
            result['query_type'] = 'all'
            result['result'] = ' -> '.join(step_actions)

        return result

    def aggregate_character_behaviors(
        self,
        character_id: str,
        nstf_graph: Dict,
        video_graph
    ) -> Dict[str, Any]:
        """Aggregate character behavior patterns across procedures."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})

        involved_procs = []
        evidence_clips = set()

        for proc_id, proc in proc_nodes.items():
            episodic_links = proc.get('episodic_links', [])

            proc_involves_character = False
            proc_clips = []

            for link in episodic_links:
                clip_id = link.get('clip_id')
                if clip_id is None:
                    continue

                clip_content = self._get_clip_content(video_graph, clip_id)
                if character_id in clip_content:
                    proc_involves_character = True
                    proc_clips.append(clip_id)

            if proc_involves_character:
                involved_procs.append({
                    'proc_id': proc_id,
                    'goal': proc.get('goal', 'Unknown'),
                    'proc_type': proc.get('proc_type', 'task'),
                    'clips': proc_clips,
                })
                evidence_clips.update(proc_clips)

        if involved_procs:
            proc_types = [p['proc_type'] for p in involved_procs]
            goals = [p['goal'] for p in involved_procs]

            type_counts = {}
            for t in proc_types:
                type_counts[t] = type_counts.get(t, 0) + 1

            summary_parts = [f"{character_id} is involved in {len(involved_procs)} procedure(s):"]
            for proc in involved_procs:
                summary_parts.append(f"  - {proc['goal']} ({proc['proc_type']})")

            if len(involved_procs) >= 2:
                common_words = self._find_common_themes(goals)
                if common_words:
                    summary_parts.append(f"Behavior pattern: Frequently involved in {', '.join(common_words)}-related activities.")
        else:
            summary_parts = [f"No procedure information found for {character_id}."]

        return {
            'character': character_id,
            'involved_procedures': involved_procs,
            'behavior_summary': '\n'.join(summary_parts),
            'evidence_clips': sorted(evidence_clips),
        }

    def _classify_query(self, query: str) -> str:
        """Classify query type: temporal, character, or procedure."""
        query_lower = query.lower()

        temporal_keywords = [
            'after', 'before', 'then', 'next', 'first', 'last', 'step',
            'how many steps', 'sequence', 'order',
        ]
        if any(kw in query_lower for kw in temporal_keywords):
            return 'temporal'

        character_keywords = [
            'familiar', 'usually', 'habit', 'often', 'good at', 'skilled',
            'frequently', 'tend to', 'pattern', 'behavior',
        ]
        if any(kw in query_lower for kw in character_keywords):
            return 'character'

        return 'procedure'

    def _extract_character_from_query(self, query: str, video_graph) -> Optional[str]:
        """Extract character ID from query text."""
        match = re.search(r'character_(\d+)', query.lower())
        if match:
            return f'character_{match.group(1)}'

        if hasattr(video_graph, 'character_name_to_id'):
            for name, char_id in video_graph.character_name_to_id.items():
                if name.lower() in query.lower():
                    return char_id

        return None

    def _find_reference_action(self, query: str, step_actions: List[str]) -> Optional[Dict]:
        """Find reference action mentioned in query."""
        query_lower = query.lower()

        for i, action in enumerate(step_actions):
            action_words = action.lower().split()
            for word in action_words:
                if len(word) > 3 and word in query_lower:
                    return {'action': action, 'index': i}

        return None

    def _find_common_themes(self, goals: List[str]) -> List[str]:
        """Find common theme words across a list of goal descriptions."""
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'with', 'and'}

        for goal in goals:
            words = goal.lower().split()
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

        common = [w for w, c in word_counts.items() if c >= 2]
        return common[:3]

    def _get_clip_content(self, video_graph, clip_id: int) -> str:
        """Get text content of a clip."""
        if not hasattr(video_graph, 'text_nodes_by_clip'):
            return ''

        if clip_id not in video_graph.text_nodes_by_clip:
            return ''

        node_ids = video_graph.text_nodes_by_clip[clip_id]
        contents = []

        for nid in node_ids:
            node = video_graph.nodes.get(nid)
            if node and hasattr(node, 'metadata'):
                node_contents = node.metadata.get('contents', [])
                contents.extend(node_contents)

        return ' '.join(contents)

    def _format_step_query_result(self, result: Dict) -> str:
        """Format temporal query result for prompt."""
        lines = [
            "[Step Sequence Query]",
            f"Procedure: {result['goal']}",
            f"Total Steps: {result['total_steps']}",
            f"Query Type: {result['query_type']}",
        ]

        if result.get('reference'):
            lines.append(f"Reference: {result['reference']}")

        lines.append(f"Answer: {result['result']}")

        if result['total_steps'] > 0:
            sequence = ' -> '.join(result['full_sequence'])
            lines.append(f"Full Sequence: {sequence}")

        return '\n'.join(lines)

    def _format_character_result(self, result: Dict) -> str:
        """Format character aggregation result for prompt."""
        lines = [
            "[Character Behavior Analysis]",
            f"Character: {result['character']}",
            "",
            result['behavior_summary'],
        ]

        if result['evidence_clips']:
            lines.append(f"\nEvidence from clips: {result['evidence_clips']}")

        return '\n'.join(lines)

    def _search_procedures(
        self,
        query: str,
        proc_embeddings: Dict,
        alpha: float = 0.3
    ) -> List[Dict]:
        """Multi-granularity procedure retrieval: score = alpha*sim(goal) + (1-alpha)*sim(steps)."""
        query_embs, _ = parallel_get_embedding("text-embedding-3-large", [query])
        query_vec = np.array(query_embs[0])
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        results = []
        for proc_id, emb_dict in proc_embeddings.items():
            goal_vec = emb_dict.get('goal_emb')
            step_vec = emb_dict.get('step_emb')

            if step_vec is None:
                step_vec = emb_dict.get('steps_emb')

            if goal_vec is None or step_vec is None:
                continue

            sim_goal = float(np.dot(query_vec, goal_vec))
            sim_step = float(np.dot(query_vec, step_vec))

            combined_sim = alpha * sim_goal + (1 - alpha) * sim_step

            if combined_sim >= self.threshold:
                results.append({
                    'proc_id': proc_id,
                    'similarity': combined_sim,
                    'sim_goal': sim_goal,
                    'sim_step': sim_step,
                    'match_type': 'combined',
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _format_procedures_for_prompt(
        self,
        matched_procs: List[Dict],
        nstf_graph: Dict
    ) -> str:
        """Format matched procedures for LLM prompt."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})
        lines = []

        for i, match in enumerate(matched_procs, 1):
            proc_id = match['proc_id']
            proc = proc_nodes.get(proc_id, {})

            lines.append(f"--- Procedure {i} (Relevance: {match['similarity']:.2f}, matched by {match['match_type']}) ---")
            lines.append(f"Goal: {proc.get('goal', 'Unknown')}")

            steps = proc.get('steps', [])
            for j, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    action = step.get('action', '')
                    if action:
                        lines.append(f"Step {j}: {action}")

            lines.append("")

        return '\n'.join(lines)

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
                clip_contents = []

                for nid in node_ids:
                    node = video_graph.nodes.get(nid)
                    if node and hasattr(node, 'metadata'):
                        contents = node.metadata.get('contents', [])
                        clip_contents.extend(contents)

                if clip_contents:
                    evidence[clip_id] = ' '.join(clip_contents)

        if not evidence and hasattr(video_graph, 'get_node_content'):
            for clip_id in clip_ids:
                try:
                    content = video_graph.get_node_content(f'clip_{clip_id}')
                    if content:
                        evidence[clip_id] = content
                except:
                    pass

        return evidence

    def _load_video_graph(self, mem_path: str):
        """Load video graph with caching."""
        if mem_path not in self._graph_cache:
            self._graph_cache[mem_path] = load_video_graph(mem_path)
        return self._graph_cache[mem_path]

    def _load_nstf_graph(self, nstf_path: str) -> Optional[Dict]:
        """Load NSTF graph with caching."""
        if nstf_path is None:
            return None

        if nstf_path not in self._nstf_cache:
            try:
                with open(nstf_path, 'rb') as f:
                    self._nstf_cache[nstf_path] = pickle.load(f)
            except Exception:
                self._nstf_cache[nstf_path] = None

        return self._nstf_cache[nstf_path]

    def _get_procedure_embeddings(self, nstf_graph: Dict, nstf_path: str) -> Dict:
        """Compute and cache procedure embeddings."""
        if nstf_path in self._embedding_cache:
            return self._embedding_cache[nstf_path]

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
                actions = [s.get('action', '') for s in steps if isinstance(s, dict) and s.get('action')]
                combined = '. '.join(actions)
                if combined.strip():
                    texts.append(combined)
                    text_info.append((proc_id, 'step'))

        if not texts:
            self._embedding_cache[nstf_path] = {}
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

        self._embedding_cache[nstf_path] = result
        return result

    def clear_cache(self):
        """Clear all caches."""
        self._graph_cache.clear()
        self._nstf_cache.clear()
        self._embedding_cache.clear()

    @property
    def mode_name(self) -> str:
        """Mode name identifier."""
        return "NSTF-Level"
