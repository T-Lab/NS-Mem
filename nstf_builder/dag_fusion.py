import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from .utils import get_normalized_embedding, batch_get_normalized_embeddings, cosine_similarity


class DAGFusion:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        ema_alpha: float = 0.3,
        min_probability: float = 0.01,
        debug: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.min_probability = min_probability
        self.debug = debug

    def should_fuse(self, proc1: Dict, proc2: Dict, goal_threshold: float = 0.80) -> bool:
        """Determine whether two procedures should be fused by goal similarity."""
        emb1 = self._get_goal_embedding(proc1)
        emb2 = self._get_goal_embedding(proc2)

        if emb1 is None or emb2 is None:
            goal1 = proc1.get('goal', '')
            goal2 = proc2.get('goal', '')
            return goal1.lower().strip() == goal2.lower().strip()

        similarity = cosine_similarity(emb1, emb2)
        return similarity >= goal_threshold

    def _get_goal_embedding(self, proc: Dict) -> Optional[np.ndarray]:
        """Get the goal embedding of a procedure."""
        embeddings = proc.get('embeddings', {})
        if 'goal_emb' in embeddings:
            emb = embeddings['goal_emb']
            if isinstance(emb, np.ndarray):
                return emb
            return np.array(emb)

        goal = proc.get('goal', '')
        if goal:
            return get_normalized_embedding(goal)
        return None

    def fuse(self, proc1: Dict, proc2: Dict, observation_counts: Dict[str, int] = None) -> Dict:
        """Fuse two procedures via node alignment, parameter pooling, and edge union."""
        if observation_counts is None:
            observation_counts = {}

        alignment = self._align_steps(
            proc1.get('steps', []),
            proc2.get('steps', [])
        )

        merged_steps, step_id_mapping = self._merge_steps(
            proc1.get('steps', []),
            proc2.get('steps', []),
            alignment
        )

        merged_edges = self._merge_edges(
            proc1.get('edges', []),
            proc2.get('edges', []),
            step_id_mapping,
            observation_counts
        )

        merged_links = self._merge_episodic_links(
            proc1.get('episodic_links', []),
            proc2.get('episodic_links', [])
        )

        merged_embedding = self._merge_embeddings(proc1, proc2)
        merged_step_emb = self._merge_step_embeddings(proc1, proc2, merged_steps)
        merged_metadata = self._merge_metadata(proc1, proc2)
        merged_dag = self._construct_dag_from_steps_and_edges(merged_steps, merged_edges)
        merged_proc_type = proc1.get('proc_type') or proc2.get('proc_type') or 'task'

        fused_proc = {
            'proc_id': proc1.get('proc_id', 'fused_proc'),
            'type': 'procedure',
            'proc_type': merged_proc_type,
            'goal': self._merge_goals(proc1.get('goal', ''), proc2.get('goal', '')),
            'description': self._merge_descriptions(
                proc1.get('description', ''),
                proc2.get('description', '')
            ),
            'steps': merged_steps,
            'edges': merged_edges,
            'dag': merged_dag,
            'episodic_links': merged_links,
            'embeddings': {
                'goal_emb': merged_embedding,
                'step_emb': merged_step_emb,
            },
            'metadata': merged_metadata,
            'fusion_info': {
                'source_procs': [proc1.get('proc_id'), proc2.get('proc_id')],
                'num_matched_steps': len(alignment['matched']),
                'num_new_steps': len(alignment['only_proc2']),
                'total_steps': len(merged_steps),
                'total_edges': len(merged_edges),
            }
        }

        return fused_proc

    def _align_steps(
        self,
        steps1: List[Dict],
        steps2: List[Dict]
    ) -> Dict[str, List]:
        """Align steps via Hungarian algorithm on embedding similarity."""
        if not steps1 or not steps2:
            return {
                'matched': [],
                'only_proc1': list(range(len(steps1))),
                'only_proc2': list(range(len(steps2))),
            }

        actions1 = [self._get_step_action(s) for s in steps1]
        actions2 = [self._get_step_action(s) for s in steps2]

        all_actions = actions1 + actions2
        all_embeddings = batch_get_normalized_embeddings(all_actions)

        emb1 = all_embeddings[:len(actions1)]
        emb2 = all_embeddings[len(actions1):]

        n1, n2 = len(steps1), len(steps2)
        similarity_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                similarity_matrix[i, j] = cosine_similarity(emb1[i], emb2[j])

        cost_matrix = 1.0 - similarity_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        matched_i = set()
        matched_j = set()

        for i, j in zip(row_ind, col_ind):
            sim = similarity_matrix[i, j]
            if sim >= self.similarity_threshold:
                matched.append((i, j, sim))
                matched_i.add(i)
                matched_j.add(j)

        only_proc1 = [i for i in range(n1) if i not in matched_i]
        only_proc2 = [j for j in range(n2) if j not in matched_j]

        return {
            'matched': matched,
            'only_proc1': only_proc1,
            'only_proc2': only_proc2,
        }

    def _get_step_action(self, step: Dict) -> str:
        """Get the action text of a step."""
        if isinstance(step, dict):
            action = step.get('action', '')
            if isinstance(action, str):
                return action
            return str(action)
        return str(step)

    def _merge_steps(
        self,
        steps1: List[Dict],
        steps2: List[Dict],
        alignment: Dict
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Merge step lists based on alignment results."""
        merged_steps = []
        step_id_mapping = {}

        for idx1, idx2, sim in alignment['matched']:
            s1 = steps1[idx1] if idx1 < len(steps1) else {}
            s2 = steps2[idx2] if idx2 < len(steps2) else {}

            merged_step = self._merge_single_step(s1, s2, sim)
            new_step_id = merged_step.get('step_id', f'step_{len(merged_steps)+1}')

            old_id1 = s1.get('step_id', f'step_{idx1+1}')
            old_id2 = s2.get('step_id', f'step_{idx2+1}')
            step_id_mapping[f'proc1_{old_id1}'] = new_step_id
            step_id_mapping[f'proc2_{old_id2}'] = new_step_id

            merged_steps.append(merged_step)

        for idx in alignment['only_proc1']:
            if idx < len(steps1):
                step = steps1[idx].copy() if isinstance(steps1[idx], dict) else {'action': str(steps1[idx])}
                old_id = step.get('step_id', f'step_{idx+1}')
                new_id = f'step_{len(merged_steps)+1}'
                step['step_id'] = new_id
                step_id_mapping[f'proc1_{old_id}'] = new_id
                merged_steps.append(step)

        for idx in alignment['only_proc2']:
            if idx < len(steps2):
                step = steps2[idx].copy() if isinstance(steps2[idx], dict) else {'action': str(steps2[idx])}
                old_id = step.get('step_id', f'step_{idx+1}')
                new_id = f'step_{len(merged_steps)+1}'
                step['step_id'] = new_id
                step_id_mapping[f'proc2_{old_id}'] = new_id
                merged_steps.append(step)

        return merged_steps, step_id_mapping

    def _merge_single_step(self, s1: Dict, s2: Dict, similarity: float) -> Dict:
        """Merge attributes of two matched steps via parameter pooling."""
        action1 = s1.get('action', '')
        action2 = s2.get('action', '')
        merged_action = action1 if len(action1) >= len(action2) else action2

        dur1 = s1.get('duration_seconds', 30)
        dur2 = s2.get('duration_seconds', 30)
        merged_duration = (dur1 + dur2) / 2

        sr1 = s1.get('success_rate', 0.9)
        sr2 = s2.get('success_rate', 0.9)
        merged_success_rate = (sr1 + sr2) / 2

        triggers1 = s1.get('triggers', [])
        triggers2 = s2.get('triggers', [])
        merged_triggers = list(set(triggers1) | set(triggers2))

        outcomes1 = s1.get('outcomes', [])
        outcomes2 = s2.get('outcomes', [])
        merged_outcomes = list(set(outcomes1) | set(outcomes2))

        return {
            'step_id': s1.get('step_id', 'step_1'),
            'action': merged_action,
            'triggers': merged_triggers,
            'outcomes': merged_outcomes,
            'duration_seconds': merged_duration,
            'success_rate': merged_success_rate,
            'merge_similarity': similarity,
        }

    def _merge_edges(
        self,
        edges1: List[Dict],
        edges2: List[Dict],
        step_id_mapping: Dict[str, str],
        observation_counts: Dict[str, int]
    ) -> List[Dict]:
        """Merge edges with Bayesian probability updates."""
        edge_dict = {}

        for edge in edges1:
            from_step = edge.get('from_step', '')
            to_step = edge.get('to_step', '')

            new_from = step_id_mapping.get(f'proc1_{from_step}', from_step)
            new_to = step_id_mapping.get(f'proc1_{to_step}', to_step)

            key = (new_from, new_to)
            if key not in edge_dict:
                edge_dict[key] = {
                    'from_step': new_from,
                    'to_step': new_to,
                    'probabilities': [],
                    'conditions': set(),
                    'observation_count': 0,
                }

            prob = edge.get('probability', 1.0)
            edge_dict[key]['probabilities'].append(prob)
            edge_dict[key]['observation_count'] += observation_counts.get(f'proc1_{from_step}_{to_step}', 1)

            cond = edge.get('condition', '')
            if cond:
                edge_dict[key]['conditions'].add(cond)

        for edge in edges2:
            from_step = edge.get('from_step', '')
            to_step = edge.get('to_step', '')

            new_from = step_id_mapping.get(f'proc2_{from_step}', from_step)
            new_to = step_id_mapping.get(f'proc2_{to_step}', to_step)

            key = (new_from, new_to)
            if key not in edge_dict:
                edge_dict[key] = {
                    'from_step': new_from,
                    'to_step': new_to,
                    'probabilities': [],
                    'conditions': set(),
                    'observation_count': 0,
                }

            prob = edge.get('probability', 1.0)
            edge_dict[key]['probabilities'].append(prob)
            edge_dict[key]['observation_count'] += observation_counts.get(f'proc2_{from_step}_{to_step}', 1)

            cond = edge.get('condition', '')
            if cond:
                edge_dict[key]['conditions'].add(cond)

        merged_edges = []
        for key, info in edge_dict.items():
            probs = info['probabilities']
            obs_count = max(info['observation_count'], 1)

            pooled_prob = self._bayesian_pool_probability(probs, obs_count)
            final_prob = max(self.min_probability, min(1.0, pooled_prob))

            conditions = list(info['conditions'])
            condition_str = ' OR '.join(conditions) if conditions else ''

            merged_edges.append({
                'from_step': info['from_step'],
                'to_step': info['to_step'],
                'probability': final_prob,
                'condition': condition_str,
                'observation_count': obs_count,
            })

        return merged_edges

    def _bayesian_pool_probability(
        self,
        observed_probs: List[float],
        observation_count: int,
    ) -> float:
        """Posterior mean of an edge transition probability under a Beta(1, 1) prior.

        Each fused observation contributes its empirical success rate; we pool by
        accumulating successes/failures and reading off the Beta posterior mean,
        which is the per-edge instance of the Beta-Binomial conjugacy used in
        Theorem 2."""
        if not observed_probs:
            return 1.0
        n = max(int(observation_count), 1)
        avg_p = sum(observed_probs) / len(observed_probs)
        avg_p = min(1.0, max(0.0, avg_p))
        successes = avg_p * n
        failures = n - successes
        alpha_post = 1.0 + successes
        beta_post = 1.0 + failures
        return alpha_post / (alpha_post + beta_post)

    def _merge_episodic_links(
        self,
        links1: List[Dict],
        links2: List[Dict]
    ) -> List[Dict]:
        """Merge and deduplicate episodic links."""
        seen_clips = set()
        merged = []

        for link in links1 + links2:
            clip_id = link.get('clip_id')
            if clip_id is not None and clip_id not in seen_clips:
                seen_clips.add(clip_id)
                merged.append(link)

        merged.sort(key=lambda x: x.get('clip_id', 0))
        return merged

    def _merge_embeddings(self, proc1: Dict, proc2: Dict) -> np.ndarray:
        """Merge goal embeddings using EMA-weighted pooling."""
        emb1 = self._get_goal_embedding(proc1)
        emb2 = self._get_goal_embedding(proc2)

        if emb1 is None and emb2 is None:
            goal = proc1.get('goal', '') or proc2.get('goal', '')
            return get_normalized_embedding(goal) if goal else np.zeros(3072)

        if emb1 is None:
            return emb2
        if emb2 is None:
            return emb1

        merged = self.ema_alpha * emb2 + (1 - self.ema_alpha) * emb1
        merged = merged / (np.linalg.norm(merged) + 1e-8)

        return merged

    def _merge_step_embeddings(self, proc1: Dict, proc2: Dict, merged_steps: List[Dict]) -> np.ndarray:
        """Merge step embeddings via EMA or recompute from merged steps."""
        emb1 = proc1.get('embeddings', {}).get('step_emb')
        emb2 = proc2.get('embeddings', {}).get('step_emb')

        if emb1 is not None and emb2 is not None:
            if isinstance(emb1, list):
                emb1 = np.array(emb1)
            if isinstance(emb2, list):
                emb2 = np.array(emb2)
            merged = self.ema_alpha * emb2 + (1 - self.ema_alpha) * emb1
            return merged / (np.linalg.norm(merged) + 1e-8)

        if emb1 is not None:
            return np.array(emb1) if isinstance(emb1, list) else emb1
        if emb2 is not None:
            return np.array(emb2) if isinstance(emb2, list) else emb2

        if merged_steps:
            actions = [s.get('action', '') for s in merged_steps if isinstance(s, dict) and s.get('action')]
            if actions:
                embs = batch_get_normalized_embeddings(actions)
                merged = np.mean(embs, axis=0)
                return merged / (np.linalg.norm(merged) + 1e-8)

        return np.zeros(3072)

    def _construct_dag_from_steps_and_edges(self, steps: List[Dict], edges: List[Dict]) -> Dict:
        """Build complete DAG structure from merged steps and edges."""
        nodes = {
            'START': {'type': 'control', 'attributes': {}},
            'GOAL': {'type': 'control', 'attributes': {}},
        }

        for s in steps:
            if isinstance(s, dict):
                step_id = s.get('step_id', '')
                if step_id:
                    nodes[step_id] = {
                        'type': 'action',
                        'action': s.get('action', ''),
                        'attributes': {
                            'object': s.get('object', ''),
                            'location': s.get('location', ''),
                            'actor': s.get('actor', ''),
                            'triggers': s.get('triggers', []),
                            'outcomes': s.get('outcomes', []),
                        }
                    }

        dag_edges = []
        for e in edges:
            from_step = e.get('from_step') or e.get('from', '')
            to_step = e.get('to_step') or e.get('to', '')

            dag_edges.append({
                'from': from_step,
                'to': to_step,
                'count': e.get('observation_count', e.get('count', 1)),
                'probability': e.get('probability', 1.0),
            })

        return {'nodes': nodes, 'edges': dag_edges}

    def _merge_goals(self, goal1: str, goal2: str) -> str:
        """Keep the longer goal description."""
        if not goal1:
            return goal2
        if not goal2:
            return goal1
        return goal1 if len(goal1) >= len(goal2) else goal2

    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        """Merge two descriptions."""
        if not desc1:
            return desc2
        if not desc2:
            return desc1
        if desc1.lower().strip() == desc2.lower().strip():
            return desc1
        return f"{desc1} | {desc2}"

    def _merge_metadata(self, proc1: Dict, proc2: Dict) -> Dict:
        """Merge metadata from two procedures."""
        meta1 = proc1.get('metadata', {})
        meta2 = proc2.get('metadata', {})

        from datetime import datetime

        return {
            'created_at': meta1.get('created_at', datetime.now().isoformat()),
            'last_updated': datetime.now().isoformat(),
            'source': 'nstf_fusion',
            'original_sources': [
                meta1.get('source', 'unknown'),
                meta2.get('source', 'unknown'),
            ],
            'fusion_count': meta1.get('fusion_count', 0) + 1,
        }


class ProcedureFusionManager:
    """Manage greedy pairwise fusion of similar procedures."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        step_align_threshold: float = 0.70,
        debug: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.dag_fusion = DAGFusion(
            similarity_threshold=step_align_threshold,
            debug=debug,
        )
        self.debug = debug

        self.stats = {
            'procedures_before': 0,
            'procedures_after': 0,
            'total_fusions': 0,
        }

    def fuse_all(self, procedure_nodes: Dict[str, Dict]) -> Dict[str, Dict]:
        """Fuse all similar procedures using greedy highest-similarity-first strategy."""
        if not procedure_nodes or len(procedure_nodes) < 2:
            return procedure_nodes

        self.stats['procedures_before'] = len(procedure_nodes)

        procs = list(procedure_nodes.values())
        proc_ids = list(procedure_nodes.keys())

        goals = [p.get('goal', '') for p in procs]
        goal_embeddings = batch_get_normalized_embeddings(goals)

        n = len(procs)
        similarity_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(goal_embeddings[i], goal_embeddings[j])
                if sim >= self.similarity_threshold:
                    similarity_pairs.append((i, j, sim))

        similarity_pairs.sort(key=lambda x: -x[2])

        fused_indices = set()
        fused_procs = []

        for i, j, sim in similarity_pairs:
            if i in fused_indices or j in fused_indices:
                continue

            fused = self.dag_fusion.fuse(procs[i], procs[j])
            fused_procs.append(fused)
            fused_indices.add(i)
            fused_indices.add(j)
            self.stats['total_fusions'] += 1

        for i, proc in enumerate(procs):
            if i not in fused_indices:
                fused_procs.append(proc)

        result = {}
        for i, proc in enumerate(fused_procs):
            proc_id = proc.get('proc_id', f'proc_{i+1}')
            result[proc_id] = proc

        self.stats['procedures_after'] = len(result)

        return result

    def incremental_fuse(
        self,
        existing_procs: Dict[str, Dict],
        new_proc: Dict
    ) -> Tuple[Dict[str, Dict], bool]:
        """Incrementally fuse a new procedure into existing set."""
        if not existing_procs:
            new_id = new_proc.get('proc_id', 'proc_1')
            return {new_id: new_proc}, False

        new_goal = new_proc.get('goal', '')
        new_emb = get_normalized_embedding(new_goal) if new_goal else None

        if new_emb is None:
            new_id = new_proc.get('proc_id', f'proc_{len(existing_procs)+1}')
            result = existing_procs.copy()
            result[new_id] = new_proc
            return result, False

        best_match_id = None
        best_match_sim = 0.0

        for proc_id, proc in existing_procs.items():
            proc_emb = self.dag_fusion._get_goal_embedding(proc)
            if proc_emb is not None:
                sim = cosine_similarity(new_emb, proc_emb)
                if sim > best_match_sim:
                    best_match_sim = sim
                    best_match_id = proc_id

        result = existing_procs.copy()

        if best_match_sim >= self.similarity_threshold and best_match_id:
            fused = self.dag_fusion.fuse(existing_procs[best_match_id], new_proc)
            fused['proc_id'] = best_match_id
            result[best_match_id] = fused
            self.stats['total_fusions'] += 1
            return result, True
        else:
            new_id = new_proc.get('proc_id', f'proc_{len(existing_procs)+1}')
            result[new_id] = new_proc
            return result, False

    def get_stats(self) -> Dict:
        """Get fusion statistics."""
        return self.stats.copy()
