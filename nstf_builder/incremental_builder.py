import os
import sys
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

from env_setup import setup_all, NSTF_MODEL_DIR
setup_all()

from .extractor import ProcedureExtractor
from .character_resolver import CharacterResolver
from .episodic_linker import EpisodicLinker
from .procedure_matcher import ProcedureMatcher
from .dag_fusion import DAGFusion, ProcedureFusionManager
from .utils import get_normalized_embedding, batch_get_normalized_embeddings, cosine_similarity


class IncrementalNSTFBuilder:
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = None,
        config_path: str = None,
        debug: bool = False,
    ):
        self.module_dir = Path(__file__).parent
        self.nstf_model_dir = self.module_dir.parent

        self.data_dir = Path(data_dir) if data_dir else self.nstf_model_dir / 'data'
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'nstf_graphs'

        self.debug = debug

        self.config = self._load_config(config_path)

        self.extractor = ProcedureExtractor(
            llm_model=self.config.get('llm_model', 'gemini-2.5-flash'),
            batch_size=1,
            max_content_chars=self.config.get('max_content_chars', 150),
            api_delay=self.config.get('api_delay_seconds', 1),
        )

        self.matcher = ProcedureMatcher(
            match_threshold=self.config.get('match_threshold', 0.70),
            debug=debug,
        )

        self.linker = EpisodicLinker(
            verify_threshold=self.config.get('verify_threshold', 0.35),
            discover_threshold=self.config.get('discover_threshold', 0.35),
            max_links_per_proc=self.config.get('max_links_per_proc', 10),
            debug=debug,
        )

        self.fusion_manager = ProcedureFusionManager(
            similarity_threshold=self.config.get('fusion_similarity_threshold', 0.80),
            step_align_threshold=self.config.get('step_align_threshold', 0.75),
            debug=debug,
        )

        self.character_resolver: Optional[CharacterResolver] = None

        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-large')
        self._embedding_api = None

        self.stats = {
            'videos_processed': 0,
            'clips_processed': 0,
            'procedures_created': 0,
            'procedures_merged': 0,
        }

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = self.module_dir / 'config' / 'default.json'

        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def embedding_api(self):
        if self._embedding_api is None:
            from mmagent.utils.chat_api import get_embedding_with_retry
            self._embedding_api = get_embedding_with_retry
        return self._embedding_api

    def load_baseline_graph(self, video_name: str, dataset: str):
        """Load baseline memory graph from disk."""
        graph_path = self.data_dir / 'memory_graphs' / dataset / f'{video_name}.pkl'
        if not graph_path.exists():
            return None
        with open(graph_path, 'rb') as f:
            return pickle.load(f)

    def get_sorted_clips(self, graph) -> List[int]:
        """Get temporally sorted clip IDs."""
        if hasattr(graph, 'text_nodes_by_clip'):
            return sorted(graph.text_nodes_by_clip.keys())
        return []

    def get_clip_content(self, graph, clip_id: int) -> Dict:
        """Get resolved content for a single clip."""
        if not hasattr(graph, 'text_nodes_by_clip'):
            return {'clip_id': clip_id, 'content': '', 'raw_content': ''}

        node_ids = graph.text_nodes_by_clip.get(clip_id, [])
        clip_texts = []

        for nid in node_ids:
            node = graph.nodes.get(nid)
            if node and hasattr(node, 'metadata'):
                contents = node.metadata.get('contents', [])
                clip_texts.extend(contents)

        combined = ' '.join(clip_texts)

        if self.character_resolver:
            resolved = self.character_resolver.resolve(combined)
        else:
            resolved = combined

        return {
            'clip_id': clip_id,
            'content': resolved,
            'raw_content': combined
        }

    def create_procedure_node(
        self,
        detected: Dict,
        clip_id: int,
        clip_content: Dict,
        proc_id: str
    ) -> Dict:
        """Create a new procedure node with dual-layer index vectors and DAG."""
        import numpy as np

        goal = detected.get('goal', '')
        if isinstance(goal, dict):
            goal = goal.get('name', str(goal))
        goal = str(goal)

        description = detected.get('description', '')

        raw_proc_type = detected.get('type', 'task')
        VALID_PROC_TYPES = {'task', 'habit', 'trait', 'social'}
        if isinstance(raw_proc_type, str) and '|' in raw_proc_type:
            parts = raw_proc_type.split('|')
            proc_type = next((p.strip() for p in parts if p.strip() in VALID_PROC_TYPES), 'task')
        elif raw_proc_type in VALID_PROC_TYPES:
            proc_type = raw_proc_type
        else:
            proc_type = 'task'

        raw_steps = detected.get('steps', [])
        edges = detected.get('edges', [])

        steps = []
        for i, s in enumerate(raw_steps):
            if isinstance(s, dict):
                step = {
                    'step_id': s.get('step_id', f'step_{i+1}'),
                    'action': s.get('action', ''),
                    'object': s.get('object', ''),
                    'location': s.get('location', ''),
                    'actor': s.get('actor', ''),
                    'triggers': s.get('triggers', []),
                    'outcomes': s.get('outcomes', []),
                    'duration_seconds': s.get('duration_seconds', 0)
                }
                steps.append(step)
            elif isinstance(s, str) and s:
                steps.append({
                    'step_id': f'step_{i+1}',
                    'action': s,
                    'object': '',
                    'location': '',
                    'actor': '',
                    'triggers': [],
                    'outcomes': [],
                    'duration_seconds': 0
                })

        objects = detected.get('objects', detected.get('key_objects', []))
        locations = detected.get('locations', detected.get('key_locations', []))
        participants = detected.get('participants', [])

        goal_text = f"{goal}. {description}" if description else goal
        if objects:
            goal_text += f" Objects: {', '.join(objects[:5])}"
        if locations:
            goal_text += f" Locations: {', '.join(locations[:5])}"
        goal_emb = get_normalized_embedding(goal_text)

        step_actions = []
        for s in steps:
            if isinstance(s, dict):
                action = s.get('action', '')
                if action:
                    full_action = action
                    if s.get('object'):
                        full_action += f" with {s['object']}"
                    if s.get('location'):
                        full_action += f" at {s['location']}"
                    step_actions.append(full_action)
            elif isinstance(s, str) and s:
                step_actions.append(s)

        if step_actions:
            step_embs = batch_get_normalized_embeddings(step_actions)
            step_emb = np.mean(step_embs, axis=0)
            step_emb = step_emb / (np.linalg.norm(step_emb) + 1e-8)
        else:
            step_emb = goal_emb.copy()

        dag = self._construct_dag(steps, edges)

        return {
            'proc_id': proc_id,
            'type': 'procedure',
            'goal': goal,
            'description': description,
            'proc_type': proc_type,
            'steps': steps,
            'edges': edges,
            'dag': dag,
            'objects': objects,
            'locations': locations,
            'participants': participants,
            'episodic_links': [{
                'clip_id': clip_id,
                'relevance': 'source',
                'similarity': 1.0,
                'content_preview': clip_content['content'][:100]
            }],
            'embeddings': {
                'goal_emb': goal_emb,
                'step_emb': step_emb,
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'source': 'incremental_nstf',
                'observation_count': 1,
                'source_clips': [clip_id],
            }
        }

    def _construct_dag(self, steps: list, edges: list) -> Dict:
        """Build procedural DAG with START/GOAL control nodes."""
        nodes = {
            'START': {'type': 'control', 'attributes': {}},
            'GOAL': {'type': 'control', 'attributes': {}}
        }

        step_ids = []
        for i, s in enumerate(steps):
            if isinstance(s, dict):
                step_id = s.get('step_id', f'step_{i+1}')
                attributes = {
                    'object': s.get('object', ''),
                    'location': s.get('location', ''),
                    'actor': s.get('actor', ''),
                    'triggers': s.get('triggers', []),
                    'outcomes': s.get('outcomes', []),
                    'duration_seconds': s.get('duration_seconds', 0)
                }
                nodes[step_id] = {
                    'type': 'action',
                    'action': s.get('action', ''),
                    'attributes': attributes
                }
                step_ids.append(step_id)
            elif isinstance(s, str):
                step_id = f'step_{i+1}'
                nodes[step_id] = {
                    'type': 'action',
                    'action': s,
                    'attributes': {'object': '', 'location': '', 'actor': '', 'triggers': [], 'outcomes': [], 'duration_seconds': 0}
                }
                step_ids.append(step_id)

        dag_edges = []
        if edges:
            for e in edges:
                dag_edges.append({
                    'from': e.get('from_step', e.get('from', '')),
                    'to': e.get('to_step', e.get('to', '')),
                    'count': e.get('count', 1),
                    'probability': e.get('probability', 1.0),
                    'condition': e.get('condition', None)
                })
        elif step_ids:
            dag_edges.append({'from': 'START', 'to': step_ids[0], 'count': 1, 'probability': 1.0, 'condition': None})
            for i in range(len(step_ids) - 1):
                dag_edges.append({'from': step_ids[i], 'to': step_ids[i+1], 'count': 1, 'probability': 1.0, 'condition': None})
            dag_edges.append({'from': step_ids[-1], 'to': 'GOAL', 'count': 1, 'probability': 1.0, 'condition': None})

        return {'nodes': nodes, 'edges': dag_edges}

    def update_procedure(
        self,
        proc: Dict,
        clip_id: int,
        clip_content: Dict,
        detected: Dict
    ):
        """Update an existing procedure with anchored EMA."""
        self.update_with_anchored_ema(
            proc,
            clip_id,
            clip_content,
            detected,
            beta=self.config.get('ema_beta', 0.9),
            drift_threshold=self.config.get('drift_threshold', 0.7),
            anchor_weight=self.config.get('anchor_weight', 0.3),
            update_clip_content=clip_content
        )

    def update_with_anchored_ema(
        self,
        proc: Dict,
        clip_id: int,
        clip_content: Dict,
        detected: Dict,
        beta: float = 0.9,
        drift_threshold: float = 0.7,
        anchor_weight: float = 0.3,
        update_clip_content: Dict = None
    ):
        """Perform anchored EMA update on procedure embeddings with drift correction."""
        import numpy as np

        embeddings = proc.get('embeddings', {})

        if 'anchor_goal_emb' not in embeddings:
            if 'goal_emb' in embeddings:
                embeddings['anchor_goal_emb'] = embeddings['goal_emb'].copy()
            else:
                goal_text = f"{proc.get('goal', '')}. {proc.get('description', '')}"
                embeddings['anchor_goal_emb'] = get_normalized_embedding(goal_text)
                embeddings['goal_emb'] = embeddings['anchor_goal_emb'].copy()

        if 'anchor_step_emb' not in embeddings and 'step_emb' in embeddings:
            embeddings['anchor_step_emb'] = embeddings['step_emb'].copy()

        proc['embeddings'] = embeddings

        new_content_emb = get_normalized_embedding(clip_content['content'])

        old_goal_emb = embeddings['goal_emb']
        anchor_goal_emb = embeddings['anchor_goal_emb']

        new_goal_emb = beta * old_goal_emb + (1 - beta) * new_content_emb
        new_goal_emb = new_goal_emb / (np.linalg.norm(new_goal_emb) + 1e-8)

        goal_drift_sim = cosine_similarity(new_goal_emb, anchor_goal_emb)

        if goal_drift_sim < drift_threshold:
            new_goal_emb = anchor_weight * anchor_goal_emb + (1 - anchor_weight) * new_goal_emb
            new_goal_emb = new_goal_emb / (np.linalg.norm(new_goal_emb) + 1e-8)

            if 'drift_events' not in proc.get('metadata', {}):
                proc['metadata']['drift_events'] = []
            proc['metadata']['drift_events'].append({
                'clip_id': clip_id,
                'type': 'goal',
                'drift_sim': float(goal_drift_sim),
                'corrected': True
            })

        embeddings['goal_emb'] = new_goal_emb

        new_steps = detected.get('steps', [])
        if new_steps and 'step_emb' in embeddings:
            step_actions = []
            for s in new_steps:
                if isinstance(s, dict):
                    action = s.get('action', '')
                    if action:
                        step_actions.append(action)
                elif isinstance(s, str):
                    step_actions.append(s)

            if step_actions:
                new_step_embs = batch_get_normalized_embeddings(step_actions)
                new_step_emb = np.mean(new_step_embs, axis=0)
                new_step_emb = new_step_emb / (np.linalg.norm(new_step_emb) + 1e-8)

                old_step_emb = embeddings['step_emb']
                updated_step_emb = beta * old_step_emb + (1 - beta) * new_step_emb
                updated_step_emb = updated_step_emb / (np.linalg.norm(updated_step_emb) + 1e-8)

                if 'anchor_step_emb' in embeddings:
                    anchor_step_emb = embeddings['anchor_step_emb']
                    step_drift_sim = cosine_similarity(updated_step_emb, anchor_step_emb)

                    if step_drift_sim < drift_threshold:
                        updated_step_emb = anchor_weight * anchor_step_emb + (1 - anchor_weight) * updated_step_emb
                        updated_step_emb = updated_step_emb / (np.linalg.norm(updated_step_emb) + 1e-8)

                embeddings['step_emb'] = updated_step_emb

        self._update_dag_edge_counts(proc, detected, update_clip_content or clip_content)

        similarity = float(np.dot(old_goal_emb, new_content_emb))
        proc['episodic_links'].append({
            'clip_id': clip_id,
            'relevance': 'update',
            'similarity': round(similarity, 4),
            'anchor_similarity': round(float(goal_drift_sim), 4),
            'content_preview': clip_content['content'][:100]
        })

        proc['metadata']['observation_count'] = proc['metadata'].get('observation_count', 0) + 1
        proc['metadata']['updated_at'] = datetime.now().isoformat()
        if 'source_clips' not in proc['metadata']:
            proc['metadata']['source_clips'] = []
        proc['metadata']['source_clips'].append(clip_id)

        if new_steps and len(new_steps) > len(proc.get('steps', [])):
            normalized_steps = []
            for i, s in enumerate(new_steps):
                if isinstance(s, dict):
                    step = {
                        'step_id': s.get('step_id', f'step_{i+1}'),
                        'action': s.get('action', ''),
                        'object': s.get('object', ''),
                        'location': s.get('location', ''),
                        'actor': s.get('actor', ''),
                        'triggers': s.get('triggers', []),
                        'outcomes': s.get('outcomes', []),
                        'duration_seconds': s.get('duration_seconds', 0)
                    }
                    normalized_steps.append(step)
                elif isinstance(s, str) and s:
                    normalized_steps.append({
                        'step_id': f'step_{i+1}',
                        'action': s,
                        'object': '',
                        'location': '',
                        'actor': '',
                        'triggers': [],
                        'outcomes': [],
                        'duration_seconds': 0
                    })

            proc['steps'] = normalized_steps
            new_edges = detected.get('edges', [])
            proc['dag'] = self._construct_dag(normalized_steps, new_edges)

    def _update_dag_edge_counts(self, proc: Dict, detected: Dict, clip_content: Dict = None):
        """Update DAG edge transition counts and recompute probabilities."""
        dag = proc.get('dag')
        if not dag:
            return

        edges = dag.get('edges', [])
        nodes = dag.get('nodes', {})
        if not edges:
            return

        new_edges = detected.get('edges', [])
        if new_edges and self._has_branching(new_edges):
            for new_edge in new_edges:
                from_step = new_edge.get('from_step') or new_edge.get('from', '')
                to_step = new_edge.get('to_step') or new_edge.get('to', '')
                self._increment_edge_count(edges, from_step, to_step)
        else:
            observed_steps = self._infer_observed_steps(nodes, detected, clip_content)

            if observed_steps:
                self._increment_edge_count(edges, 'START', observed_steps[0])

                for i in range(len(observed_steps) - 1):
                    self._increment_edge_count(edges, observed_steps[i], observed_steps[i+1])

                self._increment_edge_count(edges, observed_steps[-1], 'GOAL')

        self._recompute_edge_probabilities(dag)

    def _has_branching(self, edges: List[Dict]) -> bool:
        """Check if edge list contains branching structure."""
        from collections import Counter
        sources = [e.get('from_step') or e.get('from', '') for e in edges]
        source_counts = Counter(sources)
        return any(count > 1 for count in source_counts.values())

    def _infer_observed_steps(self, nodes: Dict, detected: Dict, clip_content: Dict = None) -> List[str]:
        """Infer which existing steps were observed based on content similarity."""
        step_node_ids = [k for k in nodes.keys() if k not in ['START', 'GOAL']]
        if not step_node_ids:
            return []

        new_steps = detected.get('steps', [])
        if new_steps:
            new_actions = []
            for s in new_steps:
                if isinstance(s, dict):
                    action = s.get('action', '')
                    if action:
                        new_actions.append(action)

            if new_actions:
                from .utils import batch_get_normalized_embeddings, cosine_similarity

                existing_actions = []
                existing_ids = []
                for sid in step_node_ids:
                    node_data = nodes.get(sid, {})
                    action = node_data.get('action', '')
                    if action:
                        existing_actions.append(action)
                        existing_ids.append(sid)

                if existing_actions and new_actions:
                    all_texts = existing_actions + new_actions
                    all_embs = batch_get_normalized_embeddings(all_texts)

                    existing_embs = all_embs[:len(existing_actions)]
                    new_embs = all_embs[len(existing_actions):]

                    observed = []
                    MATCH_THRESHOLD = 0.5

                    for new_emb in new_embs:
                        best_sim = -1
                        best_id = None
                        for i, ex_emb in enumerate(existing_embs):
                            sim = cosine_similarity(new_emb, ex_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_id = existing_ids[i]

                        if best_id and best_sim >= MATCH_THRESHOLD and best_id not in observed:
                            observed.append(best_id)

                    if observed:
                        observed.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
                        return observed

        if clip_content and clip_content.get('content'):
            from .utils import get_normalized_embedding, cosine_similarity
            clip_emb = get_normalized_embedding(clip_content['content'])

            step_sims = []
            for sid in step_node_ids:
                node_data = nodes.get(sid, {})
                action = node_data.get('action', '')
                if action:
                    action_emb = get_normalized_embedding(action)
                    sim = cosine_similarity(clip_emb, action_emb)
                    step_sims.append((sid, sim))

            OBSERVE_THRESHOLD = 0.5
            observed = [sid for sid, sim in step_sims if sim >= OBSERVE_THRESHOLD]

            if observed:
                observed.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
                return observed

        return []

    def _increment_edge_count(self, edges: List[Dict], from_step: str, to_step: str):
        """Increment transition count for a specific edge."""
        for edge in edges:
            edge_from = edge.get('from') or edge.get('from_step', '')
            edge_to = edge.get('to') or edge.get('to_step', '')

            if edge_from == from_step and edge_to == to_step:
                edge['count'] = edge.get('count', 1) + 1
                return True
        return False

    def _recompute_edge_probabilities(self, dag: Dict):
        """Posterior mean of each outgoing transition under a symmetric Dirichlet(γ)
        prior; this is the (γ + c_j) / Σ_i (γ + c_i) estimator from Theorem 1, which
        smooths sparse counts toward 1/K and converges to MLE as observations grow."""
        edges = dag.get('edges', [])

        from_counts = {}
        from_categories = {}
        for edge in edges:
            from_step = edge.get('from') or edge.get('from_step', '')
            count = edge.get('count', 1)
            from_counts[from_step] = from_counts.get(from_step, 0) + count
            from_categories[from_step] = from_categories.get(from_step, 0) + 1

        gamma = 1.0
        for edge in edges:
            from_step = edge.get('from') or edge.get('from_step', '')
            count = edge.get('count', 1)
            total = from_counts.get(from_step, 0)
            k = from_categories.get(from_step, 1)
            denom = gamma * k + total
            edge['probability'] = (gamma + count) / denom if denom > 0 else 1.0

    def build(
        self,
        video_name: str,
        dataset: str = 'web',
        max_clips: int = None,
    ) -> Optional[Dict]:
        """Incrementally build NSTF graph by processing clips sequentially."""

        start_time = time.time()

        graph = self.load_baseline_graph(video_name, dataset)
        if graph is None:
            return None

        self.character_resolver = CharacterResolver(graph, debug=self.debug)

        clips = self.get_sorted_clips(graph)
        if max_clips:
            clips = clips[:max_clips]

        nstf_graph = {
            'video_name': video_name,
            'dataset': dataset,
            'procedure_nodes': {},
            'character_mapping': self.character_resolver.mapping,
            'metadata': {
                'build_mode': 'incremental',
                'created_at': datetime.now().isoformat(),
            }
        }

        proc_counter = 0
        skipped_count = 0
        detected_count = 0

        for i, clip_id in enumerate(clips):
            clip_content = self.get_clip_content(graph, clip_id)

            if not clip_content['content'].strip():
                skipped_count += 1
                continue

            detected = self.extractor.detect_in_clip(clip_content)

            if not detected:
                skipped_count += 1
                continue

            detected_count += 1
            self.stats['clips_processed'] += 1

            matched_proc = self.matcher.match_existing(nstf_graph, detected)

            if matched_proc:
                self.update_procedure(matched_proc, clip_id, clip_content, detected)
                self.stats['procedures_merged'] += 1
            else:
                proc_counter += 1
                proc_id = f"{video_name}_proc_{proc_counter}"
                new_proc = self.create_procedure_node(detected, clip_id, clip_content, proc_id)
                nstf_graph['procedure_nodes'][proc_id] = new_proc
                self.stats['procedures_created'] += 1

            time.sleep(0.3)

        self.linker.clear_cache()
        self.matcher.clear_log()

        self.stats['videos_processed'] += 1

        proc_nodes = nstf_graph['procedure_nodes']
        if len(proc_nodes) >= 2 and self.config.get('enable_fusion', True):
            procedures_before = len(proc_nodes)
            fused_nodes = self.fusion_manager.fuse_all(proc_nodes)
            nstf_graph['procedure_nodes'] = fused_nodes
            proc_nodes = fused_nodes

        total_links = sum(len(p.get('episodic_links', [])) for p in proc_nodes.values())
        fusion_stats = self.fusion_manager.get_stats()

        nstf_graph['stats'] = {
            'total_procedures': len(proc_nodes),
            'total_links': total_links,
            'avg_links_per_proc': total_links / len(proc_nodes) if proc_nodes else 0,
            'merges': self.stats['procedures_merged'],
            'creates': self.stats['procedures_created'],
            'dag_fusions': fusion_stats.get('total_fusions', 0),
        }

        nstf_graph['metadata']['fusion_enabled'] = self.config.get('enable_fusion', True)

        output_subdir = self.output_dir / dataset
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_path = output_subdir / f'{video_name}_nstf.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(nstf_graph, f)

        return nstf_graph

    def build_batch(
        self,
        video_names: List[str],
        dataset: str = 'web',
        max_clips: int = None,
    ):
        """Batch incremental build for multiple videos."""
        for video_name in video_names:
            try:
                self.build(video_name, dataset, max_clips)
                time.sleep(2)
            except Exception as e:
                import traceback
                if self.debug:
                    traceback.print_exc()

