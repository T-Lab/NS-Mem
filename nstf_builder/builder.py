import os
import sys
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from env_setup import setup_all, NSTF_MODEL_DIR
setup_all()

from .extractor import ProcedureExtractor
from .character_resolver import CharacterResolver
from .episodic_linker import EpisodicLinker
from .dag_fusion import DAGFusion, ProcedureFusionManager
from .utils import get_normalized_embedding, batch_get_normalized_embeddings


class NSTFBuilder:
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
            batch_size=self.config.get('batch_size', 40),
            max_content_chars=self.config.get('max_content_chars', 150),
            api_delay=self.config.get('api_delay_seconds', 1),
        )

        self.episodic_linker = EpisodicLinker(
            verify_threshold=self.config.get('verify_threshold', 0.35),
            discover_threshold=self.config.get('discover_threshold', 0.50),
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
            'procedures_extracted': 0,
            'total_steps': 0,
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

    def extract_episodic_contents(self, graph, max_clips: int = None) -> List[Dict]:
        """Extract episodic node contents from graph."""
        contents = []

        for node_id, node in graph.nodes.items():
            if getattr(node, 'type', '') == 'episodic':
                metadata = getattr(node, 'metadata', {})
                clip_id = metadata.get('timestamp', 0)
                content_list = metadata.get('contents', [])

                if content_list:
                    contents.append({
                        'clip_id': clip_id,
                        'content': content_list[0] if content_list else '',
                        'node_id': node_id
                    })

        contents.sort(key=lambda x: x['clip_id'])

        if max_clips:
            contents = contents[:max_clips]

        return contents

    def extract_all_episodic(self, graph) -> List[Dict]:
        """Extract all episodic contents with character resolution applied."""
        all_contents = []

        if hasattr(graph, 'text_nodes_by_clip'):
            for clip_id in sorted(graph.text_nodes_by_clip.keys()):
                node_ids = graph.text_nodes_by_clip[clip_id]
                clip_texts = []

                for nid in node_ids:
                    node = graph.nodes.get(nid)
                    if node and hasattr(node, 'metadata'):
                        contents = node.metadata.get('contents', [])
                        clip_texts.extend(contents)

                if clip_texts:
                    combined = ' '.join(clip_texts)

                    if self.character_resolver:
                        resolved = self.character_resolver.resolve(combined)
                    else:
                        resolved = combined

                    all_contents.append({
                        'clip_id': clip_id,
                        'content': resolved,
                        'raw_content': combined
                    })
        else:
            old_contents = self.extract_episodic_contents(graph)
            for item in old_contents:
                combined = item['content']
                if self.character_resolver:
                    resolved = self.character_resolver.resolve(combined)
                else:
                    resolved = combined
                all_contents.append({
                    'clip_id': item['clip_id'],
                    'content': resolved,
                    'raw_content': combined
                })

        return all_contents

    def create_procedure_node(self, structure: Dict, proc_id: str) -> Dict:
        """Create a procedure node with dual-layer index vectors."""
        import numpy as np

        goal = structure.get('goal', proc_id)
        if isinstance(goal, dict):
            goal = goal.get('name', str(goal))
        goal = str(goal)

        description = structure.get('description', '')
        if isinstance(description, dict):
            description = str(description)

        raw_steps = structure.get('steps', [])

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

        objects = structure.get('objects', structure.get('key_objects', []))
        locations = structure.get('locations', structure.get('key_locations', []))
        participants = structure.get('participants', [])

        step_actions = []
        for s in steps:
            if isinstance(s, dict):
                action = s.get('action', '')
                if isinstance(action, str) and action:
                    full_action = action
                    if s.get('object'):
                        full_action += f" with {s['object']}"
                    if s.get('location'):
                        full_action += f" at {s['location']}"
                    step_actions.append(full_action)
            elif isinstance(s, str) and s:
                step_actions.append(s)

        goal_text = f"{goal}. {description}" if description else goal
        if objects:
            goal_text += f" Objects: {', '.join(objects[:5])}"
        if locations:
            goal_text += f" Locations: {', '.join(locations[:5])}"

        try:
            goal_embedding, _ = self.embedding_api(self.embedding_model, goal_text)
            goal_emb = np.array(goal_embedding)
            goal_emb = goal_emb / (np.linalg.norm(goal_emb) + 1e-8)
        except Exception:
            goal_emb = np.zeros(3072)

        if step_actions:
            try:
                step_embeddings = batch_get_normalized_embeddings(step_actions)
                step_emb = np.mean(step_embeddings, axis=0)
                step_emb = step_emb / (np.linalg.norm(step_emb) + 1e-8)
            except Exception:
                step_emb = goal_emb.copy()
        else:
            step_emb = goal_emb.copy()

        dag = self._construct_dag(steps, structure.get('edges', []))

        return {
            'proc_id': proc_id,
            'type': 'procedure',
            'goal': goal,
            'description': description,
            'proc_type': structure.get('proc_type', structure.get('type', 'task')),
            'steps': steps,
            'dag': dag,
            'edges': structure.get('edges', []),
            'objects': objects,
            'locations': locations,
            'participants': participants,
            'episodic_links': structure.get('episodic_links', []),
            'embeddings': {
                'goal_emb': goal_emb,
                'step_emb': step_emb,
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'observation_count': 1,
                'source_clips': structure.get('source_clips', []),
                'source': 'nstf_extraction',
            }
        }

    def _construct_dag(self, steps: List, edges: List) -> Dict:
        """Build procedural DAG with START/GOAL control nodes."""
        nodes = {
            'START': {'type': 'control', 'attributes': {}},
            'GOAL': {'type': 'control', 'attributes': {}}
        }

        for s in steps:
            if isinstance(s, dict):
                step_id = s.get('step_id', f"step_{len(nodes)-1}")
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
            elif isinstance(s, str):
                step_id = f"step_{len(nodes)-1}"
                nodes[step_id] = {
                    'type': 'action',
                    'action': s,
                    'attributes': {'object': '', 'location': '', 'actor': '', 'triggers': [], 'outcomes': [], 'duration_seconds': 0}
                }

        dag_edges = []
        step_ids = [sid for sid in nodes.keys() if sid not in ('START', 'GOAL')]

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
            dag_edges.append({
                'from': 'START',
                'to': step_ids[0],
                'count': 1,
                'probability': 1.0,
                'condition': None
            })

            for i in range(len(step_ids) - 1):
                dag_edges.append({
                    'from': step_ids[i],
                    'to': step_ids[i + 1],
                    'count': 1,
                    'probability': 1.0,
                    'condition': None
                })

            dag_edges.append({
                'from': step_ids[-1],
                'to': 'GOAL',
                'count': 1,
                'probability': 1.0,
                'condition': None
            })

        return {
            'nodes': nodes,
            'edges': dag_edges
        }

    def build(
        self,
        video_name: str,
        dataset: str = 'web',
        max_procedures: int = None,
    ) -> Optional[Dict]:
        """Build NSTF graph for a single video."""

        max_procedures = max_procedures or self.config.get('max_procedures', 5)

        start_time = time.time()

        graph = self.load_baseline_graph(video_name, dataset)
        if graph is None:
            return None

        self.character_resolver = CharacterResolver(graph, debug=self.debug)

        contents = self.extract_all_episodic(graph)

        procedures = self.extractor.detect_procedures(contents, max_procedures)

        procedure_nodes = {}
        for i, proc in enumerate(procedures):
            proc_id = f"{video_name}_proc_{i+1}"

            try:
                structure = self.extractor.extract_structure(contents, proc)
                if structure:
                    node = self.create_procedure_node(structure, proc_id)

                    verified_links = self.episodic_linker.build_verified_links(
                        procedure=structure,
                        all_episodic_contents=contents,
                    )
                    node['episodic_links'] = verified_links

                    procedure_nodes[proc_id] = node
                    self.stats['total_steps'] += len(structure.get('steps', []))
            except Exception as e:
                import traceback
                if self.debug:
                    traceback.print_exc()

            time.sleep(1)

        self.episodic_linker.clear_cache()

        if len(procedure_nodes) >= 2 and self.config.get('enable_fusion', True):
            procedure_nodes = self.fusion_manager.fuse_all(procedure_nodes)

        self.stats['procedures_extracted'] += len(procedure_nodes)
        self.stats['videos_processed'] += 1

        total_links = sum(len(p.get('episodic_links', [])) for p in procedure_nodes.values())
        fusion_stats = self.fusion_manager.get_stats()
        stats = {
            'total_procedures': len(procedure_nodes),
            'total_links': total_links,
            'avg_links_per_proc': total_links / len(procedure_nodes) if procedure_nodes else 0,
            'character_mapping_found': bool(self.character_resolver.mapping),
            'fusion_performed': fusion_stats.get('total_fusions', 0),
            'procedures_before_fusion': fusion_stats.get('procedures_before', len(procedure_nodes)),
        }

        output_subdir = self.output_dir / dataset
        output_subdir.mkdir(parents=True, exist_ok=True)

        result = {
            'video_name': video_name,
            'dataset': dataset,
            'procedure_nodes': procedure_nodes,
            'character_mapping': self.character_resolver.mapping,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_procedures': len(procedure_nodes),
                'processing_time': time.time() - start_time,
                'fusion_enabled': self.config.get('enable_fusion', True),
            },
            'stats': stats
        }

        output_path = output_subdir / f'{video_name}_nstf.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

        return result

    def build_batch(
        self,
        video_names: List[str],
        dataset: str = 'web',
        max_procedures: int = None,
    ):
        """Batch build NSTF graphs for multiple videos."""
        for video_name in video_names:
            try:
                self.build(video_name, dataset, max_procedures)
                time.sleep(2)
            except Exception:
                pass

