import os
import re
import json
import time
import gc
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .config import QAConfig, BaselineConfig, NSTFConfig, get_mode_config
from .prompts import get_system_prompt, get_instruction
from .core import Retriever, Evaluator, LLMClient
from .core.hybrid_retriever import HybridRetriever, create_retriever, RetrievalResult
from .core.name_resolver import NameResolver


SCHEMA_VERSION = "1.0"

ACTION_PATTERN = r"Action: \[(.*)\].*Content: (.*)"


def get_method_name(mode: str = None, ablation_mode: str = None) -> str:
    """Derive method name from mode or ablation_mode."""
    if mode:
        if mode == 'nstf_full':
            return 'nstf'
        return mode

    if ablation_mode is None or ablation_mode == 'full_nstf':
        return 'nstf'
    elif ablation_mode == 'baseline':
        return 'baseline'
    elif ablation_mode == 'prototype':
        return 'ablation_prototype'
    elif ablation_mode == 'structure':
        return 'ablation_structure'
    else:
        return ablation_mode


class ResultStore:
    """Structured result storage with atomic writes and JSONL indexing."""

    def __init__(self, base_dir: Path, method: str, dataset: str):
        self.base_dir = Path(base_dir)
        self.method = method
        self.dataset = dataset

        self.method_dir = self.base_dir / method
        self.dataset_dir = self.method_dir / dataset
        self.index_file = self.method_dir / f'index_{dataset}.jsonl'

        self.method_dir.mkdir(parents=True, exist_ok=True)

    def get_result_path(self, video_id: str, question_id: str) -> Path:
        """Get path for a single result file."""
        return self.dataset_dir / video_id / f'{question_id}.json'

    def exists(self, video_id: str, question_id: str) -> bool:
        """Check if a result already exists."""
        return self.get_result_path(video_id, question_id).exists()

    def get_completed_ids(self) -> set:
        """Get set of completed question IDs by scanning the filesystem."""
        completed = set()
        if not self.dataset_dir.exists():
            return completed

        for video_dir in self.dataset_dir.iterdir():
            if video_dir.is_dir():
                for json_file in video_dir.glob('*.json'):
                    if not json_file.name.endswith('.tmp'):
                        completed.add(json_file.stem)
        return completed

    def save(self, result: dict) -> bool:
        """Save a single result with atomic write."""
        video_id = result['video_id']
        question_id = result['id']

        video_dir = self.dataset_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        target_path = video_dir / f'{question_id}.json'
        tmp_path = video_dir / f'{question_id}.json.tmp'

        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            tmp_path.replace(target_path)
            self._append_index(result)
            return True

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def _append_index(self, result: dict):
        """Append an index entry."""
        index_entry = {
            'id': result['id'],
            'video_id': result['video_id'],
            'status': result.get('status', 'success'),
            'type_original': result.get('type_original', []),
            'type_query': result.get('type_query'),
            'gpt_eval': result.get('gpt_eval', False),
            'num_rounds': result.get('num_rounds', 0),
            'elapsed_time_sec': result.get('elapsed_time_sec', 0),
            'search_count': result.get('search_count', 0),
            'timestamp': result.get('timestamp'),
        }

        with open(self.index_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + '\n')


class QARunner:
    """QA system runner."""

    def __init__(
        self,
        config: QAConfig = None,
        data_dir: str = None,
        output_dir: str = None,
    ):
        self.nstf_model_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.nstf_model_dir / 'data'
        self.output_dir = Path(output_dir) if output_dir else self.nstf_model_dir / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or QAConfig.load_default()
        self._init_components()
        self.results = []

    def _init_components(self):
        """Initialize retriever, LLM client, and evaluator."""
        mode = getattr(self.config, 'mode', 'baseline')

        if mode in ['baseline', 'nstf_full', 'ablation_prototype', 'ablation_structure']:
            self.retriever = create_retriever(
                mode=mode,
                threshold=self.config.nstf_threshold,
                min_confidence=self.config.nstf_min_confidence,
                max_procedures=self.config.nstf_max_procedures,
                threshold_baseline=self.config.threshold_baseline,
                topk=self.config.topk,
                use_reranking=self.config.nstf_use_reranking,
                use_dag_paths=self.config.nstf_use_dag_paths,
            )
            self._use_hybrid = True
            self._use_v2 = False
            self._use_nstf = False
        elif self.config.retrieval_strategy == 'node_level':
            from .core.retriever_v2 import RetrieverV2
            self.retriever = RetrieverV2(
                strategy='node_level',
                threshold=self.config.node_threshold,
                topk=self.config.node_topk,
                include_timestamp=self.config.include_timestamp,
                group_by_clip=self.config.group_by_clip,
                include_semantic=self.config.include_semantic,
                preserve_clip_order=self.config.preserve_clip_order,
            )
            self._use_hybrid = False
            self._use_v2 = True
            self._use_nstf = False
        elif self.config.retrieval_strategy == 'nstf_level':
            from .core.retriever_nstf import NSTFRetriever
            self.retriever = NSTFRetriever(
                threshold=self.config.nstf_threshold,
                min_confidence=self.config.nstf_min_confidence,
                max_procedures=self.config.nstf_max_procedures,
                topk_baseline=self.config.topk,
                threshold_baseline=self.config.threshold_baseline,
                include_episodic_evidence=self.config.nstf_include_evidence,
            )
            self._use_hybrid = False
            self._use_v2 = False
            self._use_nstf = True
        else:
            self.retriever = Retriever(
                ablation_mode=self.config.ablation_mode,
                threshold=self.config.threshold,
                threshold_baseline=self.config.threshold_baseline,
                topk=self.config.topk,
            )
            self._use_hybrid = False
            self._use_v2 = False
            self._use_nstf = False

        self.llm = LLMClient(
            source=self.config.llm_source,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

        self.evaluator = Evaluator(
            model=self.config.gpt_model,
            timeout=self.config.eval_timeout,
        )

    def load_annotations(self, dataset: str = None, data_file: str = None) -> Dict:
        """Load annotation data."""
        if data_file:
            data_path = Path(data_file)
        else:
            data_path = self.data_dir / 'annotations' / f'{dataset}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_video_list(self, video_list_file: str) -> List[str]:
        """Load video list from a JSON file."""
        with open(video_list_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            videos = data
        else:
            videos = data.get('videos', [])

        return videos

    def detect_nstf_graph(self, video_name: str, dataset: str) -> Optional[str]:
        """Detect NSTF graph availability. Priority: incremental > static."""
        nstf_dir = self.data_dir / 'nstf_graphs' / dataset

        incremental_path = nstf_dir / f'{video_name}_nstf_incremental.pkl'
        if incremental_path.exists():
            return str(incremental_path)

        static_path = nstf_dir / f'{video_name}_nstf.pkl'
        if static_path.exists():
            return str(static_path)

        return None

    def prepare_data(
        self,
        annotations: Dict,
        dataset: str,
        video_names: List[str] = None,
    ) -> List[Dict]:
        """Prepare batch data from annotations."""
        data_list = []
        nstf_count = 0

        if video_names is None:
            video_names = list(annotations.keys())

        for video_name in video_names:
            if video_name not in annotations:
                continue

            video_data = annotations[video_name]

            nstf_path = self.detect_nstf_graph(video_name, dataset)
            nstf_available = nstf_path is not None
            if nstf_available:
                nstf_count += 1

            mem_path = video_data.get('mem_path')
            if not mem_path:
                mem_path = f'data/memory_graphs/{dataset}/{video_name}.pkl'
            if not os.path.isabs(mem_path):
                mem_path = str(self.data_dir.parent / mem_path)

            for qa in video_data.get('qa_list', []):
                item = {
                    'id': qa.get('question_id', f'{video_name}_Q{len(data_list)+1}'),
                    'video_id': video_name,
                    'mem_path': mem_path,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'type': qa.get('type', []),
                    'type_query': qa.get('type_query'),
                    'nstf_available': nstf_available,
                    'nstf_path': nstf_path,
                }
                if 'before_clip' in qa:
                    item['before_clip'] = qa['before_clip']
                data_list.append(item)

        return data_list

    def _get_system_prompt(self, nstf_available: bool) -> str:
        """Get system prompt based on mode and NSTF availability."""
        mode = getattr(self.config, 'mode', None)
        return get_system_prompt(
            mode=mode,
            ablation_mode=self.config.ablation_mode,
            nstf_available=nstf_available
        )

    def _get_instruction(self, nstf_available: bool) -> str:
        """Get instruction based on mode and NSTF availability."""
        mode = getattr(self.config, 'mode', None)
        return get_instruction(
            mode=mode,
            ablation_mode=self.config.ablation_mode,
            nstf_available=nstf_available
        )

    def _init_conversation(self, data: Dict) -> Dict:
        """Initialize multi-round conversation state."""
        nstf_available = data.get('nstf_available', False)
        prompt = self._get_system_prompt(nstf_available)
        data['conversations'] = [
            {"role": "system", "content": prompt.format(question=data['question'])},
            {"role": "user", "content": "Searched knowledge: {}"}
        ]
        data['finish'] = False
        data['current_clips'] = []
        data['retrieval_trace'] = []
        data['_instruction'] = self._get_instruction(nstf_available)
        return data

    def _process_response(self, data: Dict, round_idx: int) -> Dict:
        """Process model response: execute search or extract answer."""
        if data['finish']:
            return data

        response = data['conversations'][-1]['content']

        clean_response = response.split("</think>")[-1] if "</think>" in response else response
        match = re.search(ACTION_PATTERN, clean_response, re.DOTALL)

        if match:
            action = match.group(1)
            content = match.group(2)
        else:
            action = "Search"
            content = None

        if action == "Answer":
            data['response'] = content
            data['finish'] = True
        else:
            memories = {}
            retrieval_info = None

            if content:
                if self._use_hybrid:
                    result: RetrievalResult = self.retriever.search(
                        mem_path=data['mem_path'],
                        query=content,
                        current_clips=data['current_clips'],
                        nstf_path=data.get('nstf_path'),
                        before_clip=data.get('before_clip'),
                    )
                    memories = result.memories
                    data['current_clips'] = result.clips
                    retrieval_info = result.metadata
                    if retrieval_info:
                        data['nstf_decision'] = retrieval_info.get('decision', 'unknown')
                        data['query_type'] = retrieval_info.get('query_type')
                        if 'matched_procedures' in retrieval_info:
                            data['matched_procedures'] = retrieval_info['matched_procedures']

                        if retrieval_info.get('use_baseline_prompt') and not data.get('_prompt_switched'):
                            baseline_prompt = get_system_prompt(
                                mode='baseline',
                                ablation_mode=None,
                                nstf_available=False
                            )
                            data['conversations'][0]['content'] = baseline_prompt.format(question=data['question'])
                            data['_instruction'] = get_instruction(
                                mode='baseline',
                                ablation_mode=None,
                                nstf_available=False
                            )
                            data['_prompt_switched'] = True
                            data['fallback_reason'] = retrieval_info.get('fallback_reason', 'unknown')
                elif self._use_v2:
                    result = self.retriever.search(
                        mem_path=data['mem_path'],
                        query=content,
                        current_context=data.get('current_context', []),
                        before_clip=data.get('before_clip'),
                    )
                    memories = result['memories']
                    data['current_context'] = result['context']
                    retrieval_info = result['metadata']
                elif self._use_nstf:
                    memories, data['current_clips'], retrieval_info = self.retriever.search(
                        mem_path=data['mem_path'],
                        query=content,
                        current_clips=data['current_clips'],
                        nstf_path=data.get('nstf_path'),
                        before_clip=data.get('before_clip'),
                    )
                    if retrieval_info:
                        data['nstf_decision'] = retrieval_info.get('decision', 'unknown')
                        if 'matched_procedures' in retrieval_info:
                            data['matched_procedures'] = retrieval_info['matched_procedures']
                else:
                    memories, data['current_clips'], retrieval_info = self.retriever.search(
                        mem_path=data['mem_path'],
                        query=content,
                        current_clips=data['current_clips'],
                        nstf_path=data.get('nstf_path'),
                        before_clip=data.get('before_clip'),
                    )

                if memories:
                    trace_entry = {
                        'round': round_idx + 1,
                        'query': content.strip() if content else '',
                        'num_results': len(memories),
                        'clips': self._extract_clip_info(memories, retrieval_info)
                    }
                    if (self._use_nstf or self._use_hybrid) and retrieval_info:
                        trace_entry['nstf_decision'] = retrieval_info.get('decision')
                        trace_entry['matched_procedures'] = retrieval_info.get('matched_procedures', [])
                    data['retrieval_trace'].append(trace_entry)

            search_result = "Searched knowledge: " + json.dumps(
                memories, ensure_ascii=False
            ).encode("utf-8", "ignore").decode("utf-8")

            if not memories:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"

            data['conversations'].append({"role": "user", "content": search_result})

        return data

    def _extract_clip_info(self, memories: Dict, retrieval_info: Any) -> List[Dict]:
        """Extract clip info from retrieval results."""
        clips = []
        for key, value in memories.items():
            clip_id = None
            source = 'episodic'

            if key == 'NSTF_Procedures':
                source = 'procedure'
                continue
            elif key.startswith('clip_'):
                try:
                    clip_id = int(key.split('_')[1])
                except:
                    pass

            if clip_id is not None:
                clips.append({
                    'clip_id': clip_id,
                    'score': None,
                    'source': source
                })

        return clips

    def _extract_final_answer(self, data: Dict) -> str:
        """Extract the final answer from conversation."""
        if 'response' in data:
            return data['response']

        for conv in reversed(data['conversations']):
            if conv.get('role') == 'assistant':
                content = conv.get('content', '')
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

                match = re.search(ACTION_PATTERN, content, re.DOTALL)
                if match:
                    action = match.group(1)
                    answer_content = match.group(2)
                    if action == "Answer":
                        return answer_content.strip()
                    continue

        return ""

    def run_single(self, data: Dict) -> Dict:
        """Process a single question through multi-round conversation."""
        start_time = time.time()

        try:
            if self._use_v2:
                self.retriever.reset()

            data = self._init_conversation(data)

            if self._use_v2:
                data['current_context'] = []

            actual_rounds = 0

            for round_idx in range(self.config.total_round):
                if data['finish']:
                    break

                actual_rounds += 1

                messages = []
                instruction = data.get('_instruction', get_instruction(ablation_mode='baseline'))
                for conv in data['conversations']:
                    msg = conv.copy()
                    if conv['role'] == 'user' and conv == data['conversations'][-1]:
                        msg['content'] += instruction
                        if round_idx == self.config.total_round - 1:
                            msg['content'] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                    messages.append(msg)

                response = self.llm.generate(messages)
                data['conversations'].append({"role": "assistant", "content": response})

                data = self._process_response(data, round_idx)

            if 'response' not in data:
                data['response'] = self._extract_final_answer(data)

            elapsed_time = time.time() - start_time
            data['num_rounds'] = actual_rounds
            data['elapsed_time_sec'] = round(elapsed_time, 2)
            data['search_count'] = len(data.get('retrieval_trace', []))

            return data

        finally:
            if self._use_v2:
                self.retriever.reset()

    def run(
        self,
        dataset: str = 'robot',
        video_names: List[str] = None,
        video_list_file: str = None,
        data_file: str = None,
        query_type_file: str = None,
        limit: int = None,
        output_file: str = None,
        resume: bool = True,
        force: bool = False,
    ) -> Dict:
        """Run QA evaluation over a dataset."""
        annotations = self.load_annotations(dataset, data_file)

        if video_list_file:
            video_names = self.load_video_list(video_list_file)

        data_list = self.prepare_data(annotations, dataset, video_names)

        query_type_map = {}
        if query_type_file:
            query_type_map = self._load_query_type_map(query_type_file)

        if limit:
            data_list = data_list[:limit]

        use_structured_storage = (output_file is None)
        method_name = get_method_name(self.config.ablation_mode)

        if use_structured_storage:
            store = ResultStore(self.output_dir, method_name, dataset)

            if force:
                completed_ids = set()
            else:
                completed_ids = store.get_completed_ids()
        else:
            store = None
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            completed_ids = set()
            if resume and output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if 'id' in item:
                                completed_ids.add(item['id'])
                        except:
                            pass

        total_before_filter = len(data_list)
        data_list = [d for d in data_list if d['id'] not in completed_ids]

        if len(data_list) == 0 and len(completed_ids) > 0:
            if use_structured_storage:
                pass

            return {
                'total': len(completed_ids),
                'correct': 0,
                'accuracy': None,
                'avg_rounds': None,
                'avg_time_sec': None,
                'method': method_name,
                'dataset': dataset,
                'skipped': True,
                'message': f'All {len(completed_ids)} questions completed, use --force to re-run',
            }

        results_count = 0
        correct_count = 0
        total_rounds = 0
        total_time = 0.0

        for idx, data in enumerate(data_list):
            try:
                result = self.run_single(data)

                result['gpt_eval'] = self.evaluator.evaluate(
                    result['question'],
                    result.get('response', ''),
                    result['answer']
                )

                full_result = self._build_full_result(
                    result, dataset, method_name, query_type_map
                )

                if use_structured_storage:
                    store.save(full_result)
                else:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(full_result, ensure_ascii=False) + '\n')

                results_count += 1
                if full_result.get('gpt_eval', False):
                    correct_count += 1
                total_rounds += full_result.get('num_rounds', 0)
                total_time += full_result.get('elapsed_time_sec', 0)

            except Exception as e:
                error_result = self._build_error_result(
                    data, dataset, method_name, query_type_map, str(e)
                )
                if use_structured_storage:
                    store.save(error_result)
                else:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                results_count += 1
                continue

            if (idx + 1) % 50 == 0:
                gc.collect()

        accuracy = correct_count / results_count if results_count > 0 else 0
        avg_rounds = total_rounds / results_count if results_count > 0 else 0
        avg_time = total_time / results_count if results_count > 0 else 0

        return {
            'total': results_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'avg_rounds': round(avg_rounds, 2),
            'avg_time_sec': round(avg_time, 2),
            'method': method_name,
            'dataset': dataset,
        }

    def _load_query_type_map(self, query_type_file: str) -> Dict[str, str]:
        """Load query type annotation mapping."""
        query_type_map = {}
        with open(query_type_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for video_id, video_data in data.items():
            for qa in video_data.get('qa_list', []):
                qid = qa.get('question_id')
                qtype = qa.get('type')
                if qid and qtype:
                    if isinstance(qtype, str):
                        query_type_map[qid] = qtype

        return query_type_map

    def _build_full_result(
        self,
        result: Dict,
        dataset: str,
        method: str,
        query_type_map: Dict[str, str],
    ) -> Dict:
        """Build a schema-compliant result record."""
        question_id = result['id']

        return {
            'schema_version': SCHEMA_VERSION,
            'status': 'success',
            'error_message': None,
            'id': question_id,
            'video_id': result['video_id'],
            'dataset': dataset,
            'method': method,
            'question': result['question'],
            'answer': result['answer'],
            'type_original': result.get('type', []),
            'type_query': query_type_map.get(question_id) or result.get('type_query'),
            'response': result.get('response', ''),
            'gpt_eval': result.get('gpt_eval', False),
            'num_rounds': result.get('num_rounds', 0),
            'elapsed_time_sec': result.get('elapsed_time_sec', 0),
            'search_count': result.get('search_count', 0),
            'retrieval_trace': result.get('retrieval_trace', []),
            'token_stats': {
                'total_input_tokens': None,
                'total_output_tokens': None,
            },
            'nstf_available': result.get('nstf_available', False),
            'nstf_path': result.get('nstf_path'),
            'mem_path': result.get('mem_path', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.') + f'{datetime.now().microsecond // 1000:03d}',
            'llm_model': self.config.llm_model,
            'gpt_model': self.config.gpt_model,
            'conversations': result.get('conversations', []),
        }

    def _build_error_result(
        self,
        data: Dict,
        dataset: str,
        method: str,
        query_type_map: Dict[str, str],
        error_message: str,
    ) -> Dict:
        """Build an error-state result record."""
        question_id = data['id']

        return {
            'schema_version': SCHEMA_VERSION,
            'status': 'error',
            'error_message': error_message,
            'id': question_id,
            'video_id': data['video_id'],
            'dataset': dataset,
            'method': method,
            'question': data['question'],
            'answer': data['answer'],
            'type_original': data.get('type', []),
            'type_query': query_type_map.get(question_id),
            'response': '',
            'gpt_eval': False,
            'num_rounds': data.get('num_rounds', 0),
            'elapsed_time_sec': data.get('elapsed_time_sec', 0),
            'search_count': 0,
            'retrieval_trace': [],
            'token_stats': {
                'total_input_tokens': None,
                'total_output_tokens': None,
            },
            'nstf_available': data.get('nstf_available', False),
            'nstf_path': data.get('nstf_path'),
            'mem_path': data.get('mem_path', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.') + f'{datetime.now().microsecond // 1000:03d}',
            'llm_model': self.config.llm_model,
            'gpt_model': self.config.gpt_model,
            'conversations': data.get('conversations', []),
        }
