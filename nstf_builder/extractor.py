import json
import time
from typing import List, Dict, Optional
from pathlib import Path


class ProcedureExtractor:
    def __init__(
        self,
        llm_model: str = 'gemini-2.5-flash',
        batch_size: int = 20,
        max_content_chars: int = 150,
        api_delay: float = 1.0,
        support_threshold: float = 0.2,
        verify_threshold: float = 0.25,
    ):
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.max_content_chars = max_content_chars
        self.api_delay = api_delay
        self.support_threshold = support_threshold
        self.verify_threshold = verify_threshold

        self._chat_api = None

    @property
    def chat_api(self):
        if self._chat_api is None:
            from env_setup import setup_paths
            setup_paths()
            from mmagent.utils.chat_api import get_response_with_retry, generate_messages
            self._chat_api = (get_response_with_retry, generate_messages)
        return self._chat_api

    def _get_gemini_response(self, prompt: str, max_retries: int = 5, timeout: int = 45) -> Optional[str]:
        """Call LLM API with retry and timeout handling."""
        get_response, generate_messages = self.chat_api
        timeout_count = 0

        for i in range(max_retries):
            try:
                messages = generate_messages([{"type": "text", "content": prompt}])
                response, _ = get_response(self.llm_model, messages, timeout=timeout)
                return response
            except Exception as e:
                error_str = str(e).lower()

                if "504" in error_str or "timeout" in error_str or "timed out" in error_str:
                    timeout_count += 1
                    if timeout_count >= 5:
                        return None
                else:
                    return None

        return None

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response with tolerance for common format issues."""
        if not text:
            return None

        text = text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        start = text.find('{')
        end = text.rfind('}') + 1
        if start < 0 or end <= start:
            return None

        json_str = text[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        try:
            import re
            fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        try:
            import re
            detected_match = re.search(r'"detected"\s*:\s*(true|false)', json_str, re.IGNORECASE)
            if detected_match:
                if detected_match.group(1).lower() == 'false':
                    return {"detected": False}
        except:
            pass

        return None

    def detect_procedures(
        self,
        contents: List[Dict],
        max_procedures: int = 5,
    ) -> List[Dict]:
        """Mine procedural patterns from episodic contents and gate them with LLMVerify."""
        action_sequences = self._extract_action_sequences(contents)
        candidates = self._run_prefix_span(action_sequences, self.support_threshold)

        verified: List[Dict] = []
        for pattern in candidates:
            score = self._llm_verify(pattern, contents)
            if score >= self.verify_threshold:
                verified.append(pattern)

        return self._deduplicate(verified, max_procedures)

    def _extract_action_sequences(self, contents: List[Dict]) -> List[List[str]]:
        return []

    def _run_prefix_span(
        self,
        sequences: List[List[str]],
        min_support: float,
    ) -> List[Dict]:
        return []

    def _llm_verify(self, pattern: Dict, related_memories: List[Dict]) -> float:
        return 0.0

    def extract_structure(
        self,
        contents: List[Dict],
        procedure: Dict,
    ) -> Optional[Dict]:
        """Extract detailed DAG structure for a single procedure."""

        goal = procedure.get('goal', '')
        if isinstance(goal, dict):
            goal = goal.get('name', str(goal))
        goal = str(goal)

        proc_type = procedure.get('type', 'task')
        if isinstance(proc_type, dict):
            proc_type = 'task'

        source_clips = procedure.get('source_clips', [])
        clean_clips = []
        for clip in source_clips[:3]:
            if isinstance(clip, int):
                clean_clips.append(clip)
            elif isinstance(clip, dict):
                clean_clips.append(clip.get('clip_id', 1))
            else:
                try:
                    clean_clips.append(int(clip))
                except:
                    clean_clips.append(1)

        sample_contents = [c['content'][:200] for c in contents[:20]]

        episodic_links_list = [{"clip_id": c, "relevance": "source"} for c in clean_clips]
        episodic_links_json = json.dumps(episodic_links_list)

        prompt = ""

        response = self._get_gemini_response(prompt, max_retries=5, timeout=60)
        result = self._parse_json(response)

        if result:
            result = self._ensure_dag_completeness(result)
            result['source_clips'] = clean_clips

        return result

    def _ensure_dag_completeness(self, structure: Dict) -> Dict:
        """Ensure DAG has proper START/GOAL connections and edge attributes."""
        steps = structure.get('steps', [])
        edges = structure.get('edges', [])

        if not steps:
            return structure

        step_ids = []
        for s in steps:
            if isinstance(s, dict):
                step_ids.append(s.get('step_id', f'step_{len(step_ids)+1}'))

        if not step_ids:
            return structure

        has_start_edge = any(
            e.get('from_step') == 'START' or e.get('from') == 'START'
            for e in edges
        )
        has_goal_edge = any(
            e.get('to_step') == 'GOAL' or e.get('to') == 'GOAL'
            for e in edges
        )

        if not has_start_edge and step_ids:
            edges.insert(0, {
                'from_step': 'START',
                'to_step': step_ids[0],
                'count': 1,
                'probability': 1.0,
                'condition': None
            })

        if not has_goal_edge and step_ids:
            sources = {e.get('from_step') or e.get('from') for e in edges}
            targets = {e.get('to_step') or e.get('to') for e in edges}
            leaf_nodes = [sid for sid in step_ids if sid not in sources]

            if not leaf_nodes:
                leaf_nodes = [step_ids[-1]]

            for leaf in leaf_nodes:
                edges.append({
                    'from_step': leaf,
                    'to_step': 'GOAL',
                    'count': 1,
                    'probability': 1.0,
                    'condition': None
                })

        for edge in edges:
            if 'count' not in edge:
                edge['count'] = 1
            if 'probability' not in edge:
                edge['probability'] = 1.0

        structure['edges'] = edges
        return structure

    def _deduplicate(self, procedures: List[Dict], max_count: int) -> List[Dict]:
        """Deduplicate procedures by goal string."""
        seen = set()
        unique = []
        for p in procedures:
            goal = p.get('goal', '')
            if isinstance(goal, dict):
                goal = goal.get('name', str(goal))
            goal = str(goal)

            source_clips = p.get('source_clips', [])
            if source_clips:
                cleaned_clips = []
                for clip in source_clips:
                    if isinstance(clip, int):
                        cleaned_clips.append(clip)
                    elif isinstance(clip, dict):
                        cleaned_clips.append(clip.get('clip_id', 0))
                    else:
                        try:
                            cleaned_clips.append(int(clip))
                        except:
                            pass
                p['source_clips'] = cleaned_clips

            if goal and goal not in seen:
                seen.add(goal)
                unique.append(p)
        return unique[:max_count]

    def detect_in_clip(self, clip_content: Dict) -> Optional[Dict]:
        """Detect and extract procedural knowledge from a single clip."""
        content = clip_content.get('content', '')
        clip_id = clip_content.get('clip_id', 0)

        if not content or len(content.strip()) < 20:
            return None

        prompt = ""

        response = self._get_gemini_response(prompt, max_retries=3, timeout=45)
        result = self._parse_json(response)

        if result and result.get('detected'):
            result['source_clips'] = [clip_id]

            if result.get('steps'):
                result = self._ensure_dag_completeness(result)

            return result

        return None
