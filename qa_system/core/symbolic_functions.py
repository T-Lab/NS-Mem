import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict


def _node_satisfies(node: Dict, constraints: Dict) -> bool:
    """Check whether node attributes / action satisfy every entry in the constraint
    set C.  Two reserved keys are honored directly: `exclude_keywords` (none of the
    listed tokens may appear in the action) and `require_keywords` (all listed
    tokens must appear).  Other keys are matched against the node's attribute map.
    Absent attributes do not violate constraints, so partially-annotated DAGs can
    still answer constrained queries."""
    if not constraints:
        return True

    action = (node.get('action') or '').lower()
    attrs = node.get('attributes') or node.get('metadata', {}).get('attributes', {}) or {}

    exclude = constraints.get('exclude_keywords') or []
    for kw in exclude:
        if kw and kw.lower() in action:
            return False

    require = constraints.get('require_keywords') or []
    for kw in require:
        if kw and kw.lower() not in action:
            return False

    for key, expected in constraints.items():
        if key in ('exclude_keywords', 'require_keywords'):
            continue
        actual = attrs.get(key)
        if actual is None:
            continue
        if callable(expected):
            if not expected(actual):
                return False
        elif isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def extract_constraints_from_query(query: str) -> Optional[Dict]:
    """Pull `without X` / `except X` / `instead of X` / `missing X` / `other than X`
    spans out of a natural-language query and return them as an `exclude_keywords`
    constraint suitable for queryStepSequence."""
    if not query:
        return None
    patterns = [
        r'\bwithout\s+(?:the\s+|a\s+|an\s+)?([\w-]+)',
        r'\bmissing\s+(?:the\s+|a\s+|an\s+)?([\w-]+)',
        r'\bexcept\s+(?:for\s+)?(?:the\s+|a\s+|an\s+)?([\w-]+)',
        r'\binstead\s+of\s+(?:the\s+|a\s+|an\s+)?([\w-]+)',
        r'\bother\s+than\s+(?:the\s+|a\s+|an\s+)?([\w-]+)',
        r'\bif\s+(?:there\s+is\s+)?no\s+([\w-]+)',
    ]
    excluded = []
    q = query.lower()
    for pat in patterns:
        for m in re.finditer(pat, q):
            token = m.group(1).strip()
            if token and token not in excluded:
                excluded.append(token)
    if not excluded:
        return None
    return {'exclude_keywords': excluded}


@dataclass
class ProcedureResult:
    """Procedure query result."""
    proc_id: str
    goal: str
    steps: List[str]
    episodic_evidence: Dict[int, str]  # clip_id -> content
    similarity: float
    match_type: str  # "goal", "step", "combined"


@dataclass
class StepQueryResult:
    """Step query result."""
    proc_id: str
    goal: str
    total_steps: int
    query_type: str  # "count", "first", "last", "after", "before", "all", "path"
    result: str
    full_sequence: List[str]
    paths: Optional[List[List[str]]] = None  # DAG multi-paths
    reference: Optional[str] = None


@dataclass
class CharacterResult:
    """Character analysis result."""
    character_id: str
    character_name: str
    involved_procedures: List[Dict]
    behavior_summary: str
    evidence_clips: List[int]


class ProcedureDAG:
    """DAG representation of a Procedure supporting linear and branching step sequences."""

    def __init__(self, proc_id: str, goal: str):
        self.proc_id = proc_id
        self.goal = goal

        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}

        self.entry_nodes: List[str] = []
        self.exit_nodes: List[str] = []

    def add_step(self, step_id: str, action: str, attributes: Dict = None, metadata: Dict = None):
        self.nodes[step_id] = {
            'action': action,
            'attributes': attributes or {},
            'metadata': metadata or {},
        }

    def add_edge(self, from_step: str, to_step: str, probability: float = 1.0, condition: str = None):
        self.edges[(from_step, to_step)] = {
            'probability': probability,
            'condition': condition,
        }

    def build_from_procedure(self, proc: Dict):
        """Build DAG honoring the stored edges; fall back to a linear chain only
        when the procedure has no edges (single-observation case)."""
        steps = proc.get('steps', []) or []
        edges = proc.get('edges', []) or []
        if not steps:
            return

        id_alias = {}
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                sid = step.get('step_id') or f"step_{i}"
                action = step.get('action', '')
                attrs = step.get('attributes', {}) or {}
            else:
                sid = f"step_{i}"
                action = str(step)
                attrs = {}
            self.add_step(sid, action, attributes=attrs, metadata=step if isinstance(step, dict) else {})
            id_alias[i] = sid
            id_alias[sid] = sid
            id_alias[f"step_{i}"] = sid

        real_step_ids = [id_alias[i] for i in range(len(steps))]

        if edges:
            for e in edges:
                src = e.get('from_step') or e.get('from')
                dst = e.get('to_step') or e.get('to')
                if src is None or dst is None:
                    continue
                src_resolved = id_alias.get(src, src)
                dst_resolved = id_alias.get(dst, dst)
                prob = e.get('probability', 1.0)
                cond = e.get('condition')
                self.add_edge(src_resolved, dst_resolved, probability=prob, condition=cond)

            sources = {s for (s, _) in self.edges}
            targets = {t for (_, t) in self.edges}
            entry_candidates = [sid for sid in real_step_ids if sid not in targets and sid in sources]
            if not entry_candidates:
                entry_candidates = [s for s in sources if s == 'START'] or [real_step_ids[0]]
            exit_candidates = [sid for sid in real_step_ids if sid not in sources and sid in targets]
            if not exit_candidates:
                exit_candidates = [t for t in targets if t == 'GOAL'] or [real_step_ids[-1]]
            self.entry_nodes = entry_candidates
            self.exit_nodes = exit_candidates
        else:
            for i in range(len(real_step_ids) - 1):
                self.add_edge(real_step_ids[i], real_step_ids[i + 1])
            self.entry_nodes = [real_step_ids[0]]
            self.exit_nodes = [real_step_ids[-1]]

    def build_from_steps(self, steps: List[Dict]):
        self.build_from_procedure({'steps': steps, 'edges': []})

    def enumerate_paths(self, max_paths: int = 10, constraints: Dict = None) -> List[List[str]]:
        """Enumerate paths from entry to exit.  When `constraints` is given, drop any
        path containing a node whose attributes contradict the constraint set C; this is
        the A(v) |= C filter from queryStepSequence."""
        if not self.entry_nodes or not self.exit_nodes:
            if self.nodes:
                actions = [n['action'] for n in self.nodes.values() if n.get('action')]
                if not actions:
                    return []
                if constraints and not all(
                    _node_satisfies(n, constraints) for n in self.nodes.values()
                ):
                    return []
                return [actions]
            return []

        paths = []
        exit_set = set(self.exit_nodes)

        def dfs(current, path, visited):
            if len(paths) >= max_paths:
                return
            if current in visited or current not in self.nodes:
                return
            if constraints and not _node_satisfies(self.nodes[current], constraints):
                return

            visited = visited | {current}
            action = self.nodes[current].get('action', '')
            new_path = path + [action] if action else path

            if current in exit_set:
                if new_path:
                    paths.append(new_path)
                return

            successors = [t for (s, t) in self.edges if s == current]
            if not successors and new_path:
                paths.append(new_path)
                return
            for succ in successors:
                dfs(succ, new_path, visited)

        for entry in self.entry_nodes:
            dfs(entry, [], set())

        return paths

    def get_linear_sequence(self) -> List[str]:
        """Get linear step sequence (main path)."""
        paths = self.enumerate_paths(max_paths=1)
        return paths[0] if paths else []

    def find_step_by_content(self, content: str) -> Optional[str]:
        """Find step ID by fuzzy content match."""
        content_lower = content.lower()

        for step_id, node in self.nodes.items():
            action = node['action'].lower()
            if content_lower in action or action in content_lower:
                return step_id

        return None


class SymbolicFunctions:
    def __init__(self, video_graph=None, nstf_graph: Dict = None):
        self.video_graph = video_graph
        self.nstf_graph = nstf_graph

        self._dag_cache: Dict[str, ProcedureDAG] = {}

    def set_graphs(self, video_graph=None, nstf_graph: Dict = None):
        """Update graph references."""
        if video_graph:
            self.video_graph = video_graph
        if nstf_graph:
            self.nstf_graph = nstf_graph
            self._dag_cache.clear()

    def get_procedure_with_evidence(
        self,
        proc_id: str,
        include_evidence: bool = True
    ) -> ProcedureResult:
        """Get procedure with episodic evidence."""
        proc_nodes = self.nstf_graph.get('procedure_nodes', {}) if self.nstf_graph else {}
        proc = proc_nodes.get(proc_id, {})

        steps = []
        for step in proc.get('steps', []):
            if isinstance(step, dict):
                action = step.get('action', '')
                if action:
                    steps.append(action)

        evidence = {}
        if include_evidence:
            for link in proc.get('episodic_links', []):
                clip_id = link.get('clip_id')
                if clip_id is not None:
                    content = self._get_clip_content(clip_id)
                    if content:
                        evidence[clip_id] = content

        return ProcedureResult(
            proc_id=proc_id,
            goal=proc.get('goal', 'Unknown'),
            steps=steps,
            episodic_evidence=evidence,
            similarity=0.0,
            match_type='procedure',
        )

    def query_step_sequence(
        self,
        proc_id: str,
        query: str = '',
        constraints: Optional[Dict] = None,
        use_dag: bool = True,
    ) -> StepQueryResult:
        """Enumerate paths from START to GOAL in the Procedural DAG and keep only those
        whose nodes all satisfy the attribute constraints in C.  Without constraints this
        degenerates to the unfiltered path set used by the descriptive `query` modes."""
        proc_nodes = self.nstf_graph.get('procedure_nodes', {}) if self.nstf_graph else {}
        proc = proc_nodes.get(proc_id, {})

        dag = self._get_or_build_dag(proc_id, proc)

        if use_dag:
            all_paths = dag.enumerate_paths(max_paths=20, constraints=constraints)
        else:
            linear = dag.get_linear_sequence()
            all_paths = [linear] if linear else []
        linear_seq = all_paths[0] if all_paths else []

        result = StepQueryResult(
            proc_id=proc_id,
            goal=proc.get('goal', 'Unknown'),
            total_steps=len(linear_seq),
            query_type='all',
            result='',
            full_sequence=linear_seq,
            paths=all_paths if len(all_paths) > 1 else None,
        )

        if constraints is not None:
            result.query_type = 'constrained'
            if all_paths:
                result.result = '\n'.join(
                    f"Path {i+1}: " + ' -> '.join(p) for i, p in enumerate(all_paths)
                )
            else:
                result.result = 'No path satisfies the given constraints.'
            return result

        if not linear_seq:
            result.result = 'No steps found'
            return result

        query_lower = (query or '').lower()

        if any(kw in query_lower for kw in ['how many', 'count']):
            result.query_type = 'count'
            result.result = f'{len(linear_seq)} steps'

        elif any(kw in query_lower for kw in ['first', 'begin', 'start']):
            result.query_type = 'first'
            result.result = linear_seq[0]

        elif any(kw in query_lower for kw in ['last', 'final', 'end']):
            result.query_type = 'last'
            result.result = linear_seq[-1]

        elif any(kw in query_lower for kw in ['after', 'then', 'next']):
            result.query_type = 'after'
            ref_action = self._find_reference_action(query, linear_seq)
            if ref_action and ref_action['index'] < len(linear_seq) - 1:
                result.result = linear_seq[ref_action['index'] + 1]
                result.reference = ref_action['action']
            else:
                result.result = linear_seq[-1]

        elif any(kw in query_lower for kw in ['before', 'previous']):
            result.query_type = 'before'
            ref_action = self._find_reference_action(query, linear_seq)
            if ref_action and ref_action['index'] > 0:
                result.result = linear_seq[ref_action['index'] - 1]
                result.reference = ref_action['action']
            else:
                result.result = linear_seq[0]

        elif any(kw in query_lower for kw in ['path', 'alternative', 'other way']):
            result.query_type = 'path'
            if all_paths and len(all_paths) > 1:
                path_strs = [' -> '.join(p) for p in all_paths]
                result.result = f"Found {len(all_paths)} alternative paths:\n" + \
                               '\n'.join(f"  Path {i+1}: {p}" for i, p in enumerate(path_strs))
            else:
                result.result = ' -> '.join(linear_seq)
        else:
            result.query_type = 'all'
            result.result = ' -> '.join(linear_seq)

        return result

    def aggregate_character_behaviors(
        self,
        character_id: str,
        name_resolver=None
    ) -> CharacterResult:
        """Character behavior pattern aggregation."""
        proc_nodes = self.nstf_graph.get('procedure_nodes', {}) if self.nstf_graph else {}

        character_name = character_id
        if name_resolver:
            character_name = name_resolver.get_character_name(character_id)

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

                clip_content = self._get_clip_content(clip_id)
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

        summary_parts = []
        if involved_procs:
            summary_parts.append(f"{character_name} is involved in {len(involved_procs)} procedure(s):")
            for proc in involved_procs:
                summary_parts.append(f"  - {proc['goal']} ({proc['proc_type']})")

            if len(involved_procs) >= 2:
                goals = [p['goal'] for p in involved_procs]
                common_words = self._find_common_themes(goals)
                if common_words:
                    summary_parts.append(
                        f"Behavior pattern: Frequently involved in {', '.join(common_words)}-related activities."
                    )
        else:
            summary_parts.append(f"No procedure information found for {character_name}.")

        return CharacterResult(
            character_id=character_id,
            character_name=character_name,
            involved_procedures=involved_procs,
            behavior_summary='\n'.join(summary_parts),
            evidence_clips=sorted(evidence_clips),
        )

    def _get_or_build_dag(self, proc_id: str, proc: Dict) -> ProcedureDAG:
        """Get or build DAG for a Procedure."""
        if proc_id in self._dag_cache:
            return self._dag_cache[proc_id]

        dag = ProcedureDAG(proc_id, proc.get('goal', 'Unknown'))
        dag.build_from_procedure(proc)

        self._dag_cache[proc_id] = dag
        return dag

    def _get_clip_content(self, clip_id: int) -> str:
        """Get clip content."""
        if not self.video_graph:
            return ""

        if hasattr(self.video_graph, 'text_nodes_by_clip'):
            if clip_id not in self.video_graph.text_nodes_by_clip:
                return ""

            node_ids = self.video_graph.text_nodes_by_clip[clip_id]
            contents = []

            for nid in node_ids:
                node = self.video_graph.nodes.get(nid)
                if node and hasattr(node, 'metadata'):
                    node_contents = node.metadata.get('contents', [])
                    contents.extend(node_contents)

            return ' '.join(str(c) for c in contents)

        return ""

    def _find_reference_action(self, query: str, steps: List[str]) -> Optional[Dict]:
        """Find reference action from query."""
        query_lower = query.lower()

        for i, step in enumerate(steps):
            step_lower = step.lower()
            query_words = set(query_lower.split())
            step_words = set(step_lower.split())
            overlap = query_words & step_words - {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'or'}

            if len(overlap) >= 2:
                return {'index': i, 'action': step}

        return None

    def _find_common_themes(self, texts: List[str]) -> List[str]:
        """Find common theme words from a list of texts."""
        if not texts:
            return []

        word_counts = defaultdict(int)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'or', 'for', 'in', 'on', 'at'}

        for text in texts:
            words = re.findall(r'\w+', text.lower())
            unique_words = set(words) - stopwords
            for word in unique_words:
                if len(word) > 2:
                    word_counts[word] += 1

        threshold = max(2, len(texts) // 2)
        common = [word for word, count in word_counts.items() if count >= threshold]

        return common[:5]

    def format_procedure_result(self, result: ProcedureResult) -> str:
        """Format Procedure result for prompt."""
        lines = [
            f"--- Procedure (Relevance: {result.similarity:.2f}, matched by {result.match_type}) ---",
            f"Goal: {result.goal}",
        ]

        for i, step in enumerate(result.steps, 1):
            lines.append(f"Step {i}: {step}")

        if result.episodic_evidence:
            lines.append("\n[Evidence from episodic memory]:")
            for clip_id, content in result.episodic_evidence.items():
                lines.append(f"  Clip {clip_id}: {content[:200]}...")

        return '\n'.join(lines)

    def format_step_query_result(self, result: StepQueryResult) -> str:
        """Format step query result for prompt."""
        lines = [
            f"--- Procedure: {result.goal} ---",
            f"Query Type: {result.query_type}",
            f"Total Steps: {result.total_steps}",
        ]

        if result.reference:
            lines.append(f"Reference Action: {result.reference}")

        lines.append(f"Result: {result.result}")

        if result.paths and len(result.paths) > 1:
            lines.append(f"\n[Alternative paths available: {len(result.paths)}]")

        return '\n'.join(lines)

    def format_character_result(self, result: CharacterResult) -> str:
        """Format character analysis result for prompt."""
        lines = [
            f"--- Character Analysis: {result.character_name} ---",
            result.behavior_summary,
        ]

        if result.evidence_clips:
            lines.append(f"\nEvidence clips: {result.evidence_clips}")

        return '\n'.join(lines)
