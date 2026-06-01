import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONSTRAINT = "constraint"


@dataclass
class ClassificationResult:
    query_type: QueryType
    confidence: float
    method: str
    matched_pattern: Optional[str] = None
    involves_character: bool = False


class QueryClassifier:
    CONSTRAINT_PATTERNS = [
        (r'\bwithout\s+\w+', 'without_X'),
        (r'\bif\s+(?:there\s+is\s+)?no\s+\w+', 'if_no_X'),
        (r'\balternative\b', 'alternative'),
        (r'\bmissing\b', 'missing'),
        (r'\binstead\s+of\b', 'instead_of'),
        (r'\bexcept\b', 'except'),
        (r'\bother\s+than\b', 'other_than'),
    ]

    PROCEDURAL_PATTERNS = [
        (r'\bhow\s+(?:to|do|does|should|can|could)\b', 'how_to'),
        (r'\bsteps?\b', 'steps'),
        (r'\bprocedure\b', 'procedure'),
        (r'\bprocess\b', 'process'),
        (r'\bwhat\s+to\s+do\b', 'what_to_do'),
        (r'\bin\s+what\s+order\b', 'in_order'),
        (r'\bfirst.*then\b', 'first_then'),
    ]

    CHARACTER_PATTERNS = [
        (r'\b(?:usually|often|always|frequently)\b', 'usually'),
        (r'\bhabit\b', 'habit'),
        (r'\btend\s+to\b', 'tend_to'),
        (r'\b(?:good|skilled|familiar)\s+(?:at|with)\b', 'good_at'),
        (r'\bdoes\s+\w+\s+like\b', 'does_X_like'),
        (r'\bis\s+\w+\s+(?:good|skilled)\b', 'is_X_good'),
        (r'\bpersonality\b', 'personality'),
        (r'\bbehavior\b', 'behavior'),
    ]

    def __init__(
        self,
        use_llm_refinement: bool = False,
        llm_client=None,
        confidence_threshold: float = 0.7,
    ):
        self.use_llm_refinement = use_llm_refinement
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> ClassificationResult:
        rule_result = self._rule_based_classify(query)

        if rule_result.confidence >= self.confidence_threshold:
            return rule_result

        if self.use_llm_refinement and self.llm_client is not None:
            return self._llm_based_classify(query, rule_result)

        return rule_result

    def _count_matches(self, query_lower: str, patterns) -> int:
        return sum(1 for pat, _ in patterns if re.search(pat, query_lower))

    @staticmethod
    def _score(n_matches: int) -> float:
        if n_matches <= 0:
            return 0.5
        return min(1.0, 0.6 + 0.15 * n_matches)

    def _rule_based_classify(self, query: str) -> ClassificationResult:
        query_lower = (query or '').lower()

        n_constraint = self._count_matches(query_lower, self.CONSTRAINT_PATTERNS)
        n_procedural = self._count_matches(query_lower, self.PROCEDURAL_PATTERNS)
        n_character = self._count_matches(query_lower, self.CHARACTER_PATTERNS)
        involves_character = n_character > 0

        if n_constraint >= max(n_procedural, n_character) and n_constraint > 0:
            return ClassificationResult(
                query_type=QueryType.CONSTRAINT,
                confidence=self._score(n_constraint),
                method='rule',
                matched_pattern=self._first_match(query_lower, self.CONSTRAINT_PATTERNS),
                involves_character=involves_character,
            )

        if n_procedural >= n_character and n_procedural > 0:
            return ClassificationResult(
                query_type=QueryType.PROCEDURAL,
                confidence=self._score(n_procedural),
                method='rule',
                matched_pattern=self._first_match(query_lower, self.PROCEDURAL_PATTERNS),
                involves_character=involves_character,
            )

        return ClassificationResult(
            query_type=QueryType.FACTUAL,
            confidence=self._score(n_character),
            method='rule',
            matched_pattern=self._first_match(query_lower, self.CHARACTER_PATTERNS) if involves_character else None,
            involves_character=involves_character,
        )

    def _first_match(self, query_lower: str, patterns) -> Optional[str]:
        for pat, name in patterns:
            if re.search(pat, query_lower):
                return name
        return None

    def _llm_based_classify(
        self,
        query: str,
        rule_result: ClassificationResult,
    ) -> ClassificationResult:
        prompt = (
            "Classify the question into exactly one of: factual, procedural, constraint.\n"
            "- factual: direct recall of who/what/when/where.\n"
            "- procedural: how-to / step ordering.\n"
            "- constraint: requires satisfying or violating an attribute "
            "(without X, alternative to X, except X, instead of X, ...).\n"
            f"Question: {query}\n"
            "Answer with just one word: factual, procedural, or constraint."
        )
        try:
            response = self.llm_client.generate(
                [{'role': 'user', 'content': prompt}],
                timeout=15,
            )
        except Exception:
            return rule_result

        token = (response or '').strip().lower().split()
        if not token:
            return rule_result
        head = token[0].strip('.,;:').rstrip('s')
        mapping = {
            'factual': QueryType.FACTUAL,
            'procedural': QueryType.PROCEDURAL,
            'constraint': QueryType.CONSTRAINT,
        }
        chosen = mapping.get(head)
        if chosen is None:
            return rule_result
        return ClassificationResult(
            query_type=chosen,
            confidence=max(rule_result.confidence, 0.75),
            method='llm',
            matched_pattern=rule_result.matched_pattern,
            involves_character=rule_result.involves_character,
        )
