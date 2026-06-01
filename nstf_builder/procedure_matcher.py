"""Multi-signal procedure matcher for incremental fusion decisions."""

import re
import numpy as np
from typing import Dict, List, Optional, Set

from .utils import get_normalized_embedding


class ProcedureMatcher:
    """Match new procedures against existing ones using goal similarity, type, and verb overlap."""

    def __init__(
        self,
        match_threshold: float = 0.65,
        weights: Dict[str, float] = None,
        debug: bool = False
    ):
        self.match_threshold = match_threshold
        self.weights = weights or {'goal': 0.5, 'type': 0.2, 'verb': 0.3}
        self.debug = debug
        self.decision_log = []

    def match_existing(
        self,
        nstf_graph: Dict,
        detected: Dict
    ) -> Optional[Dict]:
        """Find the best matching existing procedure or return None to create new."""
        proc_nodes = nstf_graph.get('procedure_nodes', {})
        if not proc_nodes:
            return None

        detected_goal = detected.get('goal', '')
        detected_type = detected.get('type', 'task')

        detected_emb = get_normalized_embedding(detected_goal)
        detected_verbs = self._extract_verbs(detected_goal)

        candidates = []

        for proc_id, proc in proc_nodes.items():
            signals = self._compute_signals(
                detected_emb, detected_type, detected_verbs,
                proc
            )

            score = (
                self.weights['goal'] * signals['goal_sim'] +
                self.weights['type'] * signals['type_match'] +
                self.weights['verb'] * signals['verb_overlap']
            )

            candidates.append({
                'proc_id': proc_id,
                'proc': proc,
                'score': score,
                'signals': signals
            })

        best = max(candidates, key=lambda x: x['score'])

        decision = {
            'detected_goal': detected_goal[:50],
            'detected_type': detected_type,
            'best_match': best['proc_id'],
            'score': round(best['score'], 4),
            'signals': {k: round(v, 4) for k, v in best['signals'].items()},
            'decision': 'merge' if best['score'] >= self.match_threshold else 'create_new'
        }
        self.decision_log.append(decision)

        return best['proc'] if best['score'] >= self.match_threshold else None

    def _compute_signals(
        self,
        detected_emb: np.ndarray,
        detected_type: str,
        detected_verbs: Set[str],
        proc: Dict
    ) -> Dict[str, float]:
        """Compute matching signals: goal similarity, type match, verb overlap."""
        signals = {}

        proc_emb = proc.get('embeddings', {}).get('goal_emb')
        if proc_emb is not None:
            signals['goal_sim'] = float(np.dot(detected_emb, proc_emb))
        else:
            signals['goal_sim'] = 0.0

        proc_type = proc.get('proc_type', 'task')
        signals['type_match'] = 1.0 if detected_type == proc_type else 0.5

        proc_verbs = self._extract_verbs(proc.get('goal', ''))
        signals['verb_overlap'] = self._jaccard(detected_verbs, proc_verbs)

        return signals

    def _extract_verbs(self, text: str) -> Set[str]:
        """Extract action verbs from text."""
        verb_patterns = [
            r'\b(make|cook|prepare|clean|wash|wipe|put|place|store|open|close|turn|check|get|take|give|find|use|move|bring|carry|hold|set|keep|leave|start|stop|finish|begin|end)\b',
            r'\b(chop|stir|fry|boil|bake|pour|mix|season|slice|peel|heat|cool|freeze|defrost|marinate|grill|roast|steam)\b',
            r'\b(scrub|sweep|mop|dust|vacuum|rinse|dry|polish|tidy|organize|arrange|sort|dispose|throw|discard)\b',
            r'\b(greet|talk|ask|answer|tell|say|explain|discuss|help|assist|serve|offer|invite|welcome)\b',
            r'\b(making|cooking|preparing|cleaning|washing|wiping|putting|placing|storing|opening|closing|turning|checking|getting|taking)\b',
        ]

        verbs = set()
        text_lower = text.lower()
        for pattern in verb_patterns:
            matches = re.findall(pattern, text_lower)
            verbs.update(matches)

        return verbs

    def _jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def get_decision_summary(self) -> Dict:
        """Get summary of merge/create decisions."""
        if not self.decision_log:
            return {'total': 0, 'merges': 0, 'creates': 0}

        merges = sum(1 for d in self.decision_log if d['decision'] == 'merge')
        creates = len(self.decision_log) - merges

        return {
            'total': len(self.decision_log),
            'merges': merges,
            'creates': creates,
            'merge_rate': merges / len(self.decision_log) if self.decision_log else 0
        }

    def clear_log(self):
        """Clear decision log."""
        self.decision_log = []
