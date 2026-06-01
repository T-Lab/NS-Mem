"""Character ID resolver for unifying face/voice/character references."""

from typing import Dict, Set
import re


class CharacterResolver:
    """Resolve face/voice/character IDs into unified person_N identifiers."""

    def __init__(self, video_graph, debug: bool = False):
        self.mapping: Dict[str, str] = {}
        self.equivalences: Dict[str, Set[str]] = {}
        self.video_graph = video_graph
        self.debug = debug
        self._resolve()

    def _resolve(self):
        """Build character mapping from graph metadata or equivalences."""
        if hasattr(self.video_graph, 'metadata') and self.video_graph.metadata:
            mapping = self.video_graph.metadata.get('character_mapping', {})
            if mapping:
                self.mapping = mapping
                return

        self._extract_equivalences()
        self._build_unified_mapping()

    def _extract_equivalences(self):
        """Extract equivalence relations from semantic nodes."""
        if not hasattr(self.video_graph, 'nodes'):
            return

        equiv_pattern = re.compile(r'Equivalence:\s*(<[^>]+>)\s*,\s*(<[^>]+>)')

        for node_id, node in self.video_graph.nodes.items():
            node_type = getattr(node, 'type', None)
            if node_type != 'semantic':
                continue

            metadata = getattr(node, 'metadata', {})
            if not metadata:
                continue

            contents = metadata.get('contents', [])
            for content in contents:
                match = equiv_pattern.search(content)
                if match:
                    id1, id2 = match.group(1), match.group(2)
                    self._add_equivalence(id1, id2)

    def _add_equivalence(self, id1: str, id2: str):
        """Add an equivalence relation between two IDs."""
        group1 = self._find_equivalence_group(id1)
        group2 = self._find_equivalence_group(id2)

        if group1 is None and group2 is None:
            new_group = {id1, id2}
            self.equivalences[id1] = new_group
            self.equivalences[id2] = new_group
        elif group1 is None:
            group2.add(id1)
            self.equivalences[id1] = group2
        elif group2 is None:
            group1.add(id2)
            self.equivalences[id2] = group1
        elif group1 is not group2:
            merged = group1 | group2
            for id_ in merged:
                self.equivalences[id_] = merged

    def _find_equivalence_group(self, id_: str) -> Set[str]:
        """Find the equivalence group for a given ID."""
        return self.equivalences.get(id_)

    def _build_unified_mapping(self):
        """Build unified person_N mapping from equivalence groups."""
        all_ids = set()

        for id_ in self.equivalences.keys():
            all_ids.add(id_)

        id_pattern = re.compile(r'<(face|voice|character)_(\d+)>')

        if hasattr(self.video_graph, 'text_nodes_by_clip'):
            for clip_id in self.video_graph.text_nodes_by_clip.keys():
                node_ids = self.video_graph.text_nodes_by_clip[clip_id]
                for nid in node_ids:
                    node = self.video_graph.nodes.get(nid)
                    if node:
                        meta = getattr(node, 'metadata', {})
                        contents = meta.get('contents', [])
                        for c in contents:
                            matches = id_pattern.findall(c)
                            for prefix, num in matches:
                                all_ids.add(f'<{prefix}_{num}>')

        processed_groups = set()
        person_counter = 1

        for id_ in all_ids:
            if id_ in self.mapping:
                continue

            group = self._find_equivalence_group(id_)
            if group:
                group_key = frozenset(group)
                if group_key in processed_groups:
                    continue
                processed_groups.add(group_key)

                unified_name = f"person_{person_counter}"
                person_counter += 1

                for member in group:
                    self.mapping[member] = unified_name
            else:
                self.mapping[id_] = f"person_{person_counter}"
                person_counter += 1

    def resolve(self, content: str) -> str:
        """Replace face/voice/character IDs in content with unified names."""
        if not self.mapping:
            return content

        result = content
        for id_, name in self.mapping.items():
            if id_ in result:
                result = result.replace(id_, name)

        return result

    def get_mapping_context(self) -> str:
        """Generate a mapping context string for prompts."""
        if not self.mapping:
            return ""

        by_person = {}
        for id_, person in self.mapping.items():
            if person not in by_person:
                by_person[person] = []
            by_person[person].append(id_)

        parts = []
        for person, ids in by_person.items():
            if len(ids) > 1:
                parts.append(f"{person}={','.join(sorted(ids))}")

        if parts:
            return "Person Equivalences: " + "; ".join(parts)
        return ""

    def has_unresolved_ids(self, content: str) -> bool:
        """Check if content still contains unresolved IDs."""
        pattern = r'<(face|voice|character)[_\s]?\d+>'
        return bool(re.search(pattern, content, re.IGNORECASE))

    def get_unresolved_ids(self, content: str) -> list:
        """Get all unresolved IDs in content."""
        pattern = r'<(face|voice|character)[_\s]?\d+>'
        return re.findall(pattern, content, re.IGNORECASE)


def create_character_resolver(video_graph, debug: bool = False) -> CharacterResolver:
    """Factory function."""
    return CharacterResolver(video_graph, debug=debug)
