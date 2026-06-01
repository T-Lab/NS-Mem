import re
from typing import Dict, List, Optional, Set, Tuple, Any
from .cache_manager import cache


class NameResolver:
    """Maps entity IDs (face_X, voice_X, character_X) to real names."""

    def __init__(self, video_graph=None, nstf_graph: Dict = None):
        self.video_graph = video_graph
        self.nstf_graph = nstf_graph

        self._name_mapping: Dict[str, str] = {}
        self._initialized = False

    def initialize(self):
        """Initialize name mappings via lazy loading."""
        if self._initialized:
            return

        graph_id = self._get_graph_id()
        cached = cache.get_name_mapping(graph_id)
        if cached:
            self._name_mapping = cached
            self._initialized = True
            return

        if self.video_graph:
            self._build_from_video_graph()

        if self.nstf_graph:
            self._build_from_nstf_graph()

        cache.set_name_mapping(graph_id, self._name_mapping)
        self._initialized = True

    def _get_graph_id(self) -> str:
        """Get graph unique identifier."""
        if self.video_graph and hasattr(self.video_graph, 'video_id'):
            return str(self.video_graph.video_id)
        return "default"

    def _build_from_video_graph(self):
        """Build name mappings from video_graph."""
        graph = self.video_graph

        if hasattr(graph, 'character_mappings'):
            for char_id, tags in graph.character_mappings.items():
                for tag in tags:
                    self._name_mapping[tag] = char_id

        if hasattr(graph, 'reverse_character_mappings'):
            for tag, char_id in graph.reverse_character_mappings.items():
                if tag not in self._name_mapping:
                    self._name_mapping[tag] = char_id

        if hasattr(graph, 'nodes'):
            for node_id, node in graph.nodes.items():
                if not hasattr(node, 'type') or node.type != 'semantic':
                    continue

                contents = node.metadata.get('contents', []) if hasattr(node, 'metadata') else []
                for content in contents:
                    if not isinstance(content, str):
                        continue

                    equiv_match = re.match(
                        r'equivalence[:\s]+(\w+_\d+)\s+is\s+(\w+)',
                        content, re.IGNORECASE
                    )
                    if equiv_match:
                        entity_id = equiv_match.group(1)
                        real_name = equiv_match.group(2)
                        self._name_mapping[entity_id] = real_name

                        if hasattr(graph, 'reverse_character_mappings'):
                            char_id = graph.reverse_character_mappings.get(entity_id)
                            if char_id:
                                self._name_mapping[char_id] = real_name

                    name_match = re.match(
                        r"(\w+_\d+)'s?\s+name\s+is\s+(\w+)",
                        content, re.IGNORECASE
                    )
                    if name_match:
                        entity_id = name_match.group(1)
                        real_name = name_match.group(2)
                        self._name_mapping[entity_id] = real_name

    def _build_from_nstf_graph(self):
        """Supplement name mappings from nstf_graph."""
        if not self.nstf_graph:
            return

        entity_nodes = self.nstf_graph.get('entity_nodes', {})
        for entity_id, entity_data in entity_nodes.items():
            if isinstance(entity_data, dict):
                name = entity_data.get('name') or entity_data.get('label')
                if name and entity_id not in self._name_mapping:
                    self._name_mapping[entity_id] = name

        char_map = self.nstf_graph.get('character_map', {})
        for char_id, char_info in char_map.items():
            if isinstance(char_info, dict):
                name = char_info.get('name') or char_info.get('display_name')
                if name:
                    self._name_mapping[char_id] = name

    def resolve(self, entity_id: str) -> str:
        """Resolve a single ID to a name."""
        self.initialize()
        return self._name_mapping.get(entity_id, entity_id)

    def resolve_batch(self, entity_ids: List[str]) -> Dict[str, str]:
        """Batch resolve IDs."""
        self.initialize()
        return {eid: self.resolve(eid) for eid in entity_ids}

    def resolve_text(self, text: str) -> str:
        """Replace all entity IDs in text with resolved names."""
        self.initialize()

        pattern = r'<?\b(face_\d+|voice_\d+|character_\d+|person_\d+)\b>?'

        def replacer(match):
            entity_id = match.group(1)

            resolved = self._name_mapping.get(entity_id)
            if resolved and not resolved.startswith(('character_', 'person_', 'face_', 'voice_')):
                return resolved

            if entity_id.startswith('person_'):
                idx = entity_id.replace('person_', '')
                return f'character_{idx}'

            return entity_id

        return re.sub(pattern, replacer, text)

    def add_mapping(self, entity_id: str, name: str):
        """Manually add a mapping."""
        self.initialize()
        self._name_mapping[entity_id] = name

        graph_id = self._get_graph_id()
        cache.set_name_mapping(graph_id, self._name_mapping)

    def get_all_mappings(self) -> Dict[str, str]:
        """Get all current mappings."""
        self.initialize()
        return dict(self._name_mapping)

    def get_character_name(self, character_id: str) -> str:
        """Get character display name, preferring real name."""
        self.initialize()

        name = self._name_mapping.get(character_id)
        if name and not name.startswith('character_'):
            return name

        if character_id.startswith('character_'):
            idx = character_id.replace('character_', '')
            try:
                return f"Person {int(idx) + 1}"
            except ValueError:
                pass

        return character_id

    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract all entity IDs from query text."""
        pattern = r'\b(face_\d+|voice_\d+|character_\d+)\b'
        matches = re.findall(pattern, query)
        return list(set(matches))


def create_resolver(video_graph=None, nstf_graph: Dict = None) -> NameResolver:
    """Factory function to create an initialized NameResolver."""
    resolver = NameResolver(video_graph, nstf_graph)
    resolver.initialize()
    return resolver
