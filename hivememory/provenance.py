from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ProvenanceEdge:
    source_id: str
    target_id: str
    relation: str

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
        }


class ProvenanceTracker:
    def __init__(self, store):
        self.store = store
        self._read_buffer: dict[str, list[str]] = defaultdict(list)
        self._edges: list[ProvenanceEdge] = []

    def record_read(self, agent_id: str, artifact_ids: list[str]):
        self._read_buffer[agent_id].extend(artifact_ids)

    def record_write(
        self,
        agent_id: str,
        artifact_id: str,
        read_context: list[str] | None = None,
    ):
        sources = list(read_context or [])

        # merge in anything from the read buffer for this agent
        if agent_id in self._read_buffer:
            for aid in self._read_buffer[agent_id]:
                if aid not in sources:
                    sources.append(aid)
            del self._read_buffer[agent_id]

        # create edges from each source to the new artifact
        for source_id in sources:
            edge = ProvenanceEdge(
                source_id=source_id,
                target_id=artifact_id,
                relation="derived_from",
            )
            self._edges.append(edge)

        # also sync dependencies on the artifact itself
        for art in self.store._artifacts:
            if art.id == artifact_id:
                for source_id in sources:
                    if source_id not in art.dependencies:
                        art.dependencies.append(source_id)
                self.store._save_artifact(art)
                break

    def get_dag(self) -> dict:
        node_ids = set()
        for edge in self._edges:
            node_ids.add(edge.source_id)
            node_ids.add(edge.target_id)

        nodes = {}
        for art in self.store._artifacts:
            if art.id in node_ids:
                nodes[art.id] = art.to_dict()

        return {
            "nodes": nodes,
            "edges": [e.to_dict() for e in self._edges],
        }

    def get_lineage(self, artifact_id: str) -> list[str]:
        # build reverse adjacency: target -> [sources]
        parents: dict[str, list[str]] = defaultdict(list)
        for edge in self._edges:
            parents[edge.target_id].append(edge.source_id)

        # BFS backwards from artifact_id
        visited = []
        queue = parents.get(artifact_id, [])[:]
        seen = set(queue)
        while queue:
            current = queue.pop(0)
            visited.append(current)
            for parent in parents.get(current, []):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)

        return visited
