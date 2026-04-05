from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from hivememory.artifact import Conflict, Evidence, ReasoningArtifact


class HiveMemory:
    def __init__(
        self,
        db_path: str = "hivememory.db",
        model_name: str = "all-MiniLM-L6-v2",
        conflict_threshold: float = 0.8,
    ):
        self.db_path = db_path
        self.conflict_threshold = conflict_threshold
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self._artifacts: list[ReasoningArtifact] = []
        self._conflicts: list[Conflict] = []
        self._token_log: dict[str, dict[str, int]] = {}
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS conflicts (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )"""
            )

    def _load_from_db(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT data FROM artifacts").fetchall()
            for (data,) in rows:
                artifact = ReasoningArtifact.from_dict(json.loads(data))
                self._artifacts.append(artifact)
                if artifact.topic_embedding:
                    vec = np.array([artifact.topic_embedding], dtype=np.float32)
                    faiss.normalize_L2(vec)
                    self.index.add(vec)

            rows = conn.execute("SELECT data FROM conflicts").fetchall()
            for (data,) in rows:
                self._conflicts.append(Conflict.from_dict(json.loads(data)))

    def _embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _save_artifact(self, artifact: ReasoningArtifact):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (id, data) VALUES (?, ?)",
                (artifact.id, json.dumps(artifact.to_dict())),
            )

    def _save_conflict(self, conflict: Conflict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conflicts (id, data) VALUES (?, ?)",
                (conflict.id, json.dumps(conflict.to_dict())),
            )

    def _detect_conflicts(self, artifact: ReasoningArtifact):
        if not self._artifacts or not artifact.topic_embedding:
            return
        vec = np.array([artifact.topic_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(10, self.index.ntotal)
        if k == 0:
            return
        scores, indices = self.index.search(vec, k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._artifacts):
                continue
            other = self._artifacts[idx]
            if other.id == artifact.id:
                continue
            if score > self.conflict_threshold and abs(artifact.confidence - other.confidence) > 0.3:
                conflict = Conflict(
                    artifact_ids=[artifact.id, other.id],
                    description=(
                        f"High similarity ({score:.2f}) but divergent confidence: "
                        f"{artifact.confidence:.2f} vs {other.confidence:.2f}"
                    ),
                )
                self._conflicts.append(conflict)
                self._save_conflict(conflict)

    def store(self, artifact: ReasoningArtifact) -> list[Conflict]:
        """Store a pre-built artifact. Returns any new conflicts detected."""
        artifact.topic_embedding = self._embed(artifact.claim)
        self._save_artifact(artifact)
        vec = np.array([artifact.topic_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        before = len(self._conflicts)
        self._detect_conflicts(artifact)
        self._artifacts.append(artifact)
        return self._conflicts[before:]

    def write(
        self,
        claim: str,
        evidence: list[Evidence],
        confidence: float,
        agent_id: str,
        dependencies: Optional[list[str]] = None,
    ) -> ReasoningArtifact:
        embedding = self._embed(claim)
        artifact = ReasoningArtifact(
            claim=claim,
            evidence=evidence,
            confidence=confidence,
            agent_id=agent_id,
            dependencies=dependencies or [],
            topic_embedding=embedding,
        )
        self._save_artifact(artifact)
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self._detect_conflicts(artifact)
        self._artifacts.append(artifact)
        return artifact

    def query(self, query: str, top_k: int = 5) -> list[ReasoningArtifact]:
        if self.index.ntotal == 0:
            return []
        vec = np.array([self._embed(query)], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._artifacts):
                results.append(self._artifacts[idx])
        return results

    def get_all_artifacts(self) -> list[ReasoningArtifact]:
        return list(self._artifacts)

    def get_artifact(self, artifact_id: str) -> Optional[ReasoningArtifact]:
        for a in self._artifacts:
            if a.id == artifact_id:
                return a
        return None

    def conflicts(self) -> list[Conflict]:
        return [c for c in self._conflicts if not c.resolved]

    def get_conflicts(self, include_resolved: bool = False) -> list[Conflict]:
        if include_resolved:
            return list(self._conflicts)
        return self.conflicts()

    def resolve_conflict(
        self,
        conflict_id: str,
        winner_id: str = "",
        reason: str = "",
        resolved_by: str = "",
        *,
        winner_artifact_id: str = "",
        agent_id: str = "",
    ):
        return self.resolve(
            conflict_id,
            winner_artifact_id=winner_id or winner_artifact_id,
            reason=reason,
            agent_id=resolved_by or agent_id,
        )

    def resolve(
        self,
        conflict_id: str,
        winner_artifact_id: str = "",
        reason: str = "",
        agent_id: str = "",
    ):
        for conflict in self._conflicts:
            if conflict.id == conflict_id:
                conflict.resolved = True
                conflict.winner_id = winner_artifact_id
                conflict.resolution_reason = reason
                conflict.resolved_by = agent_id
                self._save_conflict(conflict)

                losers = [
                    aid for aid in conflict.artifact_ids if aid != winner_artifact_id
                ]
                for artifact in self._artifacts:
                    if artifact.id in losers:
                        artifact.status = "superseded"
                        self._save_artifact(artifact)
                return
        raise ValueError(f"conflict {conflict_id} not found")

    def provenance(self) -> dict:
        nodes = {a.id: a.to_dict() for a in self._artifacts}
        edges = []
        for a in self._artifacts:
            for dep_id in a.dependencies:
                edges.append({"from": dep_id, "to": a.id})
        return {"nodes": nodes, "edges": edges}

    def export_wiki(self, output_dir: str = "wiki"):
        os.makedirs(output_dir, exist_ok=True)
        index_lines = ["# HiveMemory Knowledge Base\n"]
        for artifact in self._artifacts:
            if artifact.status != "active":
                continue
            slug = artifact.id[:8]
            filename = f"{slug}.md"
            lines = [
                f"# {artifact.claim}\n",
                f"**Confidence:** {artifact.confidence:.2f}  ",
                f"**Agent:** {artifact.agent_id}  ",
                f"**Status:** {artifact.status}\n",
            ]
            if artifact.evidence:
                lines.append("## Evidence\n")
                for e in artifact.evidence:
                    lines.append(
                        f"- **{e.source}** (reliability: {e.reliability:.2f}): {e.content}"
                    )
            if artifact.dependencies:
                lines.append("\n## Dependencies\n")
                for dep_id in artifact.dependencies:
                    lines.append(f"- [{dep_id[:8]}]({dep_id[:8]}.md)")
            lines.append("")
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write("\n".join(lines))
            index_lines.append(f"- [{artifact.claim}]({filename})")

        index_lines.append("")
        with open(os.path.join(output_dir, "index.md"), "w") as f:
            f.write("\n".join(index_lines))

    def log_tokens(
        self, agent_id: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ):
        if agent_id not in self._token_log:
            self._token_log[agent_id] = {"prompt": 0, "completion": 0}
        self._token_log[agent_id]["prompt"] += prompt_tokens
        self._token_log[agent_id]["completion"] += completion_tokens

    def token_savings_estimate(self) -> dict:
        total_prompt = sum(v["prompt"] for v in self._token_log.values())
        total_completion = sum(v["completion"] for v in self._token_log.values())
        total = total_prompt + total_completion
        n_agents = len(self._token_log) or 1
        estimated_without = int(total * (1 + 0.3 * (n_agents - 1)))
        return {
            "total_tokens_used": total,
            "estimated_without_sharing": estimated_without,
            "estimated_tokens_saved": estimated_without - total,
        }

    def stats(self) -> dict:
        total_prompt = sum(v["prompt"] for v in self._token_log.values())
        total_completion = sum(v["completion"] for v in self._token_log.values())
        active = [a for a in self._artifacts if a.status == "active"]
        with_deps = [a for a in self._artifacts if a.dependencies]
        reuse_rate = len(with_deps) / len(self._artifacts) if self._artifacts else 0.0
        return {
            "total_artifacts": len(self._artifacts),
            "active_artifacts": len(active),
            "agents": len(set(a.agent_id for a in self._artifacts)),
            "total_conflicts": len(self._conflicts),
            "unresolved_conflicts": len(self.conflicts()),
            "reuse_rate": reuse_rate,
            "total_tokens": total_prompt + total_completion,
        }
