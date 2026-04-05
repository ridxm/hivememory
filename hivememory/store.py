import json
import sqlite3
import uuid
from datetime import datetime, timezone

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryStore:
    def __init__(self, db_path: str = "hivememory.db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map: list[str] = []
        self._rebuild_index()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                evidence_json TEXT,
                confidence REAL,
                agent_id TEXT,
                dependencies_json TEXT,
                embedding_blob BLOB,
                created_at TEXT,
                status TEXT DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS conflicts (
                id TEXT PRIMARY KEY,
                artifact_a_id TEXT NOT NULL,
                artifact_b_id TEXT NOT NULL,
                description TEXT,
                resolution TEXT,
                resolved_by TEXT,
                resolved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS provenance_edges (
                id TEXT PRIMARY KEY,
                source_artifact_id TEXT NOT NULL,
                target_artifact_id TEXT NOT NULL,
                agent_id TEXT,
                created_at TEXT
            );
        """)
        self.conn.commit()

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def save_artifact(self, artifact: dict) -> str:
        artifact_id = artifact.get("id", str(uuid.uuid4()))
        now = artifact.get("created_at", datetime.now(timezone.utc).isoformat())

        claim = artifact["claim"]
        embedding = artifact.get("embedding")
        if embedding is None:
            embedding = self.embed(claim)
        else:
            embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO artifacts (id, claim, evidence_json, confidence, agent_id,
               dependencies_json, embedding_blob, created_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                claim,
                json.dumps(artifact.get("evidence", [])),
                artifact.get("confidence", 1.0),
                artifact.get("agent_id"),
                json.dumps(artifact.get("dependencies", [])),
                embedding.tobytes(),
                now,
                artifact.get("status", "active"),
            ),
        )
        self.conn.commit()

        self.index.add(np.expand_dims(embedding, 0))
        self._id_map.append(artifact_id)

        return artifact_id

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        query = np.expand_dims(np.array(query_embedding, dtype=np.float32), 0)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def get_artifact(self, artifact_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def get_all_artifacts(self, status_filter: str | None = None) -> list[dict]:
        cur = self.conn.cursor()
        if status_filter:
            cur.execute("SELECT * FROM artifacts WHERE status = ?", (status_filter,))
        else:
            cur.execute("SELECT * FROM artifacts")
        return [self._row_to_artifact(row) for row in cur.fetchall()]

    def update_artifact_status(self, artifact_id: str, new_status: str):
        cur = self.conn.cursor()
        cur.execute("UPDATE artifacts SET status = ? WHERE id = ?", (new_status, artifact_id))
        self.conn.commit()

    def save_conflict(self, conflict: dict) -> str:
        conflict_id = conflict.get("id", str(uuid.uuid4()))
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO conflicts (id, artifact_a_id, artifact_b_id, description,
               resolution, resolved_by, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                conflict_id,
                conflict["artifact_a_id"],
                conflict["artifact_b_id"],
                conflict.get("description"),
                conflict.get("resolution"),
                conflict.get("resolved_by"),
                conflict.get("resolved_at"),
            ),
        )
        self.conn.commit()
        return conflict_id

    def get_conflicts(self, resolved: bool = False) -> list[dict]:
        cur = self.conn.cursor()
        if resolved:
            cur.execute("SELECT * FROM conflicts WHERE resolution IS NOT NULL")
        else:
            cur.execute("SELECT * FROM conflicts WHERE resolution IS NULL")
        return [dict(row) for row in cur.fetchall()]

    def resolve_conflict(self, conflict_id: str, resolution: str, resolved_by: str):
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE conflicts SET resolution = ?, resolved_by = ?, resolved_at = ? WHERE id = ?",
            (resolution, resolved_by, now, conflict_id),
        )
        self.conn.commit()

    def add_provenance_edge(self, source_id: str, target_id: str, agent_id: str) -> str:
        edge_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO provenance_edges (id, source_artifact_id, target_artifact_id,
               agent_id, created_at) VALUES (?, ?, ?, ?, ?)""",
            (edge_id, source_id, target_id, agent_id, now),
        )
        self.conn.commit()
        return edge_id

    def get_provenance_dag(self) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM provenance_edges")
        return [dict(row) for row in cur.fetchall()]

    def rebuild_index(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map = []
        cur = self.conn.cursor()
        cur.execute("SELECT id, embedding_blob FROM artifacts ORDER BY created_at")
        rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            self._id_map.append(row["id"])
            vec = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            self.index.add(np.expand_dims(vec, 0))

    def _row_to_artifact(self, row) -> dict:
        return {
            "id": row["id"],
            "claim": row["claim"],
            "evidence": json.loads(row["evidence_json"]) if row["evidence_json"] else [],
            "confidence": row["confidence"],
            "agent_id": row["agent_id"],
            "dependencies": json.loads(row["dependencies_json"]) if row["dependencies_json"] else [],
            "created_at": row["created_at"],
            "status": row["status"],
        }
