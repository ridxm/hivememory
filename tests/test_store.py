import os
import tempfile

import numpy as np
import pytest

from hivememory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path)


def test_save_and_get_artifact(store):
    artifact_id = store.save_artifact({
        "claim": "python is a programming language",
        "evidence": ["wikipedia"],
        "confidence": 0.95,
        "agent_id": "agent-1",
        "dependencies": [],
    })
    retrieved = store.get_artifact(artifact_id)
    assert retrieved is not None
    assert retrieved["claim"] == "python is a programming language"
    assert retrieved["confidence"] == 0.95
    assert retrieved["agent_id"] == "agent-1"
    assert retrieved["evidence"] == ["wikipedia"]
    assert retrieved["status"] == "active"


def test_get_artifact_missing(store):
    assert store.get_artifact("nonexistent") is None


def test_get_all_artifacts(store):
    store.save_artifact({"claim": "claim one", "agent_id": "a1"})
    store.save_artifact({"claim": "claim two", "agent_id": "a2", "status": "superseded"})
    store.save_artifact({"claim": "claim three", "agent_id": "a1"})

    all_artifacts = store.get_all_artifacts()
    assert len(all_artifacts) == 3

    active = store.get_all_artifacts(status_filter="active")
    assert len(active) == 2

    superseded = store.get_all_artifacts(status_filter="superseded")
    assert len(superseded) == 1
    assert superseded[0]["claim"] == "claim two"


def test_update_artifact_status(store):
    aid = store.save_artifact({"claim": "mutable claim", "agent_id": "a1"})
    store.update_artifact_status(aid, "archived")
    retrieved = store.get_artifact(aid)
    assert retrieved["status"] == "archived"


def test_search_returns_relevant_results(store):
    store.save_artifact({"claim": "the earth orbits the sun", "agent_id": "a1"})
    store.save_artifact({"claim": "water boils at 100 degrees celsius", "agent_id": "a1"})
    store.save_artifact({"claim": "the moon orbits the earth", "agent_id": "a1"})

    query_vec = store.embed("planets and orbits")
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    # each result is (id, score)
    ids = [r[0] for r in results]
    # the orbit-related claims should rank higher than boiling water
    top_artifact = store.get_artifact(ids[0])
    assert "orbit" in top_artifact["claim"]


def test_search_empty_index(store):
    query_vec = store.embed("anything")
    results = store.search(query_vec, top_k=5)
    assert results == []


def test_search_top_k_larger_than_index(store):
    store.save_artifact({"claim": "only one", "agent_id": "a1"})
    query_vec = store.embed("one")
    results = store.search(query_vec, top_k=10)
    assert len(results) == 1


def test_embed_produces_normalized_vector(store):
    vec = store.embed("test sentence")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_save_conflict_and_get_unresolved(store):
    a1 = store.save_artifact({"claim": "the sky is blue", "agent_id": "a1"})
    a2 = store.save_artifact({"claim": "the sky is green", "agent_id": "a2"})

    cid = store.save_conflict({
        "artifact_a_id": a1,
        "artifact_b_id": a2,
        "description": "disagreement about sky color",
    })

    unresolved = store.get_conflicts(resolved=False)
    assert len(unresolved) == 1
    assert unresolved[0]["id"] == cid
    assert unresolved[0]["description"] == "disagreement about sky color"

    resolved = store.get_conflicts(resolved=True)
    assert len(resolved) == 0


def test_resolve_conflict(store):
    a1 = store.save_artifact({"claim": "claim a", "agent_id": "a1"})
    a2 = store.save_artifact({"claim": "claim b", "agent_id": "a2"})
    cid = store.save_conflict({
        "artifact_a_id": a1,
        "artifact_b_id": a2,
        "description": "conflict",
    })

    store.resolve_conflict(cid, resolution="claim a is correct", resolved_by="arbiter")

    unresolved = store.get_conflicts(resolved=False)
    assert len(unresolved) == 0

    resolved = store.get_conflicts(resolved=True)
    assert len(resolved) == 1
    assert resolved[0]["resolution"] == "claim a is correct"
    assert resolved[0]["resolved_by"] == "arbiter"
    assert resolved[0]["resolved_at"] is not None


def test_provenance_edges(store):
    a1 = store.save_artifact({"claim": "source fact", "agent_id": "a1"})
    a2 = store.save_artifact({"claim": "derived fact", "agent_id": "a2"})

    store.add_provenance_edge(a1, a2, "a2")

    dag = store.get_provenance_dag()
    assert len(dag) == 1
    assert dag[0]["source_artifact_id"] == a1
    assert dag[0]["target_artifact_id"] == a2
    assert dag[0]["agent_id"] == "a2"


def test_rebuild_index(store):
    a1 = store.save_artifact({"claim": "cats are animals", "agent_id": "a1"})
    a2 = store.save_artifact({"claim": "dogs are animals", "agent_id": "a1"})

    store.rebuild_index()

    assert store.index.ntotal == 2
    query_vec = store.embed("pets and animals")
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    ids = {r[0] for r in results}
    assert a1 in ids
    assert a2 in ids


def test_persistence_across_instances(tmp_path):
    db_path = str(tmp_path / "persist.db")
    store1 = MemoryStore(db_path=db_path)
    aid = store1.save_artifact({"claim": "persistent fact", "agent_id": "a1"})

    store2 = MemoryStore(db_path=db_path)
    retrieved = store2.get_artifact(aid)
    assert retrieved is not None
    assert retrieved["claim"] == "persistent fact"

    # faiss index should be rebuilt from DB
    assert store2.index.ntotal == 1
    query_vec = store2.embed("persistent")
    results = store2.search(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0][0] == aid
