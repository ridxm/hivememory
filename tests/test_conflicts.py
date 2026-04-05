from __future__ import annotations

from unittest.mock import MagicMock

from hivememory.artifact import Conflict, Evidence, ReasoningArtifact
from hivememory.conflicts import ConflictDetector, _cosine_similarity


def _make_artifact(
    claim: str,
    agent_id: str = "agent-1",
    embedding: list[float] | None = None,
    evidence: list[Evidence] | None = None,
) -> ReasoningArtifact:
    return ReasoningArtifact(
        claim=claim,
        agent_id=agent_id,
        topic_embedding=embedding or [],
        evidence=evidence or [Evidence(source="test-source", content="test")],
    )


def _fake_store():
    store = MagicMock()
    store._save_artifact = MagicMock()
    store._save_conflict = MagicMock()
    return store


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


class TestConflictDetector:
    def test_no_embedding_returns_empty(self):
        store = _fake_store()
        detector = ConflictDetector(store, llm_client=MagicMock())
        new = _make_artifact("claim", embedding=[])
        result = detector.detect(new, [])
        assert result == []

    def test_low_similarity_skips_llm(self):
        store = _fake_store()
        llm = MagicMock()
        detector = ConflictDetector(store, llm_client=llm)

        new = _make_artifact("claim a", embedding=[1.0, 0.0, 0.0])
        existing = _make_artifact("claim b", embedding=[0.0, 1.0, 0.0])

        result = detector.detect(new, [existing])
        assert result == []
        llm.check_contradiction.assert_not_called()

    def test_contradiction_creates_conflict(self):
        store = _fake_store()
        llm = MagicMock()
        llm.check_contradiction.return_value = "CONTRADICTS"
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        new = _make_artifact("claim a", agent_id="agent-1", embedding=emb)
        existing = _make_artifact("claim b", agent_id="agent-2", embedding=emb)

        result = detector.detect(new, [existing])
        assert len(result) == 1
        assert new.id in result[0].artifact_ids
        assert existing.id in result[0].artifact_ids
        assert new.status == "contested"
        assert existing.status == "contested"
        store._save_conflict.assert_called_once()
        assert store._save_artifact.call_count == 2

    def test_refines_adds_dependency(self):
        store = _fake_store()
        llm = MagicMock()
        llm.check_contradiction.return_value = "REFINES"
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        new = _make_artifact("claim a", embedding=emb)
        existing = _make_artifact("claim b", embedding=emb)

        result = detector.detect(new, [existing])
        assert result == []
        assert existing.id in new.dependencies
        store._save_artifact.assert_called_once()

    def test_supports_no_side_effects(self):
        store = _fake_store()
        llm = MagicMock()
        llm.check_contradiction.return_value = "SUPPORTS"
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        new = _make_artifact("claim a", embedding=emb)
        existing = _make_artifact("claim b", embedding=emb)

        result = detector.detect(new, [existing])
        assert result == []
        store._save_artifact.assert_not_called()
        store._save_conflict.assert_not_called()

    def test_unrelated_no_side_effects(self):
        store = _fake_store()
        llm = MagicMock()
        llm.check_contradiction.return_value = "UNRELATED"
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        new = _make_artifact("claim a", embedding=emb)
        existing = _make_artifact("claim b", embedding=emb)

        result = detector.detect(new, [existing])
        assert result == []
        store._save_artifact.assert_not_called()

    def test_skips_same_artifact(self):
        store = _fake_store()
        llm = MagicMock()
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        art = _make_artifact("claim", embedding=emb)

        result = detector.detect(art, [art])
        assert result == []
        llm.check_contradiction.assert_not_called()

    def test_multiple_existing_artifacts(self):
        store = _fake_store()
        llm = MagicMock()
        llm.check_contradiction.side_effect = ["CONTRADICTS", "SUPPORTS", "CONTRADICTS"]
        detector = ConflictDetector(store, llm_client=llm)

        emb = [1.0, 0.0, 0.0]
        new = _make_artifact("new claim", embedding=emb)
        existings = [
            _make_artifact(f"claim {i}", agent_id=f"agent-{i}", embedding=emb)
            for i in range(3)
        ]

        result = detector.detect(new, existings)
        assert len(result) == 2
        assert llm.check_contradiction.call_count == 3
