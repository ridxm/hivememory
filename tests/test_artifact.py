import uuid
from datetime import datetime, timezone

from hivememory.artifact import Conflict, Evidence, ReasoningArtifact


class TestEvidence:
    def test_to_dict(self):
        e = Evidence(source="paper", content="supports claim", reliability=0.9)
        d = e.to_dict()
        assert d == {"source": "paper", "content": "supports claim", "reliability": 0.9}

    def test_from_dict(self):
        d = {"source": "web", "content": "data point", "reliability": 0.7}
        e = Evidence.from_dict(d)
        assert e.source == "web"
        assert e.content == "data point"
        assert e.reliability == 0.7

    def test_from_dict_default_reliability(self):
        e = Evidence.from_dict({"source": "x", "content": "y"})
        assert e.reliability == 1.0

    def test_roundtrip(self):
        e = Evidence(source="api", content="result", reliability=0.5)
        assert Evidence.from_dict(e.to_dict()) == e


class TestReasoningArtifact:
    def test_id_generation(self):
        a = ReasoningArtifact(claim="test", agent_id="agent-1")
        uuid.UUID(a.id)  # validates format

    def test_unique_ids(self):
        a1 = ReasoningArtifact(claim="test", agent_id="agent-1")
        a2 = ReasoningArtifact(claim="test", agent_id="agent-1")
        assert a1.id != a2.id

    def test_defaults(self):
        a = ReasoningArtifact(claim="x", agent_id="a")
        assert a.confidence == 1.0
        assert a.status == "active"
        assert a.evidence == []
        assert a.dependencies == []
        assert a.topic_embedding == []
        assert isinstance(a.created_at, datetime)

    def test_to_dict(self):
        a = ReasoningArtifact(
            claim="sky is blue",
            agent_id="agent-1",
            confidence=0.95,
            evidence=[Evidence(source="observation", content="looked up")],
        )
        d = a.to_dict()
        assert d["claim"] == "sky is blue"
        assert d["agent_id"] == "agent-1"
        assert d["confidence"] == 0.95
        assert len(d["evidence"]) == 1
        assert d["status"] == "active"
        assert "id" in d
        assert "created_at" in d

    def test_from_dict(self):
        now = datetime.now(timezone.utc)
        d = {
            "id": "abc-123",
            "claim": "water is wet",
            "agent_id": "agent-2",
            "confidence": 0.8,
            "evidence": [{"source": "test", "content": "yes"}],
            "dependencies": ["dep-1"],
            "topic_embedding": [0.1, 0.2],
            "created_at": now.isoformat(),
            "status": "contested",
        }
        a = ReasoningArtifact.from_dict(d)
        assert a.id == "abc-123"
        assert a.claim == "water is wet"
        assert a.status == "contested"
        assert len(a.evidence) == 1
        assert a.dependencies == ["dep-1"]
        assert a.topic_embedding == [0.1, 0.2]

    def test_roundtrip(self):
        a = ReasoningArtifact(
            claim="roundtrip test",
            agent_id="agent-3",
            confidence=0.6,
            evidence=[Evidence(source="s", content="c", reliability=0.5)],
            dependencies=["d1", "d2"],
            topic_embedding=[0.1, 0.2, 0.3],
        )
        restored = ReasoningArtifact.from_dict(a.to_dict())
        assert restored.claim == a.claim
        assert restored.agent_id == a.agent_id
        assert restored.confidence == a.confidence
        assert restored.dependencies == a.dependencies
        assert restored.topic_embedding == a.topic_embedding
        assert restored.status == a.status
        assert len(restored.evidence) == len(a.evidence)


class TestConflict:
    def test_id_generation(self):
        c = Conflict(artifact_ids=["a", "b"], description="mismatch")
        uuid.UUID(c.id)

    def test_defaults(self):
        c = Conflict()
        assert c.resolved is False
        assert c.winner_id is None
        assert c.resolution_reason is None
        assert c.resolved_by is None

    def test_roundtrip(self):
        c = Conflict(
            artifact_ids=["x", "y"],
            description="conflict",
            resolved=True,
            winner_id="x",
            resolution_reason="higher confidence",
            resolved_by="arbiter",
        )
        restored = Conflict.from_dict(c.to_dict())
        assert restored.id == c.id
        assert restored.artifact_ids == c.artifact_ids
        assert restored.resolved is True
        assert restored.winner_id == "x"
        assert restored.resolution_reason == "higher confidence"
