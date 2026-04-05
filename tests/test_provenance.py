from __future__ import annotations

from unittest.mock import MagicMock

from hivememory.artifact import ReasoningArtifact, Evidence
from hivememory.provenance import ProvenanceTracker


def _make_artifact(artifact_id: str, agent_id: str = "agent-1") -> ReasoningArtifact:
    return ReasoningArtifact(
        id=artifact_id,
        claim=f"claim for {artifact_id}",
        agent_id=agent_id,
        evidence=[Evidence(source="src", content="content")],
    )


def _fake_store(artifacts: list[ReasoningArtifact]):
    store = MagicMock()
    store._artifacts = list(artifacts)
    store._save_artifact = MagicMock()
    return store


class TestRecordRead:
    def test_stores_reads_in_buffer(self):
        store = _fake_store([])
        tracker = ProvenanceTracker(store)
        tracker.record_read("agent-1", ["a1", "a2"])
        assert tracker._read_buffer["agent-1"] == ["a1", "a2"]

    def test_multiple_reads_accumulate(self):
        store = _fake_store([])
        tracker = ProvenanceTracker(store)
        tracker.record_read("agent-1", ["a1"])
        tracker.record_read("agent-1", ["a2"])
        assert tracker._read_buffer["agent-1"] == ["a1", "a2"]


class TestRecordWrite:
    def test_links_read_buffer_to_write(self):
        art = _make_artifact("w1")
        store = _fake_store([art])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1", "a2"])
        tracker.record_write("agent-1", "w1")

        edges = tracker._edges
        assert len(edges) == 2
        assert edges[0].source_id == "a1"
        assert edges[0].target_id == "w1"
        assert edges[1].source_id == "a2"
        assert edges[1].target_id == "w1"

    def test_clears_buffer_after_write(self):
        art = _make_artifact("w1")
        store = _fake_store([art])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1"])
        tracker.record_write("agent-1", "w1")
        assert "agent-1" not in tracker._read_buffer

    def test_explicit_read_context(self):
        art = _make_artifact("w1")
        store = _fake_store([art])
        tracker = ProvenanceTracker(store)

        tracker.record_write("agent-1", "w1", read_context=["ctx1"])
        assert len(tracker._edges) == 1
        assert tracker._edges[0].source_id == "ctx1"

    def test_merges_context_and_buffer_without_duplicates(self):
        art = _make_artifact("w1")
        store = _fake_store([art])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1", "a2"])
        tracker.record_write("agent-1", "w1", read_context=["a1", "a3"])

        source_ids = [e.source_id for e in tracker._edges]
        assert source_ids == ["a1", "a3", "a2"]

    def test_syncs_dependencies_on_artifact(self):
        art = _make_artifact("w1")
        store = _fake_store([art])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1"])
        tracker.record_write("agent-1", "w1")

        assert "a1" in art.dependencies
        store._save_artifact.assert_called_once_with(art)

    def test_separate_agent_buffers(self):
        art1 = _make_artifact("w1", agent_id="agent-1")
        art2 = _make_artifact("w2", agent_id="agent-2")
        store = _fake_store([art1, art2])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1"])
        tracker.record_read("agent-2", ["a2"])
        tracker.record_write("agent-1", "w1")

        # agent-1 buffer cleared, agent-2 buffer intact
        assert "agent-1" not in tracker._read_buffer
        assert tracker._read_buffer["agent-2"] == ["a2"]
        assert len(tracker._edges) == 1
        assert tracker._edges[0].source_id == "a1"


class TestGetDag:
    def test_empty_dag(self):
        store = _fake_store([])
        tracker = ProvenanceTracker(store)
        dag = tracker.get_dag()
        assert dag == {"nodes": {}, "edges": []}

    def test_dag_includes_connected_nodes(self):
        a1 = _make_artifact("a1")
        a2 = _make_artifact("a2")
        w1 = _make_artifact("w1")
        store = _fake_store([a1, a2, w1])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1", "a2"])
        tracker.record_write("agent-1", "w1")

        dag = tracker.get_dag()
        assert set(dag["nodes"].keys()) == {"a1", "a2", "w1"}
        assert len(dag["edges"]) == 2


class TestGetLineage:
    def test_no_lineage(self):
        store = _fake_store([])
        tracker = ProvenanceTracker(store)
        assert tracker.get_lineage("unknown") == []

    def test_single_parent(self):
        a1 = _make_artifact("a1")
        w1 = _make_artifact("w1")
        store = _fake_store([a1, w1])
        tracker = ProvenanceTracker(store)

        tracker.record_read("agent-1", ["a1"])
        tracker.record_write("agent-1", "w1")

        lineage = tracker.get_lineage("w1")
        assert lineage == ["a1"]

    def test_multi_level_lineage(self):
        a1 = _make_artifact("a1")
        a2 = _make_artifact("a2")
        a3 = _make_artifact("a3")
        store = _fake_store([a1, a2, a3])
        tracker = ProvenanceTracker(store)

        # a1 -> a2 -> a3
        tracker.record_write("agent-1", "a2", read_context=["a1"])
        tracker.record_write("agent-1", "a3", read_context=["a2"])

        lineage = tracker.get_lineage("a3")
        assert "a2" in lineage
        assert "a1" in lineage
        assert lineage.index("a2") < lineage.index("a1")

    def test_diamond_lineage(self):
        # a1 -> b1, a1 -> b2, b1 -> c1, b2 -> c1
        a1 = _make_artifact("a1")
        b1 = _make_artifact("b1")
        b2 = _make_artifact("b2")
        c1 = _make_artifact("c1")
        store = _fake_store([a1, b1, b2, c1])
        tracker = ProvenanceTracker(store)

        tracker.record_write("agent-1", "b1", read_context=["a1"])
        tracker.record_write("agent-1", "b2", read_context=["a1"])
        tracker.record_write("agent-1", "c1", read_context=["b1", "b2"])

        lineage = tracker.get_lineage("c1")
        assert set(lineage) == {"a1", "b1", "b2"}
