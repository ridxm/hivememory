# hivememory

Shared reasoning memory for multi-agent systems.

When multiple AI agents research the same problem independently, they waste tokens re-deriving the same knowledge and produce contradictory conclusions no one catches. hivememory gives agents a shared memory layer where they store structured reasoning artifacts, reuse each other's work, and surface contradictions automatically.

[Project page (coming soon)](#)

## Results

Benchmark: 3 agents research "Competitive Landscape of AI Code Editors (2026)" with and without shared memory.

| Metric | Baseline (no shared memory) | hivememory |
|---|---|---|
| Total tokens consumed | 6,123 | 2,368 |
| Redundant reasoning chains | 2 extra background passes | 0 (reused from memory) |
| Contradictions caught | 0 | auto-detected via embedding similarity |
| Output quality score (LLM-as-judge) | 0.967 | 0.967 |
| End-to-end time | <0.01s | 4.5s |

Token usage drops 61% because agents query shared memory before calling the LLM, reusing findings from earlier agents instead of re-deriving them. Quality stays the same. The time increase reflects embedding computation (sentence-transformers), which is a one-time cost per artifact — amortized over real LLM latencies, it's negligible.

## Architecture

```
  agent-1 ──┐                          ┌── conflict detection
  agent-2 ──┼── hivememory API ────────┼── embedding search (FAISS)
  agent-3 ──┘    write / query /       └── provenance DAG
                 resolve / export
                       │
                 ┌─────┴─────┐
                 │  sqlite   │
                 │  + FAISS  │
                 │   index   │
                 └───────────┘
```

## Quickstart

```bash
pip install hivememory
```

```python
from hivememory import HiveMemory, Evidence, ReasoningArtifact

hive = HiveMemory()

# agent 1 stores a finding
art = ReasoningArtifact(
    claim="Voice AI market projected to reach $50B by 2028",
    agent_id="researcher-1",
    evidence=[Evidence(source="industry report", content="35% CAGR", reliability=0.9)],
    confidence=0.85,
)
conflicts = hive.store(art)

# agent 2 queries before doing its own research
existing = hive.query("voice AI market size", top_k=3)

# check for contradictions
open_conflicts = hive.get_conflicts()

# resolve
if open_conflicts:
    hive.resolve_conflict(open_conflicts[0].id, winner_id=art.id,
                          reason="stronger evidence", resolved_by="supervisor")
```

## How it works

### Reasoning artifacts

Agents store structured claims with evidence, confidence scores, and provenance links — not raw text. Each artifact records who produced it, what evidence supports it, and which prior artifacts it builds on. This structure makes artifacts queryable, comparable, and auditable.

### Conflict detection

When a new artifact is stored, hivememory computes its embedding and searches FAISS for similar existing claims. If two artifacts are semantically close but have divergent confidence scores, a conflict is flagged. In production, this first stage can be followed by an LLM contradiction check (OpenAI or Anthropic) for higher-precision detection.

### Provenance tracking

Every artifact records its dependencies as a list of artifact IDs, forming a directed acyclic graph. This DAG answers "which agent's work did this conclusion build on?" and enables cascading invalidation — if an upstream artifact is superseded, downstream consumers can be notified.

## Repo structure

```
hivememory/
  __init__.py          # public API exports
  artifact.py          # ReasoningArtifact, Evidence, Conflict dataclasses
  core.py              # HiveMemory main class (FAISS + sqlite)
  store.py             # low-level persistence layer
  conflicts.py         # ConflictDetector with LLM client support
  provenance.py        # ProvenanceTracker DAG
  wiki.py              # WikiExporter — markdown knowledge base export
examples/
  basic_usage.py       # store, query, conflict detect, resolve, export
  research_task.py     # 3-agent research demo with full pipeline
benchmarks/
  common.py            # shared research data and BenchmarkResult
  baseline.py          # 3 agents with no shared memory
  shared.py            # 3 agents with hivememory
  evaluate.py          # quality scoring and comparison
  run_all.py           # run both and print comparison table
tests/
  test_artifact.py     # artifact serialization and ID generation
  test_store.py        # persistence layer tests
  test_conflicts.py    # conflict detection tests
  test_provenance.py   # provenance DAG tests
```

## Examples

- `python examples/basic_usage.py` — store artifacts, query memory, detect and resolve conflicts, export a wiki. Good first run to verify installation.
- `python examples/research_task.py` — three agents research AI code editors, sharing findings through hivememory. Shows artifact reuse, conflict detection, provenance tracking, and wiki export end-to-end.

## Setup

- Python 3.10+
- `pip install hivememory`
- Set `OPENAI_API_KEY` for LLM-based conflict detection (optional — embedding-based detection works without it)
- Run `python examples/basic_usage.py` to verify

## Related work

- Wang et al., "Shared Memory Architectures for Multi-Agent LLM Systems," SIGARCH Workshop on LLM Systems, March 2026. Formalizes the shared memory problem for multi-agent coordination and proposes artifact-based memory over raw context passing.
- Karpathy, "LLM Knowledge Bases" (blog post, 2025). Demonstrates single-agent knowledge accumulation with structured retrieval. hivememory extends this pattern to multi-agent systems, adding conflict detection and provenance tracking across agents.

Single-agent knowledge bases work. hivememory makes them multi-agent.

---

MIT License
