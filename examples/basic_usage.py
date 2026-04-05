#!/usr/bin/env python3
"""Basic usage of HiveMemory: store, query, detect conflicts, resolve, export."""

import tempfile

from hivememory import Evidence, HiveMemory, ReasoningArtifact
from hivememory.wiki import WikiExporter


def main():
    print("=" * 60)
    print("HiveMemory — Basic Usage Demo")
    print("=" * 60)

    hive = HiveMemory()

    # ── agent 1: researcher finds voice AI market data ───────────
    a1 = ReasoningArtifact(
        claim="The voice AI market is projected to reach $50B by 2028",
        agent_id="market-researcher",
        evidence=[
            Evidence(
                source="industry report 2025",
                content="Voice AI market valued at $12B in 2024, growing at 35% CAGR",
                reliability=0.9,
            ),
            Evidence(
                source="analyst forecast",
                content="Enterprise adoption driving rapid growth in voice assistants",
                reliability=0.8,
            ),
        ],
        confidence=0.85,
    )
    conflicts = hive.store(a1)
    print(f"\n[market-researcher] stored artifact: {a1.claim}")
    print(f"  conflicts: {len(conflicts)}")

    # ── agent 2: tech analyst finds competing claim ──────────────
    a2 = ReasoningArtifact(
        claim="Voice AI market growth is slowing, projected $30B by 2028",
        agent_id="tech-analyst",
        evidence=[
            Evidence(
                source="gartner 2025",
                content="Voice AI hype cycle entering trough of disillusionment",
                reliability=0.85,
            ),
            Evidence(
                source="adoption survey",
                content="Consumer voice assistant usage plateauing in key markets",
                reliability=0.7,
            ),
        ],
        confidence=0.75,
    )
    conflicts = hive.store(a2)
    print(f"\n[tech-analyst] stored artifact: {a2.claim}")
    print(f"  conflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"  → {c.description}")

    # ── agent 3: use case specialist (depends on agent 1) ────────
    a3 = ReasoningArtifact(
        claim="Healthcare is the fastest-growing vertical for voice AI adoption",
        agent_id="use-case-specialist",
        evidence=[
            Evidence(
                source="health tech quarterly",
                content="Voice-powered clinical documentation reducing admin time by 40%",
                reliability=0.9,
            ),
        ],
        confidence=0.9,
        dependencies=[a1.id],  # builds on market researcher's work
    )
    conflicts = hive.store(a3)
    print(f"\n[use-case-specialist] stored artifact: {a3.claim}")
    print(f"  depends on: {a1.id[:8]}...")
    print(f"  conflicts: {len(conflicts)}")

    # ── query the memory ─────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Querying: 'voice AI market size growth'")
    results = hive.query("voice AI market size growth", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.agent_id}] {r.claim} (conf={r.confidence:.2f})")

    # ── resolve the conflict ─────────────────────────────────────
    open_conflicts = hive.get_conflicts()
    print(f"\n{len(open_conflicts)} unresolved conflict(s)")

    if open_conflicts:
        c = open_conflicts[0]
        print(f"Resolving conflict {c.id[:8]}...")
        hive.resolve_conflict(
            conflict_id=c.id,
            winner_id=a1.id,
            reason="Higher confidence source with multiple corroborating evidence",
            resolved_by="supervisor-agent",
        )
        print(f"  winner: {a1.agent_id} — {a1.claim[:50]}...")
        print(f"  remaining open conflicts: {len(hive.get_conflicts())}")

    # ── export wiki ──────────────────────────────────────────────
    print("\n" + "-" * 60)
    with tempfile.TemporaryDirectory() as td:
        WikiExporter(hive).export(td)
        print(f"Wiki exported to {td}/")
        import os

        for fname in sorted(os.listdir(td)):
            path = os.path.join(td, fname)
            size = os.path.getsize(path)
            print(f"  {fname} ({size} bytes)")

        # show INDEX.md content
        print(f"\n{'─' * 40}")
        print("INDEX.md preview:")
        print("─" * 40)
        with open(os.path.join(td, "INDEX.md")) as f:
            print(f.read()[:800])

    # ── token savings estimate ───────────────────────────────────
    # simulate some token usage
    hive.log_tokens("market-researcher", prompt_tokens=1200, completion_tokens=800)
    hive.log_tokens("tech-analyst", prompt_tokens=1500, completion_tokens=600)
    hive.log_tokens("use-case-specialist", prompt_tokens=900, completion_tokens=400)

    print("\n" + "-" * 60)
    print("Token Savings Estimate")
    print("-" * 60)
    savings = hive.token_savings_estimate()
    for k, v in savings.items():
        print(f"  {k}: {v}")

    print("\n" + "-" * 60)
    print("Stats")
    print("-" * 60)
    for k, v in hive.stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
