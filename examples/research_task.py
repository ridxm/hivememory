#!/usr/bin/env python3
"""
Multi-agent research demo: competitive landscape of AI code editors.

Three agents research different aspects of AI code editors, sharing findings
through HiveMemory. Demonstrates artifact reuse, conflict detection,
provenance tracking, and wiki export.

Run: python examples/research_task.py
"""

from __future__ import annotations

import os
import random
import time

from hivememory import Evidence, HiveMemory, ReasoningArtifact
from hivememory.wiki import WikiExporter

# ── simulated LLM research (deterministic, no API keys needed) ───────────

RESEARCH_DATA = {
    "market-position": {
        "findings": [
            {
                "claim": "GitHub Copilot holds ~55% market share in AI code assistants as of early 2026",
                "evidence": [
                    ("github blog 2026", "Copilot reached 2.5M paid subscribers in Q1 2026"),
                    ("devsurvey 2026", "55% of professional developers report using Copilot regularly"),
                ],
                "confidence": 0.9,
            },
            {
                "claim": "Cursor has emerged as the leading AI-native IDE with 800K monthly active users",
                "evidence": [
                    ("techcrunch 2026", "Cursor Series C at $3B valuation, 800K MAU"),
                    ("developer forums", "Cursor praised for agentic coding and multi-file editing"),
                ],
                "confidence": 0.85,
            },
            {
                "claim": "Claude Code and Gemini CLI are redefining the terminal-first AI coding category",
                "evidence": [
                    ("anthropic launch", "Claude Code launched as CLI-native coding agent"),
                    ("google devblog", "Gemini CLI integrates with existing terminal workflows"),
                ],
                "confidence": 0.8,
            },
        ],
    },
    "technical-capabilities": {
        "findings": [
            {
                "claim": "Multi-file agentic editing is the key differentiator among top AI code editors in 2026",
                "evidence": [
                    ("benchmark suite", "Agents solving multi-file tasks improved from 15% to 68% accuracy 2024-2026"),
                    ("developer survey", "78% cite multi-file context as most important feature"),
                ],
                "confidence": 0.88,
            },
            {
                "claim": "Context window size is no longer the primary bottleneck; retrieval quality is",
                "evidence": [
                    ("ml research", "200K+ token windows standard, but naive stuffing underperforms RAG"),
                    ("cursor engineering blog", "Codebase indexing and smart retrieval outperform raw context length"),
                ],
                "confidence": 0.82,
            },
            {
                "claim": "GitHub Copilot's code completion accuracy leads the market at 45% first-suggestion acceptance",
                "evidence": [
                    ("copilot metrics 2026", "45.2% acceptance rate across all languages"),
                    ("independent eval", "Copilot leads in single-line completion, trails in multi-file tasks"),
                ],
                "confidence": 0.85,
            },
        ],
    },
    "user-experience": {
        "findings": [
            {
                "claim": "Developer satisfaction is highest for AI-native editors (Cursor, Windsurf) vs plugin-based (Copilot in VS Code)",
                "evidence": [
                    ("nps survey 2026", "Cursor NPS: 72, Windsurf NPS: 65, Copilot NPS: 48"),
                    ("ux research", "AI-native IDEs reduce context switching by 40% vs plugin model"),
                ],
                "confidence": 0.8,
            },
            {
                "claim": "Copilot has the highest developer satisfaction due to VS Code integration",
                "evidence": [
                    ("stackoverflow survey 2026", "Copilot rated most-loved AI tool for 3rd year"),
                    ("enterprise feedback", "IT teams prefer Copilot for existing toolchain compatibility"),
                ],
                "confidence": 0.75,
            },
            {
                "claim": "Terminal-based AI tools (Claude Code, aider) are preferred by senior engineers for complex refactoring",
                "evidence": [
                    ("senior dev survey", "68% of staff+ engineers prefer CLI tools for large refactors"),
                    ("case study", "Terminal agents handle repo-wide changes more reliably than IDE plugins"),
                ],
                "confidence": 0.78,
            },
        ],
    },
}


def simulate_llm_tokens() -> tuple[int, int]:
    """Simulate token usage for a research call."""
    return random.randint(800, 2000), random.randint(300, 1000)


def agent_research(
    hive: HiveMemory,
    agent_id: str,
    topic: str,
    data: dict,
) -> list[ReasoningArtifact]:
    """Simulate an agent researching a topic and storing findings."""
    print(f"\n{'─' * 60}")
    print(f"[{agent_id}] Starting research on: {topic}")

    # step 1: query existing memory for relevant prior work
    existing = hive.query(topic, top_k=3)
    reused_ids = []
    if existing:
        print(f"  Found {len(existing)} relevant existing artifacts:")
        for art in existing:
            print(f"    • [{art.agent_id}] {art.claim[:70]}...")
            reused_ids.append(art.id)
    else:
        print("  No existing artifacts found — starting fresh")

    # step 2: "call LLM" to research (simulated)
    prompt_tok, comp_tok = simulate_llm_tokens()
    hive.log_tokens(agent_id, prompt_tok, comp_tok)
    print(f"  LLM call: {prompt_tok} prompt + {comp_tok} completion tokens")

    # step 3: store findings
    artifacts = []
    for finding in data["findings"]:
        evidence = [
            Evidence(source=src, content=content, reliability=0.8 + random.random() * 0.2)
            for src, content in finding["evidence"]
        ]
        art = ReasoningArtifact(
            claim=finding["claim"],
            agent_id=agent_id,
            evidence=evidence,
            confidence=finding["confidence"],
            dependencies=reused_ids[:2],  # depend on up to 2 prior artifacts
        )
        conflicts = hive.store(art)
        artifacts.append(art)

        status = "✓"
        if conflicts:
            status = f"⚡ {len(conflicts)} conflict(s)"
        print(f"  Stored: {art.claim[:60]}... [{status}]")
        for c in conflicts:
            arts = [hive.get_artifact(aid) for aid in c.artifact_ids]
            claims = [a.claim[:50] for a in arts if a]
            print(f"    CONFLICT: {' vs '.join(claims)}")

    return artifacts


def main():
    print("=" * 60)
    print("HiveMemory — Multi-Agent Research Demo")
    print("Topic: Competitive Landscape of AI Code Editors (2026)")
    print("=" * 60)

    random.seed(42)
    hive = HiveMemory(conflict_threshold=0.3)
    start = time.time()

    # run three agents sequentially (each checks memory before starting)
    all_artifacts = []
    agents = [
        ("market-analyst", "market position and share of AI code editors", "market-position"),
        ("tech-evaluator", "technical capabilities of AI code editors", "technical-capabilities"),
        ("ux-researcher", "user experience and developer satisfaction with AI code editors", "user-experience"),
    ]

    for agent_id, topic, data_key in agents:
        arts = agent_research(hive, agent_id, topic, RESEARCH_DATA[data_key])
        all_artifacts.extend(arts)

    elapsed = time.time() - start

    # ── resolve conflicts ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Conflict Resolution Phase")
    print("=" * 60)

    open_conflicts = hive.get_conflicts()
    print(f"\n{len(open_conflicts)} unresolved conflict(s) found")

    for c in open_conflicts:
        arts = [hive.get_artifact(aid) for aid in c.artifact_ids if hive.get_artifact(aid)]
        if len(arts) >= 2:
            # pick higher confidence as winner
            winner = max(arts, key=lambda a: a.confidence)
            loser = [a for a in arts if a.id != winner.id][0]
            hive.resolve_conflict(
                c.id,
                winner_id=winner.id,
                reason=f"Higher confidence ({winner.confidence:.2f} vs {loser.confidence:.2f})",
                resolved_by="supervisor",
            )
            print(f"  Resolved: kept [{winner.agent_id}] {winner.claim[:50]}...")

    # ── export wiki ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Exporting Wiki")
    print("=" * 60)

    wiki_dir = os.path.join(os.path.dirname(__file__), "wiki_output")
    WikiExporter(hive).export(wiki_dir)
    print(f"\nWiki exported to: {wiki_dir}/")
    for fname in sorted(os.listdir(wiki_dir)):
        size = os.path.getsize(os.path.join(wiki_dir, fname))
        print(f"  {fname} ({size} bytes)")

    # ── final stats ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Final Statistics")
    print("=" * 60)

    stats = hive.stats()
    savings = hive.token_savings_estimate()

    print(f"\n  Total artifacts:      {stats['total_artifacts']}")
    print(f"  Active artifacts:     {stats['active_artifacts']}")
    print(f"  Agents:               {stats['agents']}")
    print(f"  Conflicts found:      {stats['total_conflicts']}")
    print(f"  Conflicts resolved:   {stats['total_conflicts'] - stats['unresolved_conflicts']}")
    print(f"  Reuse rate:           {stats['reuse_rate']:.0%}")
    print(f"  Total tokens used:    {stats['total_tokens']:,}")
    print(f"  Est. tokens saved:    {savings['estimated_tokens_saved']:,}")
    print(f"  Est. without sharing: {savings['estimated_without_sharing']:,}")
    print(f"  Time elapsed:         {elapsed:.2f}s")


if __name__ == "__main__":
    main()
