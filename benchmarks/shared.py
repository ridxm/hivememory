#!/usr/bin/env python3
"""Shared memory benchmark: 3 agents research WITH hivememory."""

from __future__ import annotations

import random
import time

from hivememory import Evidence, HiveMemory, ReasoningArtifact

from benchmarks.common import (
    SUBTOPICS,
    BenchmarkResult,
    simulate_llm_call,
)


def run_shared(seed: int = 42, verbose: bool = True) -> BenchmarkResult:
    random.seed(seed)
    result = BenchmarkResult(mode="shared")
    hive = HiveMemory(conflict_threshold=0.3)
    start = time.time()

    if verbose:
        print("=" * 60)
        print("SHARED: With HiveMemory")
        print("=" * 60)

    for subtopic in SUBTOPICS:
        agent_id = subtopic["agent_id"]
        query = subtopic["query"]

        if verbose:
            print(f"\n[{agent_id}] Researching: {query}")

        # query shared memory for existing relevant work
        existing = hive.query(query, top_k=3)
        context = ""
        reused_ids = []
        if existing:
            context = " ".join(a.claim for a in existing)
            reused_ids = [a.id for a in existing]
            if verbose:
                print(f"  Reusing {len(existing)} existing artifacts from memory")
                for a in existing:
                    print(f"    • [{a.agent_id}] {a.claim[:60]}...")

        # LLM call with context from shared memory (reduces tokens)
        _, prompt_tok, comp_tok = simulate_llm_call(agent_id, query, context=context)
        result.prompt_tokens += prompt_tok
        result.completion_tokens += comp_tok
        hive.log_tokens(agent_id, prompt_tok, comp_tok)

        if verbose:
            print(f"  LLM call: {prompt_tok} prompt + {comp_tok} completion tokens")

        # store findings in shared memory
        agent_claims = []
        for finding in subtopic["findings"]:
            evidence = [
                Evidence(source=src, content=content, reliability=0.8 + random.random() * 0.2)
                for src, content in finding["evidence"]
            ]
            art = ReasoningArtifact(
                claim=finding["claim"],
                agent_id=agent_id,
                evidence=evidence,
                confidence=finding["confidence"],
                dependencies=reused_ids[:2],
            )
            conflicts = hive.store(art)
            agent_claims.append(finding["claim"])

            if verbose:
                status = "stored"
                if conflicts:
                    status = f"stored, {len(conflicts)} conflict(s) detected"
                    result.num_conflicts_detected += len(conflicts)
                print(f"  {finding['claim'][:60]}... [{status}]")

        result.agent_outputs[agent_id] = agent_claims

    # resolve conflicts
    open_conflicts = hive.get_conflicts()
    for c in open_conflicts:
        arts = [hive.get_artifact(aid) for aid in c.artifact_ids if hive.get_artifact(aid)]
        if len(arts) >= 2:
            winner = max(arts, key=lambda a: a.confidence)
            hive.resolve_conflict(
                c.id,
                winner_id=winner.id,
                reason=f"Higher confidence ({winner.confidence:.2f})",
                resolved_by="supervisor",
            )
            result.num_conflicts_resolved += 1

    stats = hive.stats()
    result.total_tokens = result.prompt_tokens + result.completion_tokens
    result.num_artifacts = stats["total_artifacts"]
    result.reuse_rate = stats["reuse_rate"]
    result.elapsed_seconds = time.time() - start

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Shared total tokens: {result.total_tokens:,}")
        print(f"Artifacts: {result.num_artifacts}")
        print(f"Conflicts detected: {result.num_conflicts_detected}")
        print(f"Conflicts resolved: {result.num_conflicts_resolved}")
        print(f"Reuse rate: {result.reuse_rate:.0%}")

    return result


if __name__ == "__main__":
    run_shared()
