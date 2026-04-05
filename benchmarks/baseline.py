#!/usr/bin/env python3
"""Baseline benchmark: 3 agents research with NO shared memory."""

from __future__ import annotations

import random
import time

from benchmarks.common import (
    SUBTOPICS,
    BenchmarkResult,
    simulate_llm_call,
)


def run_baseline(seed: int = 42, verbose: bool = True) -> BenchmarkResult:
    random.seed(seed)
    result = BenchmarkResult(mode="baseline")
    start = time.time()

    if verbose:
        print("=" * 60)
        print("BASELINE: No Shared Memory")
        print("=" * 60)

    all_claims: list[str] = []

    for subtopic in SUBTOPICS:
        agent_id = subtopic["agent_id"]
        query = subtopic["query"]

        if verbose:
            print(f"\n[{agent_id}] Researching: {query}")

        # each agent works independently — no context from other agents
        _, prompt_tok, comp_tok = simulate_llm_call(agent_id, query, context="")
        result.prompt_tokens += prompt_tok
        result.completion_tokens += comp_tok

        if verbose:
            print(f"  LLM call: {prompt_tok} prompt + {comp_tok} completion tokens")

        # agent produces findings but they're siloed
        agent_claims = []
        for finding in subtopic["findings"]:
            agent_claims.append(finding["claim"])
            all_claims.append(finding["claim"])
            if verbose:
                print(f"  Finding: {finding['claim'][:70]}...")

        result.agent_outputs[agent_id] = agent_claims

        # in baseline, each agent also repeats background research on the broader
        # topic since they can't see what others have found
        if subtopic != SUBTOPICS[0]:
            # simulate redundant "catch-up" research
            _, extra_prompt, extra_comp = simulate_llm_call(
                agent_id,
                "background context on AI code editors market",
                context="",
            )
            result.prompt_tokens += extra_prompt
            result.completion_tokens += extra_comp
            if verbose:
                print(f"  Redundant background research: {extra_prompt} + {extra_comp} tokens")

    result.total_tokens = result.prompt_tokens + result.completion_tokens
    result.num_artifacts = len(all_claims)
    result.num_conflicts_detected = 0  # no conflict detection in baseline
    result.reuse_rate = 0.0
    result.elapsed_seconds = time.time() - start

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Baseline total tokens: {result.total_tokens:,}")
        print(f"Artifacts (siloed): {result.num_artifacts}")
        print(f"Conflicts detected: 0 (no detection)")

    return result


if __name__ == "__main__":
    run_baseline()
