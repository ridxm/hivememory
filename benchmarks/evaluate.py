#!/usr/bin/env python3
"""Evaluate baseline vs shared memory outputs using LLM-as-judge (simulated)."""

from __future__ import annotations

from benchmarks.common import BenchmarkResult


def count_redundant_claims(outputs: dict[str, list[str]]) -> int:
    """Count claims that substantially overlap across agents."""
    all_claims = []
    for agent_id, claims in outputs.items():
        for claim in claims:
            all_claims.append((agent_id, claim))

    redundant = 0
    seen_topics: list[tuple[str, set[str]]] = []

    for agent_id, claim in all_claims:
        words = set(claim.lower().split())
        is_redundant = False
        for prev_agent, prev_words in seen_topics:
            if prev_agent != agent_id:
                overlap = len(words & prev_words) / max(len(words | prev_words), 1)
                if overlap > 0.4:
                    is_redundant = True
                    break
        seen_topics.append((agent_id, words))
        if is_redundant:
            redundant += 1

    return redundant


def evaluate_quality(claims: list[str]) -> float:
    """Simulated LLM-as-judge quality score (0-1).

    Scores higher for: diversity of topics, specificity (numbers/names),
    and coverage breadth.
    """
    if not claims:
        return 0.0

    # diversity: unique important words across claims
    all_words: set[str] = set()
    for claim in claims:
        all_words.update(w.lower() for w in claim.split() if len(w) > 4)
    diversity = min(len(all_words) / (len(claims) * 5), 1.0)

    # specificity: claims with numbers or proper nouns score higher
    specific = sum(
        1 for c in claims if any(ch.isdigit() for ch in c) or any(w[0].isupper() for w in c.split()[1:] if w)
    )
    specificity = specific / len(claims)

    # coverage: penalize very similar claims
    unique_enough = len(claims) - count_redundant_claims({"single": claims})
    coverage = unique_enough / len(claims) if claims else 0

    return round(diversity * 0.4 + specificity * 0.3 + coverage * 0.3, 3)


def compare(baseline: BenchmarkResult, shared: BenchmarkResult) -> dict:
    """Compare baseline and shared results."""
    baseline_claims = [c for cs in baseline.agent_outputs.values() for c in cs]
    shared_claims = [c for cs in shared.agent_outputs.values() for c in cs]

    baseline_redundant = count_redundant_claims(baseline.agent_outputs)
    shared_redundant = count_redundant_claims(shared.agent_outputs)

    baseline_quality = evaluate_quality(baseline_claims)
    shared_quality = evaluate_quality(shared_claims)

    token_savings = baseline.total_tokens - shared.total_tokens
    token_savings_pct = (token_savings / baseline.total_tokens * 100) if baseline.total_tokens else 0

    return {
        "baseline": baseline.to_dict(),
        "shared": shared.to_dict(),
        "comparison": {
            "token_savings": token_savings,
            "token_savings_pct": round(token_savings_pct, 1),
            "baseline_redundant_claims": baseline_redundant,
            "shared_redundant_claims": shared_redundant,
            "redundancy_reduction": baseline_redundant - shared_redundant,
            "baseline_quality_score": baseline_quality,
            "shared_quality_score": shared_quality,
            "conflicts_caught": shared.num_conflicts_detected,
            "conflicts_resolved": shared.num_conflicts_resolved,
            "time_baseline": round(baseline.elapsed_seconds, 3),
            "time_shared": round(shared.elapsed_seconds, 3),
        },
    }


if __name__ == "__main__":
    from benchmarks.baseline import run_baseline
    from benchmarks.shared import run_shared

    b = run_baseline(verbose=False)
    s = run_shared(verbose=False)
    result = compare(b, s)

    print("\nComparison:")
    for k, v in result["comparison"].items():
        print(f"  {k}: {v}")
