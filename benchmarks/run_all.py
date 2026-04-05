#!/usr/bin/env python3
"""Run baseline and shared benchmarks, print comparison table."""

from __future__ import annotations

import sys
import os

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.baseline import run_baseline
from benchmarks.shared import run_shared
from benchmarks.evaluate import compare


def print_table(results: dict) -> None:
    comp = results["comparison"]
    bl = results["baseline"]
    sh = results["shared"]

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON: Baseline (no memory) vs HiveMemory (shared)")
    print("=" * 70)

    rows = [
        ("Total tokens", f"{bl['total_tokens']:,}", f"{sh['total_tokens']:,}",
         f"-{comp['token_savings_pct']}%"),
        ("Prompt tokens", f"{bl['prompt_tokens']:,}", f"{sh['prompt_tokens']:,}", ""),
        ("Completion tokens", f"{bl['completion_tokens']:,}", f"{sh['completion_tokens']:,}", ""),
        ("Artifacts produced", str(bl["num_artifacts"]), str(sh["num_artifacts"]), ""),
        ("Redundant claims", str(comp["baseline_redundant_claims"]),
         str(comp["shared_redundant_claims"]),
         f"-{comp['redundancy_reduction']}"),
        ("Conflicts detected", "0 (none)", str(comp["conflicts_caught"]), ""),
        ("Conflicts resolved", "0", str(comp["conflicts_resolved"]), ""),
        ("Reuse rate", "0%", f"{sh['reuse_rate']:.0%}", ""),
        ("Quality score", str(comp["baseline_quality_score"]),
         str(comp["shared_quality_score"]), ""),
        ("Time (seconds)", str(comp["time_baseline"]), str(comp["time_shared"]), ""),
    ]

    header = f"{'Metric':<25} {'Baseline':>15} {'HiveMemory':>15} {'Delta':>10}"
    print(f"\n{header}")
    print("─" * 70)
    for label, base_val, shared_val, delta in rows:
        print(f"{label:<25} {base_val:>15} {shared_val:>15} {delta:>10}")
    print("─" * 70)

    print(f"\n  Token savings: {comp['token_savings']:,} tokens ({comp['token_savings_pct']}%)")
    print(f"  Redundancy reduction: {comp['redundancy_reduction']} fewer redundant claims")
    print(f"  Conflicts caught: {comp['conflicts_caught']} (0 in baseline)")
    print()


def main():
    print("Running baseline benchmark (no shared memory)...")
    baseline = run_baseline(seed=42, verbose=True)

    print("\n\n")

    print("Running shared benchmark (with HiveMemory)...")
    shared = run_shared(seed=42, verbose=True)

    results = compare(baseline, shared)
    print_table(results)


if __name__ == "__main__":
    main()
