#!/usr/bin/env python3
"""Real benchmark: 3 agents research with actual LLM calls (gpt-4o-mini).

Baseline: each agent independently researches 3 broad sub-topics.
Shared: agents query hivememory before each call. When prior findings exist,
the agent gets a focused prompt ("here's what we know, find what's missing")
which produces shorter, non-redundant responses.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hivememory import HiveMemory, Evidence, ReasoningArtifact

client = OpenAI()
MODEL = "gpt-4o-mini"

TOPIC = "Competitive Landscape of AI Code Editors in 2026"

# 3 agents, each with 3 sub-queries. some overlap intentionally.
AGENTS = [
    {
        "id": "product-features",
        "role": "You are a product analyst researching AI code editors in 2026.",
        "queries": [
            "Compare the multi-file editing and agentic coding capabilities of Cursor, GitHub Copilot, Claude Code, and Windsurf in 2026. Include specific numbers on file limits, accuracy rates, and speed.",
            "How do Cursor, Copilot, Claude Code, and Windsurf handle context windows and codebase retrieval in 2026? Compare token limits, retrieval methods, and accuracy.",
            "What are the unique advantages of terminal-based AI coding tools (Claude Code, aider, Gemini CLI) vs IDE-based tools (Cursor, Copilot, Windsurf) in 2026?",
        ],
    },
    {
        "id": "pricing-business",
        "role": "You are a business analyst researching AI code editor pricing and market dynamics in 2026.",
        "queries": [
            "What are the pricing tiers and subscription models for Cursor, GitHub Copilot, Claude Code, and Windsurf in 2026? Include specific dollar amounts.",
            "What is the market share and estimated revenue of GitHub Copilot vs Cursor vs Claude Code vs others in 2026?",
            "Compare the multi-file editing and agentic coding capabilities of Cursor, GitHub Copilot, Claude Code, and Windsurf in 2026. Include specific numbers on file limits, accuracy rates, and speed.",
        ],
    },
    {
        "id": "devex-analyst",
        "role": "You are a developer experience researcher studying AI code editor adoption in 2026.",
        "queries": [
            "What is the developer satisfaction (NPS) for Cursor, Copilot, Claude Code, and Windsurf in 2026? How do IDE-native vs terminal tools compare on developer satisfaction?",
            "What is the market share and estimated revenue of GitHub Copilot vs Cursor vs Claude Code vs others in 2026?",
            "What are the unique advantages of terminal-based AI coding tools (Claude Code, aider, Gemini CLI) vs IDE-based tools (Cursor, Copilot, Windsurf) in 2026?",
        ],
    },
]


def llm_call(messages: list[dict]) -> tuple[str, int, int]:
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.7, max_tokens=600)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def extract_findings(text: str) -> tuple[list[dict], int, int]:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract 2-3 key factual claims from the text. "
                "Return a JSON array: [{\"claim\": \"...\", \"evidence\": [{\"source\": \"...\", \"content\": \"...\"}], \"confidence\": 0.0-1.0}]. "
                "Only return the JSON array, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]
    resp_text, in_tok, out_tok = llm_call(messages)
    resp_text = resp_text.strip()
    if resp_text.startswith("```"):
        resp_text = resp_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        findings = json.loads(resp_text)
    except json.JSONDecodeError:
        findings = [{"claim": resp_text[:200], "evidence": [], "confidence": 0.5}]
    return findings, in_tok, out_tok


# ── BASELINE ─────────────────────────────────────────────────────────────

def run_baseline(verbose: bool = True) -> dict:
    results = {
        "mode": "baseline", "agents": {},
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_tokens": 0, "total_llm_calls": 0,
        "all_findings": [],
    }

    if verbose:
        print("=" * 70)
        print("BASELINE: No Shared Memory")
        print("=" * 70)

    start = time.time()

    for agent in AGENTS:
        agent_id = agent["id"]
        ar = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0,
              "artifacts_reused": 0, "artifacts_written": 0, "findings": []}

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"[{agent_id}]")

        for q in agent["queries"]:
            if verbose:
                print(f"  Q: {q[:70]}...")

            # full independent research
            messages = [
                {"role": "system", "content": agent["role"] + " Be specific with numbers. 200-400 words."},
                {"role": "user", "content": q},
            ]
            text, i1, o1 = llm_call(messages)
            ar["input_tokens"] += i1
            ar["output_tokens"] += o1
            ar["llm_calls"] += 1

            findings, i2, o2 = extract_findings(text)
            ar["input_tokens"] += i2
            ar["output_tokens"] += o2
            ar["llm_calls"] += 1

            if verbose:
                print(f"    {i1+i2} in + {o1+o2} out, {len(findings)} findings")
                for f in findings:
                    print(f"      {f['claim'][:75]}")

            ar["findings"].extend(f["claim"] for f in findings)
            ar["artifacts_written"] += len(findings)
            results["all_findings"].extend(findings)

        results["agents"][agent_id] = ar

    results["elapsed_seconds"] = time.time() - start
    for ar in results["agents"].values():
        results["total_input_tokens"] += ar["input_tokens"]
        results["total_output_tokens"] += ar["output_tokens"]
        results["total_llm_calls"] += ar["llm_calls"]
    results["total_tokens"] = results["total_input_tokens"] + results["total_output_tokens"]

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"BASELINE TOTAL: {results['total_tokens']:,} tokens, "
              f"{results['total_llm_calls']} calls, {results['elapsed_seconds']:.1f}s")

    return results


# ── SHARED ───────────────────────────────────────────────────────────────

def run_shared(verbose: bool = True) -> tuple[dict, HiveMemory]:
    db_path = os.path.join(tempfile.mkdtemp(), "bench.db")
    hive = HiveMemory(db_path=db_path, conflict_threshold=0.4)

    results = {
        "mode": "shared", "agents": {},
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_tokens": 0, "total_llm_calls": 0,
        "all_findings": [],
        "token_breakdown": {"original_research": 0, "focused_research": 0,
                            "extraction": 0, "conflict_detection": 0},
        "provenance_edges": [], "artifacts": [], "conflict_details": [],
        "skipped_queries": 0, "focused_queries": 0,
        "conflicts_detected": 0, "conflicts_resolved": 0,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("SHARED: With HiveMemory")
        print("=" * 70)

    start = time.time()

    for agent in AGENTS:
        agent_id = agent["id"]
        ar = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0,
              "artifacts_reused": 0, "artifacts_written": 0, "findings": [],
              "queries_skipped": 0, "queries_focused": 0}

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"[{agent_id}]")

        for q in agent["queries"]:
            if verbose:
                print(f"  Q: {q[:70]}...")

            # check memory
            scored = hive.query_with_scores(q, top_k=5)
            high = [(a, s) for a, s in scored if s >= 0.55]

            if high:
                ar["artifacts_reused"] += len(high)
                context = "\n".join(f"- [{a.agent_id}] {a.claim} (confidence: {a.confidence:.2f})"
                                     for a, s in high)

                if verbose:
                    print(f"    memory: {len(high)} relevant artifacts (top={scored[0][1]:.3f})")
                    for a, s in high[:3]:
                        print(f"      [{a.agent_id}] {a.claim[:60]}... ({s:.3f})")

                # focused call: tell LLM what we know, ask for gaps only
                messages = [
                    {"role": "system", "content": (
                        agent["role"] + "\n\n"
                        "Other analysts have already established these findings:\n"
                        f"{context}\n\n"
                        "DO NOT repeat any of the above. Focus ONLY on aspects not yet covered. "
                        "If everything is well-covered, write a brief 1-2 sentence confirmation. "
                        "Be specific with numbers. 100-200 words max."
                    )},
                    {"role": "user", "content": q},
                ]
                text, i1, o1 = llm_call(messages)
                ar["input_tokens"] += i1
                ar["output_tokens"] += o1
                ar["llm_calls"] += 1
                results["token_breakdown"]["focused_research"] += i1 + o1
                ar["queries_focused"] += 1
                results["focused_queries"] += 1

                if verbose:
                    print(f"    FOCUSED: {i1} in + {o1} out (memory-augmented)")

                # extract findings from the gap-fill response
                findings, i2, o2 = extract_findings(text)
                ar["input_tokens"] += i2
                ar["output_tokens"] += o2
                ar["llm_calls"] += 1
                results["token_breakdown"]["extraction"] += i2 + o2

                reused_ids = [a.id for a, s in high[:3]]

            else:
                # no relevant memory — full research
                if verbose:
                    if scored:
                        print(f"    memory: no strong matches (top={scored[0][1]:.3f})")
                    else:
                        print(f"    memory: empty")

                messages = [
                    {"role": "system", "content": agent["role"] + " Be specific with numbers. 200-400 words."},
                    {"role": "user", "content": q},
                ]
                text, i1, o1 = llm_call(messages)
                ar["input_tokens"] += i1
                ar["output_tokens"] += o1
                ar["llm_calls"] += 1
                results["token_breakdown"]["original_research"] += i1 + o1

                findings, i2, o2 = extract_findings(text)
                ar["input_tokens"] += i2
                ar["output_tokens"] += o2
                ar["llm_calls"] += 1
                results["token_breakdown"]["extraction"] += i2 + o2

                reused_ids = []

                if verbose:
                    print(f"    FULL: {i1+i2} in + {o1+o2} out, {len(findings)} findings")

            # store findings
            for finding in findings:
                evidence = [
                    Evidence(source=e.get("source", ""), content=e.get("content", ""), reliability=0.8)
                    for e in finding.get("evidence", [])
                ]
                art = ReasoningArtifact(
                    claim=finding["claim"], agent_id=agent_id, evidence=evidence,
                    confidence=finding.get("confidence", 0.7),
                    dependencies=reused_ids,
                )
                conflicts = hive.store(art)
                results["artifacts"].append({
                    "id": art.id, "claim": art.claim, "agent_id": agent_id,
                    "confidence": art.confidence, "dependencies": art.dependencies,
                })
                ar["artifacts_written"] += 1
                ar["findings"].append(finding["claim"])
                results["all_findings"].append(finding)

                if conflicts:
                    results["conflicts_detected"] += len(conflicts)
                    for c in conflicts:
                        results["conflict_details"].append({
                            "id": c.id, "description": c.description,
                            "artifact_ids": c.artifact_ids,
                        })
                        if verbose:
                            print(f"      CONFLICT: {c.description}")

                if verbose:
                    print(f"      stored: {finding['claim'][:65]}")

        results["agents"][agent_id] = ar

    # resolve conflicts via LLM
    all_conflicts = hive.get_conflicts()
    for c in all_conflicts:
        arts = [hive.get_artifact(aid) for aid in c.artifact_ids]
        arts = [a for a in arts if a]
        if len(arts) >= 2:
            messages = [
                {"role": "system", "content": (
                    "Two research findings may contradict each other. "
                    "Respond ONLY with JSON: "
                    "{\"is_contradiction\": true/false, \"explanation\": \"...\", \"winner_index\": 0 or 1}"
                )},
                {"role": "user", "content": (
                    f"Claim 1 [{arts[0].agent_id}]: {arts[0].claim}\n"
                    f"Claim 2 [{arts[1].agent_id}]: {arts[1].claim}"
                )},
            ]
            text, v_in, v_out = llm_call(messages)
            results["token_breakdown"]["conflict_detection"] += v_in + v_out
            results["total_llm_calls"] += 1

            try:
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                verdict = json.loads(text)
                winner_idx = int(verdict.get("winner_index", 0) or 0)
                winner = arts[min(winner_idx, len(arts) - 1)]
                reason = verdict.get("explanation", "")
                is_contra = verdict.get("is_contradiction", False)
                hive.resolve_conflict(
                    c.id, winner_id=winner.id,
                    reason=("Contradiction: " if is_contra else "Not contradictory: ") + reason,
                    resolved_by="conflict-detector",
                )
                if is_contra:
                    results["conflicts_resolved"] += 1
                if verbose:
                    label = "CONTRADICTION" if is_contra else "false positive"
                    print(f"  conflict {c.id[:8]}: {label} — {reason[:60]}...")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    # provenance
    for art_info in results["artifacts"]:
        for dep_id in art_info["dependencies"]:
            results["provenance_edges"].append({"from": dep_id, "to": art_info["id"]})

    results["elapsed_seconds"] = time.time() - start
    for ar in results["agents"].values():
        results["total_input_tokens"] += ar["input_tokens"]
        results["total_output_tokens"] += ar["output_tokens"]
        results["total_llm_calls"] += ar["llm_calls"]
    results["total_tokens"] = (results["total_input_tokens"] + results["total_output_tokens"]
                                + results["token_breakdown"]["conflict_detection"])

    total_queries = sum(len(a["queries"]) for a in AGENTS)
    results["reuse_rate"] = results["focused_queries"] / total_queries

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"SHARED TOTAL: {results['total_tokens']:,} tokens, "
              f"{results['total_llm_calls']} calls, {results['elapsed_seconds']:.1f}s")
        print(f"Focused (memory-augmented) queries: {results['focused_queries']}/{total_queries}")
        print(f"Conflicts detected: {results['conflicts_detected']}, "
              f"resolved: {results['conflicts_resolved']}")

    return results, hive


# ── QUALITY EVAL ─────────────────────────────────────────────────────────

def run_quality_eval(baseline_findings: list, shared_findings: list, n_runs: int = 3) -> dict:
    dimensions = ["completeness", "accuracy", "coherence", "contradiction_free"]
    baseline_claims = "\n".join(f"- {f['claim']}" for f in baseline_findings if isinstance(f, dict))
    shared_claims = "\n".join(f"- {f['claim']}" for f in shared_findings if isinstance(f, dict))

    baseline_scores = {d: [] for d in dimensions}
    shared_scores = {d: [] for d in dimensions}
    eval_tokens = 0

    for _ in range(n_runs):
        for label, claims, scores in [
            ("baseline", baseline_claims, baseline_scores),
            ("hivememory", shared_claims, shared_scores),
        ]:
            messages = [
                {"role": "system", "content": (
                    "Rate research findings 1-10. Return ONLY JSON: "
                    "{\"completeness\": N, \"accuracy\": N, \"coherence\": N, \"contradiction_free\": N}"
                )},
                {"role": "user", "content": f"Topic: {TOPIC}\n\nFindings ({label}):\n{claims}"},
            ]
            text, i_tok, o_tok = llm_call(messages)
            eval_tokens += i_tok + o_tok
            try:
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                r = json.loads(text)
                for d in dimensions:
                    scores[d].append(r.get(d, 5))
            except (json.JSONDecodeError, KeyError):
                for d in dimensions:
                    scores[d].append(5)

    return {
        "baseline": {d: round(sum(v) / len(v), 1) for d, v in baseline_scores.items()},
        "shared": {d: round(sum(v) / len(v), 1) for d, v in shared_scores.items()},
        "eval_tokens": eval_tokens, "n_runs": n_runs,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  REAL BENCHMARK: {TOPIC}")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    baseline = run_baseline(verbose=True)
    shared, hive = run_shared(verbose=True)

    print(f"\n{'=' * 70}")
    print("LLM-as-judge quality eval (3 runs)...")
    print("=" * 70)
    quality = run_quality_eval(baseline["all_findings"], shared["all_findings"])

    full = {
        "topic": TOPIC, "model": MODEL,
        "baseline": {
            "agents": baseline["agents"],
            "total_input_tokens": baseline["total_input_tokens"],
            "total_output_tokens": baseline["total_output_tokens"],
            "total_tokens": baseline["total_tokens"],
            "total_llm_calls": baseline["total_llm_calls"],
            "elapsed_seconds": round(baseline["elapsed_seconds"], 2),
        },
        "shared": {
            "agents": shared["agents"],
            "total_input_tokens": shared["total_input_tokens"],
            "total_output_tokens": shared["total_output_tokens"],
            "total_tokens": shared["total_tokens"],
            "total_llm_calls": shared["total_llm_calls"],
            "elapsed_seconds": round(shared["elapsed_seconds"], 2),
            "conflicts_detected": shared["conflicts_detected"],
            "conflicts_resolved": shared["conflicts_resolved"],
            "reuse_rate": shared["reuse_rate"],
            "focused_queries": shared["focused_queries"],
            "token_breakdown": shared["token_breakdown"],
            "provenance_edges": shared["provenance_edges"],
            "artifacts": shared["artifacts"],
            "conflict_details": shared.get("conflict_details", []),
        },
        "quality": quality,
        "comparison": {
            "token_savings": baseline["total_tokens"] - shared["total_tokens"],
            "token_savings_pct": round(
                (baseline["total_tokens"] - shared["total_tokens"])
                / max(baseline["total_tokens"], 1) * 100, 1),
            "llm_call_savings": baseline["total_llm_calls"] - shared["total_llm_calls"],
        },
    }

    path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(path, "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print("=" * 70)
    bl, sh, cmp = full["baseline"], full["shared"], full["comparison"]
    print(f"\n{'Metric':<35} {'Baseline':>12} {'HiveMemory':>12} {'Delta':>10}")
    print("─" * 70)
    print(f"{'Total tokens':<35} {bl['total_tokens']:>12,} {sh['total_tokens']:>12,} {'-' + str(cmp['token_savings_pct']) + '%':>10}")
    print(f"{'Input tokens':<35} {bl['total_input_tokens']:>12,} {sh['total_input_tokens']:>12,}")
    print(f"{'Output tokens':<35} {bl['total_output_tokens']:>12,} {sh['total_output_tokens']:>12,}")
    print(f"{'LLM calls':<35} {bl['total_llm_calls']:>12} {sh['total_llm_calls']:>12} {cmp['llm_call_savings']:>10}")
    print(f"{'Memory-augmented queries':<35} {'0':>12} {sh['focused_queries']:>12}")
    print(f"{'Conflicts detected':<35} {'0':>12} {sh['conflicts_detected']:>12}")
    print(f"{'Conflicts resolved':<35} {'0':>12} {sh['conflicts_resolved']:>12}")
    print(f"{'Reuse rate':<35} {'0%':>12} {sh['reuse_rate']:>11.0%}")
    print(f"{'Time (seconds)':<35} {bl['elapsed_seconds']:>12.1f} {sh['elapsed_seconds']:>12.1f}")
    print("─" * 70)

    print(f"\nQuality (avg of {quality['n_runs']} runs):")
    for d in ["completeness", "accuracy", "coherence", "contradiction_free"]:
        print(f"  {d:<25} baseline: {quality['baseline'][d]:.1f}  hivememory: {quality['shared'][d]:.1f}")


if __name__ == "__main__":
    main()
