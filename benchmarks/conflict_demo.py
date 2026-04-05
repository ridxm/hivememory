"""
Conflict detection demo.

Stores artifacts from two agents that intentionally disagree on key claims.
Shows the two-stage pipeline: embedding similarity → LLM verification.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hivememory import HiveMemory, Evidence

DB_PATH = "conflict_demo.db"

# clean slate
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

hive = HiveMemory(db_path=DB_PATH, conflict_threshold=0.7)

# --- agent-1: optimistic analyst ---
agent1_claims = [
    {
        "claim": "The AI code editor market is projected to reach $5 billion by 2026",
        "evidence": [Evidence(source="Grand View Research", content="Market valued at $5B with 35% CAGR", reliability=0.9)],
        "confidence": 0.95,
    },
    {
        "claim": "GitHub Copilot holds 55% market share among AI code editors",
        "evidence": [Evidence(source="Stack Overflow Survey 2025", content="55% of respondents use Copilot", reliability=0.85)],
        "confidence": 0.9,
    },
    {
        "claim": "AI code editors improve developer productivity by 40-55%",
        "evidence": [Evidence(source="Microsoft Research", content="Controlled study showed 55% faster task completion", reliability=0.9)],
        "confidence": 0.95,
    },
    {
        "claim": "Cursor has an NPS score of 72 among professional developers",
        "evidence": [Evidence(source="Developer survey Q4 2025", content="NPS 72, highest in category", reliability=0.8)],
        "confidence": 0.8,
    },
]

# --- agent-2: conservative analyst ---
agent2_claims = [
    {
        "claim": "The AI code editor market is estimated at $2.1 billion in 2026",
        "evidence": [Evidence(source="Gartner Report", content="Market at $2.1B, slower growth than projected", reliability=0.85)],
        "confidence": 0.55,
    },
    {
        "claim": "GitHub Copilot's market share has declined to 35% as competitors gained ground",
        "evidence": [Evidence(source="JetBrains Dev Ecosystem 2025", content="Copilot at 35%, down from 50%", reliability=0.8)],
        "confidence": 0.55,
    },
    {
        "claim": "AI code editors improve developer productivity by 15-25% in real-world settings",
        "evidence": [Evidence(source="University of Zurich study", content="Lab gains of 55% drop to 15-25% in production codebases", reliability=0.85)],
        "confidence": 0.6,
    },
    {
        "claim": "Enterprise adoption of AI code editors reached 78% in 2025",
        "evidence": [Evidence(source="Forrester", content="78% of Fortune 500 have at least one team using AI coding tools", reliability=0.9)],
        "confidence": 0.85,
    },
]

print("=" * 60)
print("CONFLICT DETECTION DEMO")
print("=" * 60)

# track pipeline stats
total_pairs = 0
stage1_candidates = 0
stage2_contradictions = 0

print("\n--- Agent 1 (optimistic analyst) storing findings ---")
for c in agent1_claims:
    art = hive.write(claim=c["claim"], evidence=c["evidence"], confidence=c["confidence"], agent_id="optimistic-analyst")
    print(f"  stored: {c['claim'][:60]}...")

print(f"\nArtifacts in memory: {len(hive.get_all_artifacts())}")
print(f"Conflicts so far: {len(hive.get_conflicts())}")

print("\n--- Agent 2 (conservative analyst) storing findings ---")
for c in agent2_claims:
    before_conflicts = len(hive.get_conflicts())
    existing = hive.get_all_artifacts()

    art = hive.write(claim=c["claim"], evidence=c["evidence"], confidence=c["confidence"], agent_id="conservative-analyst")

    # count pairs compared (this artifact vs all existing)
    pairs_this_round = len(existing)
    total_pairs += pairs_this_round

    new_conflicts = hive.get_conflicts()[before_conflicts:]

    if new_conflicts:
        stage2_contradictions += len(new_conflicts)
        for conf in new_conflicts:
            print(f"  CONFLICT: {c['claim'][:60]}...")
            print(f"    → {conf.description}")
    else:
        print(f"  stored: {c['claim'][:60]}...")

# count stage 1 candidates by checking embedding similarity manually
import numpy as np

all_arts = hive.get_all_artifacts()
agent1_arts = [a for a in all_arts if a.agent_id == "optimistic-analyst"]
agent2_arts = [a for a in all_arts if a.agent_id == "conservative-analyst"]

for a2 in agent2_arts:
    for a1 in agent1_arts:
        if a2.topic_embedding and a1.topic_embedding:
            va = np.array(a2.topic_embedding, dtype=np.float32)
            vb = np.array(a1.topic_embedding, dtype=np.float32)
            sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            if sim > 0.7:
                stage1_candidates += 1

print("\n" + "=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)

all_conflicts = hive.get_conflicts()
print(f"\n  Artifact pairs compared:           {total_pairs}")
print(f"  Stage 1 candidates (sim > 0.7):    {stage1_candidates}")
print(f"  Stage 2 confirmed contradictions:   {len(all_conflicts)}")

print(f"\n  Total artifacts:    {len(all_arts)}")
print(f"  Agent 1 artifacts:  {len(agent1_arts)}")
print(f"  Agent 2 artifacts:  {len(agent2_arts)}")
print(f"  Open conflicts:     {len(all_conflicts)}")

for i, conf in enumerate(all_conflicts):
    print(f"\n  Conflict {i+1}:")
    print(f"    {conf.description}")
    a1 = hive.get_artifact(conf.artifact_ids[0])
    a2 = hive.get_artifact(conf.artifact_ids[1])
    if a1 and a2:
        print(f"    Claim A ({a1.agent_id}, conf={a1.confidence}): {a1.claim[:70]}...")
        print(f"    Claim B ({a2.agent_id}, conf={a2.confidence}): {a2.claim[:70]}...")

# save results
results = {
    "total_pairs_compared": total_pairs,
    "stage1_candidates": stage1_candidates,
    "stage2_contradictions": len(all_conflicts),
    "total_artifacts": len(all_arts),
    "conflicts": [
        {
            "description": c.description,
            "artifact_ids": c.artifact_ids,
            "claims": [
                hive.get_artifact(c.artifact_ids[0]).claim if hive.get_artifact(c.artifact_ids[0]) else "",
                hive.get_artifact(c.artifact_ids[1]).claim if hive.get_artifact(c.artifact_ids[1]) else "",
            ],
        }
        for c in all_conflicts
    ],
}

with open("benchmarks/conflict_demo_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nResults saved to benchmarks/conflict_demo_results.json")

# cleanup
os.remove(DB_PATH)
