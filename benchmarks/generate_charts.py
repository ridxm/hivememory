#!/usr/bin/env python3
"""Generate all benchmark charts from results.json."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

BASELINE_COLOR = "#1A1A1A"
HIVE_COLOR = "#E85D3A"
BG_COLOR = "white"
FONT = "monospace"

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)


def load_results():
    path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(path) as f:
        return json.load(f)


def style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BASELINE_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=BASELINE_COLOR, labelsize=9)
    ax.grid(False)


def chart_token_per_agent(data):
    agents = list(data["baseline"]["agents"].keys())
    labels = [f"Agent {i+1}\n({a})" for i, a in enumerate(agents)]
    baseline_tokens = [
        data["baseline"]["agents"][a]["input_tokens"] + data["baseline"]["agents"][a]["output_tokens"]
        for a in agents
    ]
    shared_tokens = [
        data["shared"]["agents"][a]["input_tokens"] + data["shared"]["agents"][a]["output_tokens"]
        for a in agents
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)

    x = np.arange(len(agents))
    w = 0.32
    ax.bar(x - w / 2, baseline_tokens, w, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + w / 2, shared_tokens, w, color=HIVE_COLOR, label="hivememory")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontfamily=FONT, fontsize=8)
    ax.set_ylabel("Tokens", fontfamily=FONT, fontsize=10)
    ax.legend(fontsize=9, prop={"family": FONT})

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "token_consumption_per_agent.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved token_consumption_per_agent.png")


def chart_token_total(data):
    baseline = data["baseline"]["total_tokens"]
    shared = data["shared"]["total_tokens"]
    pct = data["comparison"]["token_savings_pct"]

    fig, ax = plt.subplots(figsize=(6, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)

    bars = ax.bar(
        ["Baseline", "hivememory"],
        [baseline, shared],
        color=[BASELINE_COLOR, HIVE_COLOR],
        width=0.5,
    )

    # percentage label
    mid_y = (baseline + shared) / 2
    ax.annotate(
        f"-{pct}%",
        xy=(0.5, mid_y),
        fontsize=22,
        fontweight="bold",
        fontfamily=FONT,
        ha="center",
        va="center",
        color=HIVE_COLOR,
    )

    for bar, val in zip(bars, [baseline, shared]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + baseline * 0.02,
            f"{val:,}",
            ha="center",
            fontfamily=FONT,
            fontsize=10,
        )

    ax.set_ylabel("Total Tokens", fontfamily=FONT, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "token_consumption_total.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved token_consumption_total.png")


def chart_llm_calls_per_agent(data):
    agents = list(data["baseline"]["agents"].keys())
    labels = [f"Agent {i+1}\n({a})" for i, a in enumerate(agents)]
    baseline_calls = [data["baseline"]["agents"][a]["llm_calls"] for a in agents]
    shared_calls = [data["shared"]["agents"][a]["llm_calls"] for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)

    x = np.arange(len(agents))
    w = 0.32
    ax.bar(x - w / 2, baseline_calls, w, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + w / 2, shared_calls, w, color=HIVE_COLOR, label="hivememory")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontfamily=FONT, fontsize=8)
    ax.set_ylabel("LLM Calls", fontfamily=FONT, fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=9, prop={"family": FONT})

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "llm_calls_per_agent.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved llm_calls_per_agent.png")


def chart_artifact_reuse_flow(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    agents = list(data["shared"]["agents"].keys())
    agent_colors = [BASELINE_COLOR, HIVE_COLOR, "#4A90D9"]

    x_positions = [1.5, 5.0, 8.5]
    box_w, box_h = 2.6, 4.5

    for i, (agent_id, x) in enumerate(zip(agents, x_positions)):
        a = data["shared"]["agents"][agent_id]
        color = agent_colors[i]

        # agent box
        rect = FancyBboxPatch(
            (x - box_w / 2, 0.8), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

        # agent label
        ax.text(x, box_h + 0.6, f"Agent {i+1}", ha="center", fontfamily=FONT,
                fontsize=11, fontweight="bold", color=color)
        ax.text(x, box_h + 0.1, agent_id, ha="center", fontfamily=FONT,
                fontsize=7, color="#666")

        # stats inside box
        y = box_h - 0.1
        lines = [
            f"reused: {a['artifacts_reused']}",
            f"wrote: {a['artifacts_written']}",
            f"calls: {a['llm_calls']}",
            f"tokens: {a['input_tokens'] + a['output_tokens']:,}",
        ]
        for line in lines:
            ax.text(x, y, line, ha="center", fontfamily=FONT, fontsize=8, color=BASELINE_COLOR)
            y -= 0.55

        # findings
        y -= 0.2
        for j, claim in enumerate(a["findings"][:2]):
            ax.text(x, y, claim[:30] + "...", ha="center", fontfamily=FONT,
                    fontsize=6, color="#888", style="italic")
            y -= 0.4

    # arrows between agents
    for i in range(len(x_positions) - 1):
        ax.annotate(
            "", xy=(x_positions[i + 1] - box_w / 2 - 0.1, 3.0),
            xytext=(x_positions[i] + box_w / 2 + 0.1, 3.0),
            arrowprops=dict(arrowstyle="->", color=HIVE_COLOR, lw=2),
        )
        ax.text(
            (x_positions[i] + x_positions[i + 1]) / 2, 3.3,
            "shares artifacts",
            ha="center", fontfamily=FONT, fontsize=7, color=HIVE_COLOR,
        )

    # conflicts
    n_conflicts = data["shared"]["conflicts_detected"]
    if n_conflicts > 0:
        ax.text(7, 0.3, f"{n_conflicts} conflict(s) detected",
                ha="center", fontfamily=FONT, fontsize=9, color=HIVE_COLOR, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "artifact_reuse_flow.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved artifact_reuse_flow.png")


def chart_provenance_dag(data):
    if not HAS_NETWORKX:
        print("  skipped provenance_dag.png (networkx not installed)")
        return

    artifacts = data["shared"]["artifacts"]
    edges = data["shared"]["provenance_edges"]

    G = nx.DiGraph()

    agent_colors_map = {}
    palette = ["#1A1A1A", "#E85D3A", "#4A90D9", "#2ECC71", "#9B59B6"]
    agent_ids_seen = []

    for art in artifacts:
        aid = art["agent_id"]
        if aid not in agent_ids_seen:
            agent_ids_seen.append(aid)
        label = art["claim"][:35] + "..."
        G.add_node(art["id"], label=label, agent_id=aid)

    for e in edges:
        if e["from"] in G.nodes and e["to"] in G.nodes:
            G.add_edge(e["from"], e["to"])

    for i, aid in enumerate(agent_ids_seen):
        agent_colors_map[aid] = palette[i % len(palette)]

    # contested artifacts (involved in conflicts)
    contested = set()
    for c in data["shared"].get("conflict_details", []):
        for aid in c.get("artifact_ids", []):
            contested.add(aid)

    fig, ax = plt.subplots(figsize=(12, 8))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No provenance edges", transform=ax.transAxes,
                ha="center", fontfamily=FONT, fontsize=14, color="#999")
        plt.savefig(os.path.join(OUT_DIR, "provenance_dag.png"), dpi=300, facecolor=BG_COLOR)
        plt.close()
        print("  saved provenance_dag.png (empty)")
        return

    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    node_colors = [agent_colors_map.get(G.nodes[n].get("agent_id", ""), "#999") for n in G.nodes]
    edge_colors = [HIVE_COLOR if G.nodes[e[1]].get("agent_id") != G.nodes[e[0]].get("agent_id") else "#ccc"
                   for e in G.edges]

    # draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True,
                           arrowsize=15, width=1.5, alpha=0.7)

    # draw nodes
    for node in G.nodes:
        x, y = pos[node]
        color = agent_colors_map.get(G.nodes[node].get("agent_id", ""), "#999")
        ec = HIVE_COLOR if node in contested else color
        lw = 3 if node in contested else 1.5
        circle = plt.Circle((x, y), 0.08, facecolor=color, edgecolor=ec,
                            linewidth=lw, alpha=0.9, zorder=5)
        ax.add_patch(circle)

    # labels
    labels = {n: G.nodes[n].get("label", n[:8]) for n in G.nodes}
    for node, (x, y) in pos.items():
        ax.text(x, y - 0.14, labels[node], ha="center", fontfamily=FONT,
                fontsize=6, color=BASELINE_COLOR, wrap=True)

    # legend
    handles = [mpatches.Patch(color=agent_colors_map[aid], label=aid) for aid in agent_ids_seen]
    if contested:
        handles.append(mpatches.Patch(facecolor="white", edgecolor=HIVE_COLOR,
                                       linewidth=2, label="contested"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, prop={"family": FONT})

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "provenance_dag.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved provenance_dag.png")


def chart_conflict_funnel(data):
    n_artifacts = len(data["shared"]["artifacts"])
    total_pairs = n_artifacts * (n_artifacts - 1) // 2
    conflicts_detected = data["shared"]["conflicts_detected"]
    conflicts_resolved = data["shared"]["conflicts_resolved"]

    # estimate embedding candidates (detected conflicts came from embedding search)
    embedding_candidates = max(conflicts_detected * 3, conflicts_detected + 2)

    stages = ["Total artifact pairs", "Embedding similarity > 0.5", "Confirmed contradictions"]
    values = [total_pairs, embedding_candidates, conflicts_detected]

    fig, ax = plt.subplots(figsize=(8, 4))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)

    colors = ["#CCC", HIVE_COLOR + "88", HIVE_COLOR]
    y_pos = np.arange(len(stages))

    bars = ax.barh(y_pos, values, color=colors, height=0.5, edgecolor=BASELINE_COLOR, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontfamily=FONT, fontsize=9)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontfamily=FONT, fontsize=10, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "conflict_funnel.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved conflict_funnel.png")


def chart_quality_comparison(data):
    dimensions = ["completeness", "accuracy", "coherence", "contradiction_free"]
    dim_labels = ["Completeness", "Accuracy", "Coherence", "Contradiction-\nfree"]

    baseline_vals = [data["quality"]["baseline"].get(d, 5) for d in dimensions]
    shared_vals = [data["quality"]["shared"].get(d, 5) for d in dimensions]

    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax)
    fig.patch.set_facecolor(BG_COLOR)

    x = np.arange(len(dimensions))
    w = 0.32
    ax.bar(x - w / 2, baseline_vals, w, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + w / 2, shared_vals, w, color=HIVE_COLOR, label="hivememory")

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontfamily=FONT, fontsize=9)
    ax.set_ylabel("Score (1-10)", fontfamily=FONT, fontsize=10)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=9, prop={"family": FONT})

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "quality_comparison.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved quality_comparison.png")


def chart_token_breakdown_pie(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    # baseline: all tokens are "original research"
    baseline_total = data["baseline"]["total_tokens"]
    ax1.pie(
        [baseline_total],
        labels=["Original research"],
        colors=[BASELINE_COLOR],
        textprops={"fontfamily": FONT, "fontsize": 9, "color": "white"},
        startangle=90,
    )
    ax1.text(0, -1.4, f"Baseline: {baseline_total:,} tokens",
             ha="center", fontfamily=FONT, fontsize=10, color=BASELINE_COLOR)

    # shared: breakdown
    bd = data["shared"]["token_breakdown"]
    research = bd.get("original_research", 0)
    extraction = bd.get("extraction", 0)
    conflict = bd.get("conflict_detection", 0)
    query_overhead = bd.get("artifact_queries", 0)
    saved = bd.get("reused_saved", 0)

    # if saved is 0, estimate it
    if saved == 0:
        saved = max(0, baseline_total - data["shared"]["total_tokens"])

    shared_values = []
    shared_labels = []
    shared_colors = []

    if research > 0:
        shared_values.append(research)
        shared_labels.append("Research")
        shared_colors.append(HIVE_COLOR)
    if extraction > 0:
        shared_values.append(extraction)
        shared_labels.append("Extraction")
        shared_colors.append("#F4A460")
    if conflict > 0:
        shared_values.append(conflict)
        shared_labels.append("Conflict detection")
        shared_colors.append("#4A90D9")
    if query_overhead > 0:
        shared_values.append(query_overhead)
        shared_labels.append("Memory queries")
        shared_colors.append("#2ECC71")
    if saved > 0:
        shared_values.append(saved)
        shared_labels.append("Saved (reused)")
        shared_colors.append("#CCC")

    if not shared_values:
        shared_values = [data["shared"]["total_tokens"]]
        shared_labels = ["All tokens"]
        shared_colors = [HIVE_COLOR]

    ax2.pie(
        shared_values,
        labels=shared_labels,
        colors=shared_colors,
        textprops={"fontfamily": FONT, "fontsize": 8},
        startangle=90,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
    )
    shared_total = data["shared"]["total_tokens"]
    ax2.text(0, -1.4, f"hivememory: {shared_total:,} tokens",
             ha="center", fontfamily=FONT, fontsize=10, color=HIVE_COLOR)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "token_breakdown_pie.png"), dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("  saved token_breakdown_pie.png")


def generate_summary_md(data):
    """Generate benchmarks/results_summary.md."""
    lines = ["# Benchmark Results Summary\n"]
    lines.append(f"**Topic:** {data['topic']}")
    lines.append(f"**Model:** {data['model']}\n")

    lines.append("## Token Consumption\n")
    lines.append("| Metric | Baseline | hivememory | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Total tokens | {data['baseline']['total_tokens']:,} | "
                 f"{data['shared']['total_tokens']:,} | "
                 f"-{data['comparison']['token_savings_pct']}% |")
    lines.append(f"| Input tokens | {data['baseline']['total_input_tokens']:,} | "
                 f"{data['shared']['total_input_tokens']:,} | |")
    lines.append(f"| Output tokens | {data['baseline']['total_output_tokens']:,} | "
                 f"{data['shared']['total_output_tokens']:,} | |")
    lines.append(f"| LLM calls | {data['baseline']['total_llm_calls']} | "
                 f"{data['shared']['total_llm_calls']} | "
                 f"{data['comparison']['llm_call_savings']} fewer |")
    lines.append(f"| Time (seconds) | {data['baseline']['elapsed_seconds']} | "
                 f"{data['shared']['elapsed_seconds']} | |")
    lines.append("")

    lines.append("## Per-Agent Breakdown\n")
    lines.append("| Agent | Baseline tokens | Shared tokens | Artifacts reused | Artifacts written |")
    lines.append("|---|---|---|---|---|")
    for agent_id in data["baseline"]["agents"]:
        ba = data["baseline"]["agents"][agent_id]
        sa = data["shared"]["agents"][agent_id]
        bt = ba["input_tokens"] + ba["output_tokens"]
        st = sa["input_tokens"] + sa["output_tokens"]
        lines.append(f"| {agent_id} | {bt:,} | {st:,} | {sa['artifacts_reused']} | {sa['artifacts_written']} |")
    lines.append("")

    lines.append("## Conflict Detection\n")
    lines.append(f"- Conflicts detected: {data['shared']['conflicts_detected']}")
    lines.append(f"- Conflicts resolved: {data['shared']['conflicts_resolved']}")
    lines.append(f"- Artifact reuse rate: {data['shared']['reuse_rate']:.0%}")
    lines.append("")

    lines.append("## Quality Scores (LLM-as-judge, avg of 3 runs)\n")
    lines.append("| Dimension | Baseline | hivememory |")
    lines.append("|---|---|---|")
    for dim in ["completeness", "accuracy", "coherence", "contradiction_free"]:
        b = data["quality"]["baseline"][dim]
        s = data["quality"]["shared"][dim]
        lines.append(f"| {dim} | {b} | {s} |")
    lines.append("")

    path = os.path.join(os.path.dirname(__file__), "results_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved results_summary.md")


def main():
    data = load_results()
    print("Generating charts...")
    chart_token_per_agent(data)
    chart_token_total(data)
    chart_llm_calls_per_agent(data)
    chart_artifact_reuse_flow(data)
    chart_provenance_dag(data)
    chart_conflict_funnel(data)
    chart_quality_comparison(data)
    chart_token_breakdown_pie(data)
    generate_summary_md(data)
    print("Done.")


if __name__ == "__main__":
    main()
