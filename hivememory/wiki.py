from __future__ import annotations

import os
from collections import defaultdict

from hivememory.core import HiveMemory


class WikiExporter:
    """Export a HiveMemory store to a set of interconnected markdown files."""

    def __init__(self, store: HiveMemory):
        self._store = store

    def export(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        artifacts = self._store.get_all_artifacts()
        conflicts = self._store.get_conflicts(include_resolved=True)

        # group artifacts by agent_id as topic clusters
        clusters: dict[str, list] = defaultdict(list)
        for art in artifacts:
            clusters[art.agent_id].append(art)

        artifact_map = {a.id: a for a in artifacts}

        self._write_index(output_dir, clusters, artifact_map)
        for agent_id, arts in clusters.items():
            self._write_topic(output_dir, agent_id, arts, artifact_map)
        self._write_conflicts(output_dir, conflicts, artifact_map)
        self._write_provenance(output_dir, artifacts, artifact_map)

    def _write_index(self, output_dir, clusters, artifact_map):
        lines = ["# HiveMemory Knowledge Base\n"]
        lines.append("## Topics\n")
        for agent_id, arts in sorted(clusters.items()):
            fname = _safe_filename(agent_id)
            lines.append(f"- [{agent_id}]({fname}.md) ({len(arts)} artifacts)")
        lines.append("")
        lines.append("## All Artifacts\n")
        lines.append("| ID | Claim | Agent | Confidence | Status |")
        lines.append("|---|---|---|---|---|")
        for art in sorted(artifact_map.values(), key=lambda a: a.created_at):
            short_id = art.id[:8]
            claim_short = art.claim[:80].replace("|", "\\|")
            lines.append(
                f"| {short_id} | {claim_short} | {art.agent_id} "
                f"| {art.confidence:.2f} | {art.status} |"
            )
        lines.append("")
        lines.append("## Other Pages\n")
        lines.append("- [Conflicts](CONFLICTS.md)")
        lines.append("- [Provenance](PROVENANCE.md)")
        lines.append("")

        with open(os.path.join(output_dir, "INDEX.md"), "w") as f:
            f.write("\n".join(lines))

    def _write_topic(self, output_dir, agent_id, arts, artifact_map):
        fname = _safe_filename(agent_id)
        lines = [f"# Topic: {agent_id}\n"]

        for art in sorted(arts, key=lambda a: a.created_at):
            lines.append(f"## {art.claim[:100]}\n")
            lines.append(f"- **ID:** `{art.id}`")
            lines.append(f"- **Confidence:** {art.confidence:.2f}")
            lines.append(f"- **Agent:** {art.agent_id}")
            lines.append(f"- **Status:** {art.status}")
            lines.append(f"- **Created:** {art.created_at.isoformat()}")

            if art.evidence:
                lines.append("\n### Evidence\n")
                for ev in art.evidence:
                    lines.append(
                        f"- [{ev.source}] {ev.content} "
                        f"(reliability: {ev.reliability:.1f})"
                    )

            if art.dependencies:
                lines.append("\n### Dependencies\n")
                for dep_id in art.dependencies:
                    dep = artifact_map.get(dep_id)
                    if dep:
                        lines.append(
                            f"- `{dep_id[:8]}` — {dep.claim[:60]}"
                        )
                    else:
                        lines.append(f"- `{dep_id[:8]}` (not found)")

            # backlinks: artifacts that depend on this one
            backlinks = [
                a for a in artifact_map.values() if art.id in a.dependencies
            ]
            if backlinks:
                lines.append("\n### Referenced By\n")
                for bl in backlinks:
                    lines.append(f"- `{bl.id[:8]}` — {bl.claim[:60]}")

            lines.append("")

        lines.append(f"\n[← Back to Index](INDEX.md)")
        lines.append("")

        with open(os.path.join(output_dir, f"{fname}.md"), "w") as f:
            f.write("\n".join(lines))

    def _write_conflicts(self, output_dir, conflicts, artifact_map):
        lines = ["# Conflicts\n"]

        unresolved = [c for c in conflicts if not c.resolved]
        resolved = [c for c in conflicts if c.resolved]

        lines.append(f"**Total:** {len(conflicts)} | "
                      f"**Unresolved:** {len(unresolved)} | "
                      f"**Resolved:** {len(resolved)}\n")

        if unresolved:
            lines.append("## Unresolved\n")
            for c in unresolved:
                lines.append(f"### Conflict `{c.id[:8]}`\n")
                lines.append(f"- **Description:** {c.description}")
                lines.append("- **Artifacts:**")
                for aid in c.artifact_ids:
                    art = artifact_map.get(aid)
                    if art:
                        lines.append(f"  - `{aid[:8]}` [{art.agent_id}]: {art.claim[:80]}")
                    else:
                        lines.append(f"  - `{aid[:8]}` (not found)")
                lines.append("")

        if resolved:
            lines.append("## Resolved\n")
            for c in resolved:
                lines.append(f"### Conflict `{c.id[:8]}`\n")
                lines.append(f"- **Description:** {c.description}")
                lines.append(f"- **Winner:** `{c.winner_id[:8] if c.winner_id else 'N/A'}`")
                lines.append(f"- **Reason:** {c.resolution_reason or 'N/A'}")
                lines.append(f"- **Resolved by:** {c.resolved_by or 'N/A'}")
                lines.append("")

        lines.append("\n[← Back to Index](INDEX.md)")
        lines.append("")

        with open(os.path.join(output_dir, "CONFLICTS.md"), "w") as f:
            f.write("\n".join(lines))

    def _write_provenance(self, output_dir, artifacts, artifact_map):
        lines = ["# Provenance DAG\n"]
        lines.append("Directed acyclic graph showing which artifacts built on which.\n")
        lines.append("```")
        # build adjacency: dep → artifact
        has_edges = False
        for art in sorted(artifacts, key=lambda a: a.created_at):
            if art.dependencies:
                for dep_id in art.dependencies:
                    dep = artifact_map.get(dep_id)
                    dep_label = dep.claim[:40] if dep else "unknown"
                    lines.append(
                        f"  [{dep_id[:8]} | {dep_label}]"
                        f" --> [{art.id[:8]} | {art.claim[:40]}]"
                    )
                    has_edges = True

        if not has_edges:
            # show all artifacts as roots
            for art in sorted(artifacts, key=lambda a: a.created_at):
                lines.append(f"  [{art.id[:8]} | {art.claim[:50]}]")

        lines.append("```\n")

        lines.append("## Artifact Origins\n")
        lines.append("| Artifact | Agent | Depends On |")
        lines.append("|---|---|---|")
        for art in sorted(artifacts, key=lambda a: a.created_at):
            deps = ", ".join(d[:8] for d in art.dependencies) or "—"
            lines.append(f"| `{art.id[:8]}` {art.claim[:40]} | {art.agent_id} | {deps} |")
        lines.append("")

        lines.append("\n[← Back to Index](INDEX.md)")
        lines.append("")

        with open(os.path.join(output_dir, "PROVENANCE.md"), "w") as f:
            f.write("\n".join(lines))


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
