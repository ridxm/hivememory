"""Shared research data and utilities for benchmarks."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

RESEARCH_TOPIC = "Competitive Landscape of AI Code Editors (2026)"

SUBTOPICS = [
    {
        "agent_id": "market-analyst",
        "query": "market position and share of AI code editors 2026",
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
    {
        "agent_id": "tech-evaluator",
        "query": "technical capabilities and benchmarks of AI code editors",
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
                    ("cursor blog", "Codebase indexing and smart retrieval outperform raw context length"),
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
    {
        "agent_id": "ux-researcher",
        "query": "user experience and developer satisfaction with AI code editors",
        "findings": [
            {
                "claim": "Developer satisfaction is highest for AI-native editors (Cursor, Windsurf) vs plugin-based tools",
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
]


@dataclass
class BenchmarkResult:
    mode: str  # "baseline" or "shared"
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_artifacts: int = 0
    num_conflicts_detected: int = 0
    num_conflicts_resolved: int = 0
    reuse_rate: float = 0.0
    elapsed_seconds: float = 0.0
    agent_outputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "num_artifacts": self.num_artifacts,
            "num_conflicts_detected": self.num_conflicts_detected,
            "num_conflicts_resolved": self.num_conflicts_resolved,
            "reuse_rate": self.reuse_rate,
            "elapsed_seconds": self.elapsed_seconds,
        }


def simulate_llm_call(
    agent_id: str,
    query: str,
    context: str = "",
) -> tuple[str, int, int]:
    """Simulate an LLM research call. Returns (response, prompt_tokens, completion_tokens).

    When context is provided (from shared memory), the prompt is smaller because
    the agent doesn't need to re-derive existing knowledge.
    """
    base_prompt = len(query.split()) * 10  # ~10 tokens per word
    base_completion = random.randint(400, 800)

    if context:
        # shared memory reduces prompt size — agent reuses prior findings
        # instead of asking the LLM to derive them from scratch
        context_savings = min(len(context.split()) * 3, base_prompt // 2)
        prompt_tokens = base_prompt - context_savings + 50  # 50 for the context ref
        completion_tokens = int(base_completion * 0.7)  # shorter response, less redundancy
    else:
        # no shared memory — agent must derive everything independently
        # add "redundancy tax": each agent repeats background research
        prompt_tokens = base_prompt + random.randint(200, 500)
        completion_tokens = base_completion + random.randint(100, 300)

    response = f"[{agent_id}] researched: {query}"
    return response, prompt_tokens, completion_tokens
