from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from hivememory.artifact import Conflict, ReasoningArtifact


class LLMClient(Protocol):
    def check_contradiction(self, a: ReasoningArtifact, b: ReasoningArtifact) -> str:
        ...


class OpenAIConflictClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai

        self.client = openai.OpenAI()
        self.model = model

    def check_contradiction(self, a: ReasoningArtifact, b: ReasoningArtifact) -> str:
        a_sources = ", ".join(e.source for e in a.evidence)
        b_sources = ", ".join(e.source for e in b.evidence)
        prompt = (
            "Two research agents produced these findings. "
            "Do they contradict each other?\n\n"
            f"Agent {a.agent_id} claims: {a.claim}\n"
            f"Based on: {a_sources}\n\n"
            f"Agent {b.agent_id} claims: {b.claim}\n"
            f"Based on: {b_sources}\n\n"
            "Respond with exactly one word: "
            "CONTRADICTS, SUPPORTS, UNRELATED, or REFINES"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().upper()


class AnthropicConflictClient:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def check_contradiction(self, a: ReasoningArtifact, b: ReasoningArtifact) -> str:
        a_sources = ", ".join(e.source for e in a.evidence)
        b_sources = ", ".join(e.source for e in b.evidence)
        prompt = (
            "Two research agents produced these findings. "
            "Do they contradict each other?\n\n"
            f"Agent {a.agent_id} claims: {a.claim}\n"
            f"Based on: {a_sources}\n\n"
            f"Agent {b.agent_id} claims: {b.claim}\n"
            f"Based on: {b_sources}\n\n"
            "Respond with exactly one word: "
            "CONTRADICTS, SUPPORTS, UNRELATED, or REFINES"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class ConflictDetector:
    def __init__(
        self,
        store,
        llm_client: Optional[LLMClient] = None,
    ):
        self.store = store
        self.llm_client = llm_client or OpenAIConflictClient()

    def detect(
        self,
        new_artifact: ReasoningArtifact,
        existing_artifacts: list[ReasoningArtifact],
    ) -> list[Conflict]:
        if not new_artifact.topic_embedding:
            return []

        # stage 1: cosine similarity filter
        candidates = []
        for existing in existing_artifacts:
            if not existing.topic_embedding:
                continue
            if existing.id == new_artifact.id:
                continue
            sim = _cosine_similarity(
                new_artifact.topic_embedding, existing.topic_embedding
            )
            if sim > 0.75:
                candidates.append(existing)

        # stage 2: LLM verification
        conflicts = []
        for candidate in candidates:
            verdict = self.llm_client.check_contradiction(new_artifact, candidate)

            if verdict == "CONTRADICTS":
                conflict = Conflict(
                    artifact_ids=[new_artifact.id, candidate.id],
                    description=(
                        f"LLM detected contradiction between "
                        f"agent {new_artifact.agent_id} and "
                        f"agent {candidate.agent_id}"
                    ),
                )
                new_artifact.status = "contested"
                candidate.status = "contested"
                self.store._save_artifact(new_artifact)
                self.store._save_artifact(candidate)
                self.store._save_conflict(conflict)
                conflicts.append(conflict)

            elif verdict == "REFINES":
                if candidate.id not in new_artifact.dependencies:
                    new_artifact.dependencies.append(candidate.id)
                    self.store._save_artifact(new_artifact)

        return conflicts
