from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Evidence:
    source: str
    content: str
    reliability: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": self.content,
            "reliability": self.reliability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Evidence:
        return cls(
            source=data["source"],
            content=data["content"],
            reliability=data.get("reliability", 1.0),
        )


@dataclass
class ReasoningArtifact:
    claim: str
    agent_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 1.0
    dependencies: list[str] = field(default_factory=list)
    topic_embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "dependencies": self.dependencies,
            "topic_embedding": self.topic_embedding,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReasoningArtifact:
        return cls(
            id=data["id"],
            claim=data["claim"],
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            confidence=data.get("confidence", 1.0),
            agent_id=data["agent_id"],
            dependencies=data.get("dependencies", []),
            topic_embedding=data.get("topic_embedding", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=data.get("status", "active"),
        )


@dataclass
class Conflict:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_ids: list[str] = field(default_factory=list)
    description: str = ""
    resolved: bool = False
    winner_id: Optional[str] = None
    resolution_reason: Optional[str] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "artifact_ids": self.artifact_ids,
            "description": self.description,
            "resolved": self.resolved,
            "winner_id": self.winner_id,
            "resolution_reason": self.resolution_reason,
            "resolved_by": self.resolved_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Conflict:
        return cls(
            id=data["id"],
            artifact_ids=data.get("artifact_ids", []),
            description=data.get("description", ""),
            resolved=data.get("resolved", False),
            winner_id=data.get("winner_id"),
            resolution_reason=data.get("resolution_reason"),
            resolved_by=data.get("resolved_by"),
        )
