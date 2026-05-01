"""Pydantic models for structured agent inputs and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    APPROVE = "APPROVE"
    FLAG = "FLAG"
    REJECT = "REJECT"


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    relevance_score: float = 0.0


class AgentDraft(BaseModel):
    agent_name: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cited_sources: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class DeliberationResponse(BaseModel):
    agent_name: str
    responses: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of other agent names to this agent's response to their draft.",
    )
    revised_verdict: Optional[Verdict] = None
    revised_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConsensusSynthesis(BaseModel):
    majority_verdict: Verdict
    synthesized_rationale: str
    points_of_agreement: list[str] = Field(default_factory=list)
    points_of_dissent: list[str] = Field(default_factory=list)
    combined_confidence: float = Field(ge=0.0, le=1.0)


class AuditLogEntry(BaseModel):
    timestamp: str
    query: str
    agents: dict[str, AgentAuditEntry] = Field(default_factory=dict)
    consensus: Optional[ConsensusSynthesis] = None
    total_latency_seconds: float = 0.0


class AgentAuditEntry(BaseModel):
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    independent_draft: Optional[AgentDraft] = None
    deliberation_response: Optional[DeliberationResponse] = None


# Rebuild forward references
AuditLogEntry.model_rebuild()
