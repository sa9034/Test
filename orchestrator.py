"""Council orchestrator — independent drafting, deliberation, consensus synthesis."""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from agents import run_deliberation, run_independent_draft
from audit import save_audit_log
from llm_backend import LLMBackend, get_default_backend, parse_json_from_response
from models import (
    AgentAuditEntry,
    AgentDraft,
    AuditLogEntry,
    ConsensusSynthesis,
    DeliberationResponse,
    RetrievedChunk,
    Verdict,
)
from rag import retrieve_chunks

AGENT_NAMES = [config.AGENT_RISK, config.AGENT_SAFETY, config.AGENT_GOVERNANCE]

CONSENSUS_SYSTEM_PROMPT = """\
You are the **Council Orchestrator** for a panel of AI Governance Experts.

Your role is to synthesize the independent assessments and deliberation responses from three domain experts into a final council decision.

You will receive:
1. The original query
2. Each expert's independent draft (verdict, confidence, reasoning, sources)
3. Each expert's deliberation responses to the other experts

Produce a final synthesis as a JSON object with this exact structure:
{
  "majority_verdict": "APPROVE" | "FLAG" | "REJECT",
  "synthesized_rationale": "<comprehensive rationale integrating all perspectives>",
  "points_of_agreement": ["<point 1>", "<point 2>", ...],
  "points_of_dissent": ["<point 1>", ...],
  "combined_confidence": <float 0-1>
}

Rules:
- The majority_verdict should reflect the majority position. If all three disagree, lean toward FLAG.
- The combined_confidence should weigh each expert's final confidence (revised if available, otherwise original).
- Clearly articulate where experts agreed and where they diverged.
"""


def _build_consensus_prompt(
    query: str,
    drafts: dict[str, AgentDraft],
    deliberations: dict[str, DeliberationResponse],
) -> str:
    parts = [f"## Original Query\n{query}\n"]

    parts.append("## Independent Drafts\n")
    for name, draft in drafts.items():
        parts.append(
            f"### {name}\n"
            f"- Verdict: {draft.verdict.value}\n"
            f"- Confidence: {draft.confidence}\n"
            f"- Reasoning: {draft.reasoning}\n"
            f"- Cited sources: {', '.join(draft.cited_sources)}\n"
        )

    parts.append("## Deliberation Responses\n")
    for name, delib in deliberations.items():
        parts.append(f"### {name}")
        for other_name, response in delib.responses.items():
            parts.append(f"  Response to {other_name}: {response}")
        if delib.revised_verdict:
            parts.append(f"  Revised verdict: {delib.revised_verdict.value}")
        if delib.revised_confidence is not None:
            parts.append(f"  Revised confidence: {delib.revised_confidence}")
        parts.append("")

    parts.append("Produce the final consensus synthesis as a JSON object.")
    return "\n".join(parts)


def synthesize_consensus(
    query: str,
    drafts: dict[str, AgentDraft],
    deliberations: dict[str, DeliberationResponse],
    llm: LLMBackend | None = None,
) -> ConsensusSynthesis:
    """Produce the final council consensus from drafts and deliberation."""
    llm = llm or get_default_backend()
    user_message = _build_consensus_prompt(query, drafts, deliberations)

    raw = llm.generate(CONSENSUS_SYSTEM_PROMPT, user_message)
    try:
        data = parse_json_from_response(raw)
    except ValueError:
        # Fallback: compute majority mechanically
        final_verdicts = []
        for name in AGENT_NAMES:
            d = deliberations.get(name)
            if d and d.revised_verdict:
                final_verdicts.append(d.revised_verdict)
            else:
                final_verdicts.append(drafts[name].verdict)
        counts = Counter(v.value for v in final_verdicts)
        majority = Verdict(counts.most_common(1)[0][0])

        return ConsensusSynthesis(
            majority_verdict=majority,
            synthesized_rationale=f"(Fallback synthesis) Raw LLM output: {raw[:500]}",
            points_of_agreement=[],
            points_of_dissent=[],
            combined_confidence=0.5,
        )

    return ConsensusSynthesis(
        majority_verdict=Verdict(data.get("majority_verdict", "FLAG")),
        synthesized_rationale=data.get("synthesized_rationale", ""),
        points_of_agreement=data.get("points_of_agreement", []),
        points_of_dissent=data.get("points_of_dissent", []),
        combined_confidence=float(data.get("combined_confidence", 0.5)),
    )


def run_council(
    query: str,
    llm: LLMBackend | None = None,
) -> tuple[dict[str, AgentDraft], dict[str, DeliberationResponse], ConsensusSynthesis, AuditLogEntry]:
    """Execute the full Council of Experts protocol for a query.

    Returns (drafts, deliberations, consensus, audit_log).
    """
    llm = llm or get_default_backend()
    sequential = getattr(llm, "provider_name", "") == "groq"
    start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # --- Phase 0: RAG retrieval (parallel — local, no API calls) ---
    agent_chunks: dict[str, list[RetrievedChunk]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(retrieve_chunks, query, name): name for name in AGENT_NAMES
        }
        for fut in as_completed(futures):
            name = futures[fut]
            agent_chunks[name] = fut.result()

    # --- Phase 1: Independent drafting ---
    # Sequential for Groq (rate limits), parallel for Anthropic
    drafts: dict[str, AgentDraft] = {}
    if sequential:
        for name in AGENT_NAMES:
            drafts[name] = run_independent_draft(name, query, agent_chunks[name], llm)
            time.sleep(2)  # Rate limit buffer
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(run_independent_draft, name, query, agent_chunks[name], llm): name
                for name in AGENT_NAMES
            }
            for fut in as_completed(futures):
                name = futures[fut]
                drafts[name] = fut.result()

    # --- Phase 2: Deliberation ---
    deliberations: dict[str, DeliberationResponse] = {}
    if sequential:
        for name in AGENT_NAMES:
            other_drafts = [drafts[n] for n in AGENT_NAMES if n != name]
            deliberations[name] = run_deliberation(name, query, drafts[name], other_drafts, llm)
            time.sleep(2)
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for name in AGENT_NAMES:
                other_drafts = [drafts[n] for n in AGENT_NAMES if n != name]
                futures[pool.submit(run_deliberation, name, query, drafts[name], other_drafts, llm)] = name
            for fut in as_completed(futures):
                name = futures[fut]
                deliberations[name] = fut.result()

    # --- Phase 3: Consensus synthesis ---
    if sequential:
        time.sleep(2)
    consensus = synthesize_consensus(query, drafts, deliberations, llm)

    total_latency = time.time() - start

    # --- Build audit log ---
    agents_audit: dict[str, AgentAuditEntry] = {}
    for name in AGENT_NAMES:
        agents_audit[name] = AgentAuditEntry(
            retrieved_chunks=agent_chunks.get(name, []),
            independent_draft=drafts.get(name),
            deliberation_response=deliberations.get(name),
        )

    audit_entry = AuditLogEntry(
        timestamp=timestamp,
        query=query,
        agents=agents_audit,
        consensus=consensus,
        total_latency_seconds=round(total_latency, 2),
    )

    save_audit_log(audit_entry)

    return drafts, deliberations, consensus, audit_entry
