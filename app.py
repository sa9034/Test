"""Gradio web app — Council of Experts AI Governance Evaluation interface."""

from __future__ import annotations

import json
import textwrap
import traceback

import gradio as gr

import config
from llm_backend import get_backend
from models import AgentDraft, ConsensusSynthesis, DeliberationResponse, Verdict
from orchestrator import AGENT_NAMES, run_council

# ---------------------------------------------------------------------------
# Helper: fetch VeriMedia-style repo description from a GitHub URL
# ---------------------------------------------------------------------------

def _fetch_file_content(api_base: str, path: str, headers: dict, max_size: int = 8000) -> str | None:
    """Fetch a single file's raw content from GitHub API."""
    import requests
    try:
        resp = requests.get(
            f"{api_base}/contents/{path}",
            headers={**headers, "Accept": "application/vnd.github.v3.raw"},
            timeout=15,
        )
        if resp.ok and len(resp.text) < max_size * 2:
            return resp.text[:max_size]
    except Exception:
        pass
    return None


def _fetch_tree_recursive(api_base: str, headers: dict) -> list[str]:
    """Fetch the full file tree of a repo."""
    import requests
    try:
        resp = requests.get(f"{api_base}/git/trees/HEAD?recursive=1", headers=headers, timeout=15)
        if resp.ok:
            tree = resp.json().get("tree", [])
            return [item["path"] for item in tree if item.get("type") == "blob"]
    except Exception:
        pass
    return []


def fetch_github_description(url: str) -> str:
    """Fetch a GitHub repo's full context: metadata, README, key source files, and file tree."""
    import requests

    url = url.strip().rstrip("/")
    parts = url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
    if len(parts) < 2:
        return f"(Could not parse GitHub URL: {url})"

    owner, repo = parts[0], parts[1]
    api_base = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    sections = []

    # --- Repo metadata ---
    try:
        resp = requests.get(api_base, headers=headers, timeout=15)
        if resp.ok:
            data = resp.json()
            sections.append(f"# AI System Under Evaluation: {data.get('full_name', f'{owner}/{repo}')}")
            if data.get("description"):
                sections.append(f"**Description:** {data['description']}")
            if data.get("language"):
                sections.append(f"**Primary Language:** {data['language']}")
            if data.get("topics"):
                sections.append(f"**Topics:** {', '.join(data['topics'])}")
            if data.get("license", {}) and data["license"].get("name"):
                sections.append(f"**License:** {data['license']['name']}")
    except Exception:
        sections.append(f"# AI System Under Evaluation: {owner}/{repo}")

    # --- Full file tree ---
    all_files = _fetch_tree_recursive(api_base, headers)
    if all_files:
        sections.append(f"\n## Complete File Structure ({len(all_files)} files)")
        sections.append("```")
        for f in all_files[:150]:
            sections.append(f)
        if len(all_files) > 150:
            sections.append(f"... and {len(all_files) - 150} more files")
        sections.append("```")

    # --- README ---
    try:
        readme_resp = requests.get(
            f"{api_base}/readme",
            headers={**headers, "Accept": "application/vnd.github.v3.raw"},
            timeout=15,
        )
        if readme_resp.ok:
            sections.append(f"\n## README\n{readme_resp.text[:6000]}")
    except Exception:
        pass

    # --- Key source files: try to fetch the most informative ones ---
    # Priority patterns for AI agent repos
    priority_patterns = [
        "app.py", "main.py", "server.py", "api.py",
        "requirements.txt", "package.json", "pyproject.toml",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".env.example", "config.py", "settings.py",
    ]
    # Also grab any Python file that looks like core logic
    extra_py = [
        f for f in all_files
        if f.endswith(".py")
        and "/" not in f  # top-level only for now
        and f not in priority_patterns
    ]

    files_to_fetch = []
    for pattern in priority_patterns:
        matches = [f for f in all_files if f.endswith(pattern) or f == pattern]
        files_to_fetch.extend(matches[:1])
    files_to_fetch.extend(extra_py[:5])

    # Also grab key files from subdirectories
    important_dirs = [
        f for f in all_files
        if f.endswith(".py") and f.count("/") == 1  # one level deep
    ]
    files_to_fetch.extend(important_dirs[:8])

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files_to_fetch:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    fetched_count = 0
    for filepath in unique_files[:10]:
        content = _fetch_file_content(api_base, filepath, headers, max_size=3000)
        if content:
            sections.append(f"\n## Source File: {filepath}\n```\n{content}\n```")
            fetched_count += 1

    if fetched_count == 0:
        sections.append("\n(No source files could be fetched.)")

    result = "\n".join(sections)
    # Keep under ~6000 tokens (~24000 chars) to fit Groq free tier limits
    max_chars = 24000
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (truncated for context length)"

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

VERDICT_EMOJI = {
    Verdict.APPROVE: "APPROVE",
    Verdict.FLAG: "FLAG",
    Verdict.REJECT: "REJECT",
}


def format_draft(draft: AgentDraft) -> str:
    lines = [
        f"## {draft.agent_name}",
        f"**Verdict:** {VERDICT_EMOJI.get(draft.verdict, draft.verdict.value)}",
        f"**Confidence:** {draft.confidence:.0%}",
        f"\n### Reasoning\n{draft.reasoning}",
    ]
    if draft.cited_sources:
        lines.append("\n### Cited Sources")
        for s in draft.cited_sources:
            lines.append(f"- {s}")
    return "\n".join(lines)


def format_chunks(draft: AgentDraft) -> str:
    if not draft.retrieved_chunks:
        return "*No documents retrieved. Add relevant PDFs/docs to the `./corpus/` folder and run ingestion.*"
    parts = []
    for i, c in enumerate(draft.retrieved_chunks, 1):
        page_info = f" (p.{c.page})" if c.page else ""
        parts.append(f"**Source {i}:** {c.source}{page_info} — relevance {c.relevance_score:.2f}\n\n> {c.text[:500]}{'...' if len(c.text) > 500 else ''}")
    return "\n\n---\n\n".join(parts)


def format_deliberation(deliberations: dict[str, DeliberationResponse]) -> str:
    parts = []
    for name, delib in deliberations.items():
        parts.append(f"## {name}")
        for other, response in delib.responses.items():
            parts.append(f"**To {other}:**\n{response}\n")
        if delib.revised_verdict:
            parts.append(f"*Revised verdict:* {delib.revised_verdict.value}")
        if delib.revised_confidence is not None:
            parts.append(f"*Revised confidence:* {delib.revised_confidence:.0%}")
        parts.append("---")
    return "\n\n".join(parts)


def format_consensus(consensus: ConsensusSynthesis) -> str:
    lines = [
        f"# Council Final Decision",
        f"## Verdict: {VERDICT_EMOJI.get(consensus.majority_verdict, consensus.majority_verdict.value)}",
        f"**Combined Confidence:** {consensus.combined_confidence:.0%}",
        f"\n## Synthesized Rationale\n{consensus.synthesized_rationale}",
    ]
    if consensus.points_of_agreement:
        lines.append("\n## Points of Agreement")
        for p in consensus.points_of_agreement:
            lines.append(f"- {p}")
    if consensus.points_of_dissent:
        lines.append("\n## Points of Dissent")
        for p in consensus.points_of_dissent:
            lines.append(f"- {p}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    query_input: str,
    input_type: str,
    provider: str,
    anthropic_key: str,
    groq_key: str,
    progress=gr.Progress(),
):
    """Run the full Council of Experts evaluation."""
    # Resolve the query
    if input_type == "GitHub URL":
        progress(0.05, desc="Fetching repository information...")
        repo_description = fetch_github_description(query_input)
        query = (
            f"Evaluate the following AI system for governance, safety, and ethical compliance:\n\n"
            f"{repo_description}"
        )
    else:
        query = query_input

    if not query.strip():
        raise gr.Error("Please provide a query or GitHub URL.")

    # Set up LLM backend — use the key field matching the selected provider
    kwargs = {}
    if provider == "anthropic" and anthropic_key.strip():
        kwargs["api_key"] = anthropic_key.strip()
    elif provider == "groq" and groq_key.strip():
        kwargs["api_key"] = groq_key.strip()

    try:
        llm = get_backend(provider=provider, **kwargs)
    except ValueError as e:
        raise gr.Error(str(e))

    # Run council
    progress(0.1, desc="Phase 1: Independent expert assessments...")
    try:
        drafts, deliberations, consensus, audit_entry = run_council(query, llm=llm)
    except Exception as e:
        raise gr.Error(f"Council evaluation failed: {e}\n{traceback.format_exc()}")

    progress(1.0, desc="Complete!")

    # Format outputs
    # Tab 1: Independent assessments (one per agent)
    agent_assessments = []
    agent_sources = []
    for name in AGENT_NAMES:
        draft = drafts[name]
        agent_assessments.append(format_draft(draft))
        agent_sources.append(format_chunks(draft))

    # Tab 2: Deliberation
    deliberation_text = format_deliberation(deliberations)

    # Tab 3: Consensus
    consensus_text = format_consensus(consensus)

    # Tab 4: Full audit log
    audit_json = json.dumps(audit_entry.model_dump(mode="json"), indent=2, default=str)

    return (
        agent_assessments[0],  # Risk assessment
        agent_sources[0],      # Risk sources
        agent_assessments[1],  # Ethics assessment
        agent_sources[1],      # Ethics sources
        agent_assessments[2],  # Governance assessment
        agent_sources[2],      # Governance sources
        deliberation_text,     # Deliberation exchange
        consensus_text,        # Final consensus
        audit_json,            # Full audit log
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Council of Experts — AI Governance Evaluation",
    ) as app:
        gr.Markdown(
            "# Council of Experts — AI Safety Lab\n"
            "Submit an AI system (by GitHub URL or text description) for evaluation by three specialized experts: "
            "**Risk & Technical Standards** (NIST, ISO, OWASP, MITRE ATLAS), "
            "**AI Safety & Red-Team** (6-dimension adversarial assessment), and "
            "**Governance & Regulatory** (EU AI Act, UN, UNESCO).\n\n"
            "The council follows a three-phase protocol: independent drafting → cross-expert deliberation → consensus synthesis."
        )

        with gr.Row():
            with gr.Column(scale=3):
                input_type = gr.Radio(
                    choices=["GitHub URL", "Text Description"],
                    value="GitHub URL",
                    label="Input Type",
                )
                query_input = gr.Textbox(
                    label="AI System to Evaluate",
                    placeholder="https://github.com/FlashCarrot/VeriMedia  or paste a detailed description...",
                    lines=5,
                )
            with gr.Column(scale=1):
                provider = gr.Radio(
                    choices=["anthropic", "groq"],
                    value="groq",
                    label="LLM Provider",
                    info="Choose between Claude (Anthropic) or Groq (Llama).",
                )
                anthropic_key = gr.Textbox(
                    label="Anthropic API Key",
                    placeholder="sk-ant-... (or leave blank for env var)",
                    type="password",
                )
                groq_key = gr.Textbox(
                    label="Groq API Key",
                    placeholder="gsk_... (or leave blank for env var)",
                    type="password",
                )
                run_btn = gr.Button("Run Evaluation", variant="primary", size="lg")

        # Outputs
        with gr.Tabs():
            with gr.Tab("Risk & Technical Expert"):
                gr.Markdown("*Evaluates against NIST AI RMF, ISO/IEC 42001, OWASP LLM Top 10, and MITRE ATLAS/ATT&CK.*")
                risk_assessment = gr.Markdown(label="Assessment")
                with gr.Accordion("View Retrieved Sources", open=False):
                    risk_sources = gr.Markdown()

            with gr.Tab("AI Safety & Red-Team Expert"):
                gr.Markdown("*Red-team style evaluation across 6 dimensions: Harmfulness, Bias & Fairness, Transparency, Deception, Privacy, Accountability.*")
                ethics_assessment = gr.Markdown(label="Assessment")
                with gr.Accordion("View Retrieved Sources", open=False):
                    ethics_sources = gr.Markdown()

            with gr.Tab("Governance & Regulatory Expert"):
                gr.Markdown("*Evaluates against EU AI Act, UN Governing AI for Humanity, UNESCO AI Ethics.*")
                gov_assessment = gr.Markdown(label="Assessment")
                with gr.Accordion("View Retrieved Sources", open=False):
                    gov_sources = gr.Markdown()

            with gr.Tab("Deliberation Exchange"):
                deliberation_output = gr.Markdown()

            with gr.Tab("Final Consensus Decision"):
                consensus_output = gr.Markdown()

            with gr.Tab("Full Audit Log"):
                audit_output = gr.Code(language="json", label="Audit Log JSON")

        run_btn.click(
            fn=evaluate,
            inputs=[query_input, input_type, provider, anthropic_key, groq_key],
            outputs=[
                risk_assessment, risk_sources,
                ethics_assessment, ethics_sources,
                gov_assessment, gov_sources,
                deliberation_output,
                consensus_output,
                audit_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
