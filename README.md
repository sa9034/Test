---
title: Council of Experts — AI Governance Evaluation
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# Council of Experts — AI Safety Lab

A multi-agent system that evaluates AI systems against governance, safety, security, and regulatory frameworks using a **Council of Experts** protocol.

Three specialized agents independently assess an AI system, engage in cross-expert deliberation, and produce a synthesized council verdict (**APPROVE / FLAG / REJECT**):

- **Risk & Technical Standards Expert** — NIST AI RMF, ISO/IEC 42001, OWASP LLM Top 10, MITRE ATLAS/ATT&CK
- **AI Safety & Red-Team Expert** — 6-dimension adversarial evaluation (harmfulness, bias, transparency, deception, privacy, accountability)
- **Governance & Regulatory Expert** — EU AI Act, UN Governing AI for Humanity, UNESCO AI Ethics

---

## Quick Start (Local)

### 1. Clone and Install

```bash
git clone https://github.com/Airsoftmonkey/newUNICC.git
cd newUNICC
pip install -r requirements.txt
```

### 2. Set Your API Key

The system supports two LLM providers. Set **one** of the following environment variables:

**Anthropic (Claude) — default:**
```bash
export ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Groq (Llama 3.3 70B):**
```bash
export GROQ_API_KEY=your_groq_key_here
export LLM_PROVIDER=groq
```

> You can also paste your API key directly in the web interface (optional override field).

### 3. Ingest the Reference Corpus

The repo ships with 7 governance framework documents pre-loaded in `./corpus/`:
- `nist_ai_rmf.pdf` — NIST AI Risk Management Framework (full PDF)
- `owasp_llm_top10.md` — OWASP Top 10 for LLM Applications
- `mitre_atlas_ai_threats.md` — MITRE ATLAS + ATT&CK techniques for AI
- `ai_safety_redteam_guide.md` — 6-dimension safety evaluation methodology
- `ai_bias_fairness_privacy_guide.md` — Bias, fairness, privacy, accountability
- `eu_ai_act_summary.md` — EU AI Act risk tiers and obligations
- `un_governing_ai_humanity.md` — UN HLAB report on AI governance

Run ingestion to chunk and embed them into ChromaDB (takes ~1 minute on first run as the embedding model downloads):

```bash
python ingest.py
```

Filename-based tagging automatically routes each document to the correct expert:

| Domain | Filename keywords |
|--------|------------------|
| Risk & Technical | `nist`, `iso`, `rmf`, `42001`, `owasp`, `mitre`, `attack`, `threat`, `vulnerability` |
| AI Safety & Red-Team | `safety`, `harm`, `bias`, `fairness`, `privacy`, `deception`, `accountability`, `red_team` |
| Governance & Regulatory | `eu_ai_act`, `regulation`, `governing`, `compliance`, `unesco`, `oversight`, `human_rights` |

To add your own documents, drop PDFs/DOCX/TXT/MD files into `./corpus/` and re-run `python ingest.py`.

### 4. Launch the Web App

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

### 5. Submit an AI System for Evaluation

- **By GitHub URL:** Paste a repository URL (e.g., `https://github.com/FlashCarrot/VeriMedia`) and select "GitHub URL" as input type. The system automatically fetches the repo's README, description, and file structure.
- **By Text Description:** Paste or type a detailed description of the AI system.

Select your preferred LLM provider (Anthropic or Groq) and click **Run Evaluation**.

---

## Deploy to Hugging Face Spaces

This repo includes HF Spaces metadata and works out of the box:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — select **Gradio** as the SDK.
2. Connect it to your GitHub repo (`https://github.com/Airsoftmonkey/newUNICC`), or push directly:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/newUNICC
   git push hf main
   ```
3. In your Space settings, add your API key as a **Secret**:
   - `ANTHROPIC_API_KEY` — for Claude
   - `GROQ_API_KEY` + `LLM_PROVIDER=groq` — for Groq
4. The Space will auto-build and launch the Gradio app.

---

## How It Works

### Three-Phase Council Protocol

1. **Independent Drafting** — Each expert receives the query plus its domain-specific RAG context and produces a verdict (APPROVE/FLAG/REJECT), confidence score, reasoning, and cited sources. Experts do NOT see each other's drafts.

2. **Deliberation Round** — All drafts are revealed. Each expert responds to the other two: agreeing, disagreeing, or refining their position with justification.

3. **Consensus Synthesis** — An orchestrator reads all drafts and deliberation responses to produce the final council decision: majority verdict, synthesized rationale, points of agreement, points of dissent, and a combined confidence score.

### Output Tabs

| Tab | Contents |
|-----|----------|
| Risk & Technical Expert | Independent assessment + retrieved source documents (NIST, OWASP, MITRE) |
| AI Safety & Red-Team Expert | 6-dimension adversarial evaluation + retrieved sources |
| Governance & Regulatory Expert | Independent assessment + retrieved sources (EU AI Act, UN, UNESCO) |
| Deliberation Exchange | Cross-expert critique and responses |
| Final Consensus Decision | Majority verdict, rationale, agreement/dissent points |
| Full Audit Log | Structured JSON with all data, timestamps, latency |

### Audit Trail

Every query produces a timestamped JSON file in `./audit_logs/` containing the full evaluation trace: retrieved chunks, independent drafts, deliberation responses, final consensus, and total latency.

---

## Project Structure

```
.
├── app.py              # Gradio web interface
├── orchestrator.py     # Council orchestration (3-phase protocol)
├── agents.py           # Agent definitions and system prompts
├── rag.py              # RAG retrieval with domain filtering
├── embeddings.py       # ChromaDB + sentence-transformers setup
├── ingest.py           # Document ingestion script
├── llm_backend.py      # Swappable LLM backend (Anthropic / Groq)
├── models.py           # Pydantic data models
├── config.py           # Configuration constants
├── audit.py            # Audit trail logging
├── requirements.txt    # Python dependencies
├── corpus/             # Place governance framework documents here
├── audit_logs/         # Auto-generated evaluation logs
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required for Anthropic) | Anthropic API key |
| `GROQ_API_KEY` | (required for Groq) | Groq API key |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `groq` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model ID |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |

---

## Swapping the LLM Backend

The system is designed for backend modularity. To add a new provider:

1. Subclass `LLMBackend` in `llm_backend.py`
2. Implement the `generate(system_prompt, user_message) -> str` method
3. Register it in `get_backend()`

This makes it straightforward to swap in local fine-tuned models or other API providers.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies
