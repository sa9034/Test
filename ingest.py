"""Document ingestion script — reads PDFs/DOCX from ./corpus/, chunks, embeds, stores in ChromaDB."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import tiktoken

import config
from embeddings import get_chroma_client, get_or_create_collection

# ---------------------------------------------------------------------------
# Document readers
# ---------------------------------------------------------------------------

def read_pdf(path: Path) -> list[tuple[str, int]]:
    """Return list of (text, page_number) tuples from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def read_docx(path: Path) -> list[tuple[str, int]]:
    """Return list of (text, page_number=1) from a DOCX (no real page notion)."""
    from docx import Document

    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if full_text.strip():
        return [(full_text, 1)]
    return []


def read_txt(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.strip():
        return [(text, 1)]
    return []


READERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_txt,
    ".md": read_txt,
}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into chunks of approximately `chunk_size` tokens with `overlap`."""
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Domain tagging
# ---------------------------------------------------------------------------

def infer_domains(filename: str) -> list[str]:
    """Return list of domain tags based on filename substrings."""
    name_lower = filename.lower()
    domains = []
    for domain, keywords in config.DOMAIN_TAGS.items():
        if any(kw in name_lower for kw in keywords):
            domains.append(domain)
    # If no specific domain matched, tag with all domains (general docs)
    if not domains:
        domains = ["risk", "ethics", "governance"]
    return domains


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_corpus(corpus_dir: Path | None = None, verbose: bool = True) -> int:
    """Ingest all supported documents from the corpus directory. Returns chunk count."""
    corpus_dir = corpus_dir or config.CORPUS_DIR
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    total_chunks = 0
    files = [f for f in corpus_dir.iterdir() if f.suffix.lower() in READERS]

    if not files:
        print(f"No supported documents found in {corpus_dir}")
        return 0

    for filepath in sorted(files):
        reader = READERS.get(filepath.suffix.lower())
        if not reader:
            continue

        if verbose:
            print(f"Processing: {filepath.name}")

        pages = reader(filepath)
        domains = infer_domains(filepath.name)
        file_chunks = 0

        for page_text, page_num in pages:
            chunks = chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.sha256(
                    f"{filepath.name}:p{page_num}:c{i}".encode()
                ).hexdigest()[:16]

                collection.upsert(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": filepath.name,
                            "page": page_num,
                            "chunk_index": i,
                            "domain_risk": "risk" in domains,
                            "domain_safety": "safety" in domains,
                            "domain_governance": "governance" in domains,
                        }
                    ],
                )
                file_chunks += 1

        total_chunks += file_chunks
        if verbose:
            print(f"  -> {file_chunks} chunks, domains: {domains}")

    if verbose:
        print(f"\nIngestion complete: {total_chunks} total chunks from {len(files)} files.")
    return total_chunks


if __name__ == "__main__":
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else config.CORPUS_DIR
    ingest_corpus(corpus)
