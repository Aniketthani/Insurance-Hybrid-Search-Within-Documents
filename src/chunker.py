"""
Phase 2: Domain-Aware Chunking Strategy
========================================
Fixes the "8k Context Window Window-Wash" gotcha.
- Clause-based splitting (section headers = chunk boundaries)
- Hierarchical Parent-Child chunking:
    parent = full section (1000–2000 tokens) → stored for LLM context
    child  = small sub-clause (200–400 tokens) → used for vector search
- Overlapping metadata prepended to every chunk (LOB, doc type, etc.)
- Numerical value tagging to prevent "Silent Numerical Mismatch"
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.parser import ParsedDocument, DocumentBlock


# Insurance-specific section heading patterns
SECTION_PATTERNS = [
    r"^(section|clause|article|schedule|endorsement|exhibit|appendix|part)\s+[\dA-Z]",
    r"^\d+\.\s+[A-Z]",              # "1. Definitions"
    r"^[A-Z][A-Z\s]{4,50}$",        # ALL CAPS headings
    r"^(EXCLUSIONS|DEFINITIONS|CONDITIONS|COVERAGE|PREMIUM|DEDUCTIBLE|LIMIT)",
    r"^(WHEREAS|NOW THEREFORE|IN WITNESS)",  # treaty language
]
SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)

# Detect monetary / numerical values for audit tagging
NUMERIC_RE = re.compile(
    r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?|"
    r"\d+[\d,]*(?:\.\d+)?\s*%|"
    r"xs\s*\$[\d,]+|"
    r"\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)"
)


@dataclass
class Chunk:
    """A single retrieval unit."""
    chunk_id: str
    text: str                  # full text with metadata header
    raw_text: str              # clean text without header (for display)
    chunk_type: str            # "child" | "parent"
    parent_id: Optional[str]   # parent chunk_id for child chunks
    doc_name: str
    page_num: int
    section_title: str
    lob: str                   # Line of Business
    doc_category: str          # Policy | Claims | Underwriting | Compliance
    token_count: int
    numeric_values: List[str]  # extracted $ amounts, % values
    metadata: Dict = field(default_factory=dict)


class InsuranceChunker:
    """
    Hierarchical Parent-Child chunker for insurance documents.
    
    child chunks  → 200-400 tokens (indexed as vectors, used for search)
    parent chunks → 1000-2000 tokens (stored, injected into LLM context)
    
    Each chunk gets a metadata header:
    [Line of Business: <LOB>] [Document Type: <type>] [Section: <title>]
    """

    CHILD_TARGET_TOKENS  = 300
    CHILD_MAX_TOKENS     = 450
    PARENT_TARGET_TOKENS = 1500
    OVERLAP_TOKENS       = 75    # ~15% overlap at boundaries

    # Rough token estimate: 1 token ≈ 4 chars
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        lob: str = "Property & Casualty",
        doc_category: str = "Policy",
        verbose: bool = True,
    ):
        self.lob = lob
        self.doc_category = doc_category
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def chunk(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """
        Produce child + parent chunks from a ParsedDocument.
        Returns list sorted: parents first, then children.
        """
        sections = self._split_into_sections(parsed_doc)
        all_chunks: List[Chunk] = []

        for sec_title, sec_blocks in sections.items():
            sec_text = "\n\n".join(b.content for b in sec_blocks)
            sec_page = sec_blocks[0].page_num if sec_blocks else 1

            # Build parent chunk (full section)
            parent = self._make_parent(sec_title, sec_text, sec_page, parsed_doc.doc_name)
            all_chunks.append(parent)

            # Build child chunks (sub-clauses)
            children = self._make_children(parent, sec_title, sec_text, sec_page, parsed_doc.doc_name)
            all_chunks.extend(children)

        if self.verbose:
            parents  = sum(1 for c in all_chunks if c.chunk_type == "parent")
            children = sum(1 for c in all_chunks if c.chunk_type == "child")
            print(f"  ✓ Chunking: {parents} parent sections | {children} child chunks")

        return all_chunks

    # ------------------------------------------------------------------ #
    #  Section splitting                                                   #
    # ------------------------------------------------------------------ #

    def _split_into_sections(self, doc: ParsedDocument) -> Dict[str, List[DocumentBlock]]:
        """Group document blocks into named sections by heading detection."""
        sections: Dict[str, List[DocumentBlock]] = {}
        current_title = "Preamble"
        sections[current_title] = []

        for block in doc.blocks:
            if block.block_type == "heading" or (
                block.block_type == "text"
                and SECTION_RE.match(block.content.strip())
                and len(block.content.strip()) < 120
            ):
                title = block.content.strip()[:80]
                # Deduplicate headings
                base = title
                i = 2
                while title in sections:
                    title = f"{base} ({i})"
                    i += 1
                current_title = title
                sections[current_title] = []
            else:
                sections[current_title].append(block)

        # Remove empty sections
        return {k: v for k, v in sections.items() if v}

    # ------------------------------------------------------------------ #
    #  Chunk construction                                                  #
    # ------------------------------------------------------------------ #

    def _make_parent(self, title: str, text: str, page: int, doc_name: str) -> Chunk:
        meta_header = self._meta_header(title)
        full_text   = f"{meta_header}\n\n{text}"
        chunk_id    = self._make_id(doc_name, title, "parent")
        nums        = NUMERIC_RE.findall(text)

        return Chunk(
            chunk_id     = chunk_id,
            text         = full_text,
            raw_text     = text,
            chunk_type   = "parent",
            parent_id    = None,
            doc_name     = doc_name,
            page_num     = page,
            section_title= title,
            lob          = self.lob,
            doc_category = self.doc_category,
            token_count  = self._token_est(full_text),
            numeric_values = [n for n in nums if n.strip()],
            metadata     = {"doc_name": doc_name, "section": title, "page": page},
        )

    def _make_children(
        self, parent: Chunk, title: str, text: str, page: int, doc_name: str
    ) -> List[Chunk]:
        """Split section text into overlapping child chunks."""
        sentences = self._split_sentences(text)
        children  = []
        window: List[str] = []
        window_tokens = 0
        child_idx = 0

        for sent in sentences:
            sent_tokens = self._token_est(sent)
            if window_tokens + sent_tokens > self.CHILD_MAX_TOKENS and window:
                child = self._finalize_child(
                    window, title, page, doc_name, parent.chunk_id, child_idx
                )
                children.append(child)
                child_idx += 1

                # Overlap: keep last N tokens worth of sentences
                overlap_sents = []
                overlap_tok   = 0
                for s in reversed(window):
                    t = self._token_est(s)
                    if overlap_tok + t > self.OVERLAP_TOKENS:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += t
                window = overlap_sents
                window_tokens = overlap_tok

            window.append(sent)
            window_tokens += sent_tokens

        if window:
            child = self._finalize_child(window, title, page, doc_name, parent.chunk_id, child_idx)
            children.append(child)

        return children

    def _finalize_child(
        self,
        sentences: List[str],
        title: str,
        page: int,
        doc_name: str,
        parent_id: str,
        idx: int,
    ) -> Chunk:
        raw     = " ".join(sentences).strip()
        header  = self._meta_header(title)
        full    = f"{header}\n\n{raw}"
        cid     = self._make_id(doc_name, title, f"child_{idx}")
        nums    = NUMERIC_RE.findall(raw)

        return Chunk(
            chunk_id     = cid,
            text         = full,
            raw_text     = raw,
            chunk_type   = "child",
            parent_id    = parent_id,
            doc_name     = doc_name,
            page_num     = page,
            section_title= title,
            lob          = self.lob,
            doc_category = self.doc_category,
            token_count  = self._token_est(full),
            numeric_values = [n for n in nums if n.strip()],
            metadata     = {
                "doc_name": doc_name,
                "section": title,
                "page": page,
                "parent_id": parent_id,
            },
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _meta_header(self, section: str) -> str:
        return (
            f"[Line of Business: {self.lob}] "
            f"[Document Type: {self.doc_category}] "
            f"[Section: {section}]"
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Insurance-aware sentence splitter — respects clause numbers."""
        # Keep clause numbering together (e.g. "1.2.3 text" stays as one unit)
        text = re.sub(r"\n+", " \n ", text)
        # Split on sentence boundaries but not inside "$X,XXX" or "1.2"
        parts = re.split(r"(?<=[.?!])\s+(?=[A-Z\d\(])", text)
        # Also split on newlines that look like new clauses
        final = []
        for part in parts:
            sub = re.split(r"\n\s*(?=\d+\.\s|[A-Z]{2})", part)
            final.extend(sub)
        return [p.strip() for p in final if p.strip()]

    def _token_est(self, text: str) -> int:
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _make_id(self, doc_name: str, title: str, suffix: str) -> str:
        raw = f"{doc_name}::{title}::{suffix}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
