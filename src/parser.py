"""
Phase 1: High-Fidelity Document Processing
==========================================
Fixes the "Broker Slip Table Scramble" gotcha.
- PyMuPDF for layout-aware text + table block extraction
- pdfplumber for precise table cell recovery
- Preserves multi-column structure, schedules, deductible tables
"""

import re
import fitz          # PyMuPDF
import pdfplumber
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console

console = Console()


@dataclass
class DocumentBlock:
    """A single logical block from a parsed document."""
    block_id: str
    block_type: str          # "text" | "table" | "heading"
    content: str             # rendered text / markdown table
    page_num: int
    bbox: Optional[tuple] = None
    raw_table: Optional[List[List[str]]] = None   # structured cells for tables


@dataclass
class ParsedDocument:
    """Full parsed document with metadata."""
    source_path: str
    doc_name: str
    total_pages: int
    blocks: List[DocumentBlock] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(b.content for b in self.blocks)


class InsuranceDocumentParser:
    """
    Layout-aware parser for P&C / Reinsurance documents.
    
    Strategy:
      1. Use PyMuPDF to extract blocks with position (bbox) info
      2. Use pdfplumber to recover table structure cleanly
      3. Merge both: tables as markdown, text blocks as prose
      4. Detect headings by font size heuristics
    """

    # Heading detection: lines in ALL CAPS or lines shorter than 80 chars
    # at top of page with large font get treated as section headers
    HEADING_MIN_FONT_SIZE = 11.0
    TABLE_MIN_COLS = 2

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self, source: str) -> ParsedDocument:
        """
        Parse a PDF file or a raw text string.
        Returns a ParsedDocument with all blocks.
        """
        path = Path(source)
        if path.suffix.lower() == ".pdf" and path.exists():
            return self._parse_pdf(str(path))
        else:
            # Treat as raw text (useful for demo / tests)
            return self._parse_text(source, doc_name="inline_text")

    # ------------------------------------------------------------------ #
    #  PDF parsing                                                         #
    # ------------------------------------------------------------------ #

    def _parse_pdf(self, path: str) -> ParsedDocument:
        doc_name = Path(path).stem
        if self.verbose:
            console.print(f"[cyan]📄 Parsing PDF:[/] {path}")

        blocks: List[DocumentBlock] = []
        block_counter = 0

        # ── Step 1: extract tables with pdfplumber (table-aware) ─────────
        table_bboxes_by_page: Dict[int, List] = {}
        with pdfplumber.open(path) as plumb_pdf:
            total_pages = len(plumb_pdf.pages)
            for pg_idx, page in enumerate(plumb_pdf.pages):
                tables = page.extract_tables()
                page_tables = []
                for table in tables:
                    if not table or len(table[0]) < self.TABLE_MIN_COLS:
                        continue
                    md = self._table_to_markdown(table)
                    blk = DocumentBlock(
                        block_id=f"p{pg_idx+1}_tbl_{block_counter}",
                        block_type="table",
                        content=md,
                        page_num=pg_idx + 1,
                        raw_table=table,
                    )
                    blocks.append(blk)
                    block_counter += 1
                    # Record bboxes to avoid double-extracting table text
                    tbl_obj = page.find_tables()
                    for t in tbl_obj:
                        page_tables.append(t.bbox)
                table_bboxes_by_page[pg_idx] = page_tables

        # ── Step 2: extract text blocks with PyMuPDF ─────────────────────
        with fitz.open(path) as pdf:
            total_pages = pdf.page_count
            for pg_idx in range(total_pages):
                page = pdf[pg_idx]
                page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:   # 0 = text, 1 = image
                        continue

                    bbox = block.get("bbox", ())
                    # Skip if this bbox overlaps a table we already captured
                    if self._overlaps_table(bbox, table_bboxes_by_page.get(pg_idx, [])):
                        continue

                    text, is_heading, _ = self._extract_block_text(block)
                    if not text.strip():
                        continue

                    blk = DocumentBlock(
                        block_id=f"p{pg_idx+1}_txt_{block_counter}",
                        block_type="heading" if is_heading else "text",
                        content=text.strip(),
                        page_num=pg_idx + 1,
                        bbox=bbox,
                    )
                    blocks.append(blk)
                    block_counter += 1

        # Sort blocks by page, then vertical position
        blocks.sort(key=lambda b: (b.page_num, b.bbox[1] if b.bbox else 0))

        if self.verbose:
            n_tables = sum(1 for b in blocks if b.block_type == "table")
            n_text   = sum(1 for b in blocks if b.block_type in ("text","heading"))
            console.print(f"  [green]✓[/] {total_pages} pages | {n_text} text blocks | {n_tables} tables")

        return ParsedDocument(
            source_path=path,
            doc_name=doc_name,
            total_pages=total_pages,
            blocks=blocks,
            metadata={"source": path, "pages": total_pages},
        )

    # ------------------------------------------------------------------ #
    #  Text / demo parsing                                                 #
    # ------------------------------------------------------------------ #

    def _parse_text(self, text: str, doc_name: str) -> ParsedDocument:
        lines = text.split("\n")
        blocks = []
        current_lines = []
        block_counter = 0

        def flush(btype="text"):
            nonlocal block_counter
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append(DocumentBlock(
                    block_id=f"txt_{block_counter}",
                    block_type=btype,
                    content=content,
                    page_num=1,
                ))
                block_counter += 1
            current_lines.clear()

        for line in lines:
            stripped = line.strip()
            # Detect section headings (ALL CAPS short lines or ## prefixed)
            if re.match(r"^#{1,3}\s", stripped) or (stripped.isupper() and 5 < len(stripped) < 80):
                flush()
                blocks.append(DocumentBlock(
                    block_id=f"hdr_{block_counter}",
                    block_type="heading",
                    content=stripped.lstrip("# "),
                    page_num=1,
                ))
                block_counter += 1
            elif stripped.startswith("|") and "|" in stripped[1:]:
                # Markdown table row
                flush()
                current_lines.append(stripped)
            elif current_lines and current_lines[0].startswith("|"):
                current_lines.append(stripped)
                if not stripped.startswith("|"):
                    flush("table")
            else:
                current_lines.append(line)

        flush()
        return ParsedDocument(
            source_path="<inline>",
            doc_name=doc_name,
            total_pages=1,
            blocks=blocks,
            metadata={"source": "inline"},
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _extract_block_text(self, block: dict):
        """Extract text from a PyMuPDF block dict; detect heading."""
        lines = []
        max_font_size = 0.0
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                size = span.get("size", 0)
                if size > max_font_size:
                    max_font_size = size
            lines.append(line_text)
        text = "\n".join(lines)
        is_heading = (
            max_font_size >= self.HEADING_MIN_FONT_SIZE
            and len(text.strip()) < 120
            and text.strip() == text.strip().upper()
        )
        return text, is_heading, max_font_size

    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert pdfplumber table (list of rows) to Markdown."""
        if not table:
            return ""
        # Clean cells
        cleaned = []
        for row in table:
            cleaned.append([str(c).replace("\n", " ").strip() if c else "" for c in row])

        header = cleaned[0]
        sep    = ["---"] * len(header)
        rows   = cleaned[1:]

        def row_str(r):
            return "| " + " | ".join(r) + " |"

        md_lines = [row_str(header), row_str(sep)] + [row_str(r) for r in rows]
        return "\n".join(md_lines)

    def _overlaps_table(self, bbox: tuple, table_bboxes: List) -> bool:
        """Check if a text block bbox overlaps any known table region."""
        if not bbox or not table_bboxes:
            return False
        x0, y0, x1, y1 = bbox
        for tb in table_bboxes:
            tx0, ty0, tx1, ty1 = tb
            overlap = not (x1 < tx0 or x0 > tx1 or y1 < ty0 or y0 > ty1)
            if overlap:
                return True
        return False
