"""Extract chapter text from a PDF using its embedded table of contents.

Writes one cleaned .txt per chapter into the output directory. Uses PyMuPDF's
block-level text extraction to drop footnotes sitting at the bottom of pages,
then normalizes whitespace, fixes hyphenated line breaks, and optionally
applies a pronunciation respelling dictionary for TTS.

Usage:
    python extract_chapters.py --pdf book.pdf --out-dir chapters_text/

Requires the PDF to have an embedded TOC (bookmarks). Scanned PDFs without
extractable text need OCR first (e.g. ocrmypdf).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

ROMAN_TO_WORD = {
    "I": "One", "II": "Two", "III": "Three", "IV": "Four", "V": "Five",
    "VI": "Six", "VII": "Seven", "VIII": "Eight", "IX": "Nine", "X": "Ten",
    "XI": "Eleven", "XII": "Twelve", "XIII": "Thirteen", "XIV": "Fourteen",
    "XV": "Fifteen", "XVI": "Sixteen", "XVII": "Seventeen", "XVIII": "Eighteen",
    "XIX": "Nineteen", "XX": "Twenty",
}

FOOTNOTE_Y_CUTOFF = 0.70  # fraction of page height below which "1. Xxx" blocks are footnotes
FOOTNOTE_START_RE = re.compile(r"^\s*\d+\.\s+\S")

# Generic header/footer patterns applied after repeating-line detection.
GENERIC_HEADER_PATTERNS = [
    re.compile(r"^\s*Chapter\s+[A-Za-z0-9]+\s*$", re.MULTILINE),  # running chapter header
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                      # bare page numbers
]

# A short line appearing on at least this fraction of pages is almost
# certainly a running header/footer (book title, section title). Evaluated
# case-insensitively, so "SHRI TITLE" and "Shri Title" are the same line.
RUNNING_HEADER_MIN_FRACTION = 0.5
RUNNING_HEADER_MAX_LEN = 60


def load_pronunciations(module_path: Path) -> dict[str, str]:
    """Load a PRONUNCIATIONS dict from a user-supplied Python file."""
    spec = importlib.util.spec_from_file_location("user_pronunciations", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pronunciations = getattr(module, "PRONUNCIATIONS", None)
    if not isinstance(pronunciations, dict):
        raise ValueError(
            f"{module_path} must define a top-level PRONUNCIATIONS dict[str, str]"
        )
    return pronunciations


def identify_running_headers(pages: list[str]) -> set[str]:
    """Lowercase short lines that appear on at least
    RUNNING_HEADER_MIN_FRACTION of pages — almost certainly running headers.
    """
    if not pages:
        return set()
    per_page_lines: list[set[str]] = []
    for page_text in pages:
        lowered = {
            line.strip().lower()
            for line in page_text.split("\n")
            if 0 < len(line.strip()) <= RUNNING_HEADER_MAX_LEN
        }
        per_page_lines.append(lowered)
    all_lines: set[str] = set().union(*per_page_lines)
    min_pages = max(2, int(RUNNING_HEADER_MIN_FRACTION * len(pages) + 0.5))
    return {
        line for line in all_lines
        if sum(1 for s in per_page_lines if line in s) >= min_pages
    }


def clean_text(
    raw_pages: list[str], pronunciations: dict[str, str] | None = None
) -> str:
    """Strip headers/footers, fix hyphenation, normalize whitespace, respell."""
    # Drop lines that repeat across pages (running headers like book titles).
    # Done case-insensitively so uppercase/mixed-case variants are both caught.
    repeaters = identify_running_headers(raw_pages)
    raw = "\n".join(raw_pages)
    if repeaters:
        text = "\n".join(
            line for line in raw.split("\n")
            if line.strip().lower() not in repeaters
        )
    else:
        text = raw

    # Strip generic header patterns (chapter name on its own line, page numbers).
    for pat in GENERIC_HEADER_PATTERNS:
        text = pat.sub("", text)

    # Join hyphenated line breaks: "begin-\nning" -> "beginning"
    text = re.sub(r"-\n([a-z])", r"\1", text)

    # Collapse single line breaks within paragraphs into spaces,
    # but keep paragraph breaks (double newlines).
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Strip footnote superscript digits attached to words: "Foo1" -> "Foo"
    text = re.sub(r"([A-Za-z])\d+\b", r"\1", text)

    # Collapse any run of whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Expand "Chapter I" / "Chapter II" for better TTS narration
    def expand_roman(m: re.Match) -> str:
        roman = m.group(1).upper()
        return f"Chapter {ROMAN_TO_WORD.get(roman, roman)}"

    text = re.sub(r"\bChapter\s+([IVXLCDM]+)\b", expand_roman, text)

    # Apply user pronunciation respellings (longest keys first so compound
    # names win over their substrings).
    if pronunciations:
        for key in sorted(pronunciations, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(key)}\b", pronunciations[key], text)

    return text.strip()


def extract_page_text_dropping_footnotes(page: fitz.Page) -> str:
    """Get page text as blocks, dropping blocks that look like footnotes."""
    page_height = page.rect.height
    cutoff_y = page_height * FOOTNOTE_Y_CUTOFF
    body_blocks: list[str] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if y0 > cutoff_y and FOOTNOTE_START_RE.match(text):
            continue
        body_blocks.append(text)
    return "\n".join(body_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Source PDF")
    parser.add_argument("--out-dir", type=Path, default=Path("chapters_text"),
                        help="Output directory for .txt files")
    parser.add_argument("--chapter-regex", default=r"^chapter\b",
                        help="Case-insensitive regex matched against TOC entry "
                             "labels to decide which entries are chapters. "
                             "Default matches entries starting with 'Chapter'.")
    parser.add_argument("--pronunciations", type=Path, default=None,
                        help="Optional Python file defining a PRONUNCIATIONS "
                             "dict[str, str] applied before writing output.")
    parser.add_argument("--no-announce", action="store_true",
                        help="Do not prepend the chapter title line to the text.")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    pronunciations = None
    if args.pronunciations:
        if not args.pronunciations.exists():
            sys.exit(f"Pronunciations file not found: {args.pronunciations}")
        pronunciations = load_pronunciations(args.pronunciations)
        print(f"Loaded {len(pronunciations)} pronunciation entries")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    toc = doc.get_toc()
    if not toc:
        sys.exit(
            "PDF has no embedded table of contents. This tool needs TOC "
            "bookmarks to identify chapter boundaries."
        )

    chapter_regex = re.compile(args.chapter_regex, re.IGNORECASE)
    chapters = [
        (entry[1], entry[2])
        for entry in toc
        if entry[0] == 1 and chapter_regex.search(entry[1])
    ]
    if not chapters:
        sys.exit(
            f"No TOC entries matched --chapter-regex {args.chapter_regex!r}. "
            f"First few TOC entries: {[e[1] for e in toc[:5]]}"
        )

    ranges: list[tuple[str, int, int]] = []
    for i, (label, start) in enumerate(chapters):
        end = chapters[i + 1][1] if i + 1 < len(chapters) else doc.page_count + 1
        ranges.append((label, start, end))

    print(f"Found {len(ranges)} chapters in {args.pdf.name}")
    for i, (label, start, end) in enumerate(ranges, 1):
        pages = range(start - 1, end - 1)
        raw_pages = [extract_page_text_dropping_footnotes(doc[p]) for p in pages]
        cleaned = clean_text(raw_pages, pronunciations=pronunciations)

        # Prepend a spoken chapter announcement unless suppressed
        if not args.no_announce:
            spoken_title = label
            for roman, word in ROMAN_TO_WORD.items():
                spoken_title = re.sub(rf"\b{roman}\b", word, spoken_title)
            cleaned = f"{spoken_title}.\n\n{cleaned}"

        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        out_path = args.out_dir / f"{i:02d}_{slug}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        word_count = len(cleaned.split())
        print(f"  {out_path.name:<35} pages {start:>3}-{end-1:<3}  {word_count:>5} words")

    print(f"\nWrote {len(ranges)} chapter files to {args.out_dir}/")


if __name__ == "__main__":
    main()
