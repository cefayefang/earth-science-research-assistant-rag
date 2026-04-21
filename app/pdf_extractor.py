import re
from pathlib import Path
from .schemas import PageText, ParsedPaper


def _clean_text(text: str) -> str:
    # merge hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # remove lines that are just page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_pymupdf(pdf_path: Path) -> list[PageText]:
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        # use dict-based extraction for better column handling
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))  # sort by row then x
        raw = "\n".join(b[4] for b in blocks if b[4].strip())
        cleaned = _clean_text(raw)
        pages.append(PageText(page_num=i + 1, raw_text=raw, cleaned_text=cleaned))
    doc.close()
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> list[PageText]:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            pages.append(PageText(page_num=i + 1, raw_text=raw, cleaned_text=cleaned))
    return pages


def extract_pdf(pdf_path: Path, local_id: str, openalex_id: str | None = None) -> ParsedPaper | None:
    if not pdf_path.exists():
        return None

    try:
        pages = _extract_with_pymupdf(pdf_path)
    except Exception:
        try:
            pages = _extract_with_pdfplumber(pdf_path)
        except Exception as e:
            print(f"  [error] {pdf_path.name}: {e}")
            return None

    full_text = "\n\n".join(p.cleaned_text for p in pages if p.cleaned_text)

    return ParsedPaper(
        local_id=local_id,
        openalex_id=openalex_id,
        filename=pdf_path.name,
        pages=pages,
        full_cleaned_text=full_text,
    )
