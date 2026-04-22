import re
import tiktoken
from ..core.schemas import ParsedPaper, Chunk

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

SECTION_PATTERNS = re.compile(
    r"^(abstract|introduction|background|methods?|methodology|data|"
    r"results?|discussion|conclusion|references|acknowledgements?|"
    r"supplementary|appendix)",
    re.IGNORECASE,
)


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _guess_section(paragraph: str) -> str | None:
    first_line = paragraph.strip().split("\n")[0].strip()
    if SECTION_PATTERNS.match(first_line) and len(first_line) < 80:
        return first_line.lower()
    return None


def _fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _TOKENIZER.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_TOKENIZER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def chunk_paper(
    paper: ParsedPaper,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_idx = 0
    current_section = None

    paragraphs = _split_into_paragraphs(paper.full_cleaned_text)

    buffer = ""
    buffer_pages: set[int] = set()

    # rough page mapping
    page_boundaries: list[tuple[int, str]] = []
    for page in paper.pages:
        page_boundaries.append((page.page_num, page.cleaned_text))

    def _guess_pages(text_snippet: str) -> list[int]:
        pages = []
        for pnum, ptext in page_boundaries:
            if any(w in ptext for w in text_snippet.split()[:5] if len(w) > 4):
                pages.append(pnum)
        return pages or [page_boundaries[0][0]] if page_boundaries else [1]

    def _flush(buf: str, pages: list[int], section: str | None):
        nonlocal chunk_idx
        if not buf.strip():
            return
        if _count_tokens(buf) > chunk_size * 1.5:
            for sub in _fixed_size_chunks(buf, chunk_size, overlap):
                chunk_idx += 1
                chunks.append(Chunk(
                    chunk_id=f"{paper.local_id}_chunk_{chunk_idx:03d}",
                    local_id=paper.local_id,
                    openalex_id=paper.openalex_id,
                    filename=paper.filename,
                    page_range=[pages[0], pages[-1]] if pages else [1, 1],
                    section_guess=section,
                    text=sub,
                ))
        else:
            chunk_idx += 1
            chunks.append(Chunk(
                chunk_id=f"{paper.local_id}_chunk_{chunk_idx:03d}",
                local_id=paper.local_id,
                openalex_id=paper.openalex_id,
                filename=paper.filename,
                page_range=[pages[0], pages[-1]] if pages else [1, 1],
                section_guess=section,
                text=buf.strip(),
            ))

    for para in paragraphs:
        section_hint = _guess_section(para)
        if section_hint:
            if buffer:
                _flush(buffer, list(buffer_pages), current_section)
            current_section = section_hint
            buffer = para + "\n\n"
            buffer_pages = set(_guess_pages(para))
        else:
            if _count_tokens(buffer + para) > chunk_size:
                _flush(buffer, list(buffer_pages), current_section)
                buffer = para + "\n\n"
                buffer_pages = set(_guess_pages(para))
            else:
                buffer += para + "\n\n"
                buffer_pages.update(_guess_pages(para))

    if buffer:
        _flush(buffer, list(buffer_pages), current_section)

    return chunks
