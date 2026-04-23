from __future__ import annotations

from io import BytesIO
from typing import Tuple, List, Dict, Any
import re


MAX_SUMMARY_CHARS = 18000
MAP_CHUNK_CHARS = 9000
MAX_MAP_CHUNKS = 8

TEMPLATE_RESEARCH = "research"
TEMPLATE_BOOK = "book"
TEMPLATE_GENERIC = "generic"

_PROCESS_NOTE_PATTERNS = [
    r"^\s*note\s*:",
    r"^\s*i\s+(merged|summarized|combined|have merged|have summarized)",
    r"^\s*as requested",
    r"^\s*this summary",
    r"^\s*in this summary",
]


def extract_document_text(filename: str, content_type: str, data: bytes) -> Tuple[str, str]:
    """Extract plain text from common document types.

    Returns (extracted_text, parser_used).
    """
    lower_name = (filename or "").lower()
    content_type = (content_type or "").lower()

    if lower_name.endswith(".pdf") or "pdf" in content_type:
        return _extract_pdf_text(data), "pdf"

    if lower_name.endswith(".docx") or "word" in content_type:
        return _extract_docx_text(data), "docx"

    if lower_name.endswith((".txt", ".md", ".csv", ".json", ".html", ".htm")):
        return _decode_text_bytes(data), "text"

    # Best-effort fallback for unknown formats.
    return _decode_text_bytes(data), "fallback-text"


def build_summary_prompt(
    text: str,
    age: int,
    template: str,
    profession: str = "",
    expertise_level: str = "",
    area_of_interest: str = "",
) -> str:
    trimmed_text = text[:MAX_SUMMARY_CHARS]

    instructions = _template_instructions(template)
    quality_rules = _template_quality_rules(template)
    output_format = _format_spec_text(template, age)
    learner_profile = (
        "Learner profile:\n"
        f"- Profession: {profession or 'Not provided'}\n"
        f"- Expertise level: {expertise_level or 'Not provided'}\n"
        f"- Area of interest: {area_of_interest or 'Not provided'}"
    )

    return f"""
You are AgeXplain, an age-adaptive tutor.

Summarize the uploaded document for a learner aged {age}.

{learner_profile}

Document type guidance:
{instructions}

Output format (exact section titles):
{output_format}

Rules:
- Stay faithful to source text.
- Do not invent missing facts.
- If evidence is weak for a section, include "Not clearly stated in the document" at most once in that section.
- Return only the requested section headings and bullet points.
- Do not include notes, disclaimers, or comments about your process.
- Do not write phrases like "I merged", "I summarized", "as requested", or "Note:".
- Do not include any extra heading beyond the required ones.
- Keep bullets concrete and specific; avoid vague filler.
- If the text has numbers/benchmarks/results, preserve them accurately.
- Do NOT copy front matter metadata (copyright/editorial/production names) into summary bullets.
- Use the learner profile only to tune wording depth and framing, never to alter document facts.
- If expertise is Beginner: define the first key technical term in plain language.
- If expertise is Advanced and age >= 19: avoid basic definitions and use precise terminology.
- If area of interest is provided, add at most one brief relatable framing phrase when natural.
{quality_rules}

Document text:
{trimmed_text}
""".strip()


def summarize_with_fallback(
    text: str,
    age: int,
    llm,
    profession: str = "",
    expertise_level: str = "",
    area_of_interest: str = "",
) -> Tuple[str, str]:
    """Try LLM summary first; fall back to extractive summary if LLM is unavailable.

    Returns (summary_text, mode) where mode is "llm" or "fallback".
    """
    template = _detect_summary_template(text)
    prompt = build_summary_prompt(
        text,
        age,
        template,
        profession=profession,
        expertise_level=expertise_level,
        area_of_interest=area_of_interest,
    )
    try:
        if len(text) > MAX_SUMMARY_CHARS:
            summary = _summarize_long_document(
                text,
                age,
                llm,
                template,
                profession=profession,
                expertise_level=expertise_level,
                area_of_interest=area_of_interest,
            )
        else:
            summary = llm.invoke(prompt).strip()
        if summary:
            cleaned = _clean_summary_text(summary)
            normalized = _normalize_summary(cleaned, age, template, source_text=text)
            with_topics = _append_topic_suggestions(normalized, text, llm=llm)
            return with_topics, "llm"
    except Exception:
        pass

    fallback = _fallback_summary(text, age, template)
    return _append_topic_suggestions(fallback, text, llm=None), "fallback"


def _summarize_long_document(
    text: str,
    age: int,
    llm,
    template: str,
    profession: str = "",
    expertise_level: str = "",
    area_of_interest: str = "",
) -> str:
    chunks = _chunk_text(text, MAP_CHUNK_CHARS, MAX_MAP_CHUNKS)
    partials: List[str] = []

    map_sections = _template_map_sections(template)
    quality_rules = _template_quality_rules(template)
    learner_profile = (
        "Learner profile:\n"
        f"- Profession: {profession or 'Not provided'}\n"
        f"- Expertise level: {expertise_level or 'Not provided'}\n"
        f"- Area of interest: {area_of_interest or 'Not provided'}"
    )

    for idx, chunk in enumerate(chunks, start=1):
        section_spec = "\n\n".join(
            [f"{name}:\n- {rule}" for name, rule in map_sections]
        )
        map_prompt = f"""
You are AgeXplain, an age-adaptive tutor.

Summarize this part of a long document for age {age}.
Chunk {idx} of {len(chunks)}.

{learner_profile}

Return exactly these sections:
{section_spec}

Rules:
- Return only these sections and bullet points.
- Do not include notes, disclaimers, or process comments.
- Do not write phrases like "I merged", "I summarized", "as requested", or "Note:".
- Use learner profile only to adapt wording depth and familiarity, never to alter facts.
{quality_rules}

Chunk text:
{chunk}
""".strip()
        partial = llm.invoke(map_prompt).strip()
        if partial:
            partials.append(partial)

    if not partials:
        return llm.invoke(
            build_summary_prompt(
                text[:MAX_SUMMARY_CHARS],
                age,
                template,
                profession=profession,
                expertise_level=expertise_level,
                area_of_interest=area_of_interest,
            )
        ).strip()

    output_format = _format_spec_text(template, age)

    partials_text = "\n\n".join(partials)
    reduce_prompt = f"""
You are AgeXplain, an age-adaptive tutor.

You are given chunk summaries of a long document.
Create one final merged summary for age {age}.

{learner_profile}

Output format (exact section titles):
{output_format}

Rules:
- Merge duplicates across chunks.
- Stay faithful to source chunk summaries.
- If evidence is weak for a section, include "Not clearly stated in the document" at most once in that section.
- Return only the requested section headings and bullet points.
- Do not include notes, disclaimers, or comments about your process.
- Do not write phrases like "I merged", "I summarized", "as requested", or "Note:".
- Keep adaptation aligned to learner profile (tone/depth only), without changing factual claims.
{quality_rules}

Chunk summaries:
{partials_text}
""".strip()

    return llm.invoke(reduce_prompt).strip()


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF support requires pypdf package") from exc

    reader = PdfReader(BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def _extract_docx_text(data: bytes) -> str:
    try:
        from docx import Document
    except Exception as exc:
        raise RuntimeError("DOCX support requires python-docx package") from exc

    doc = Document(BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs).strip()


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def _fallback_summary(text: str, age: int, template: str) -> str:
    sentences = _split_sentences(text)

    first_line = sentences[0] if sentences else "Not clearly stated in the document."
    student = _simplify_for_age(" ".join(sentences[:4]) if sentences else first_line, age)

    if template == TEMPLATE_BOOK:
        sections = {
            "Core Idea": [first_line],
            "Main Lessons": _ensure_range_count(sentences[:5], 3, 5),
            "Author's Evidence/Arguments": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["argue", "example", "evidence", "claim", "case"])][:4],
                2,
                4,
            ),
            "Critiques or Weak Points": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["limit", "weak", "bias", "challenge", "critic"])][:3],
                2,
                3,
            ),
            "Why It Matters": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["important", "impact", "matter", "decision", "use"])][:2],
                1,
                2,
            ),
        }
    elif template == TEMPLATE_GENERIC:
        sections = {
            "Overview": [first_line],
            "Important Points": _ensure_range_count(sentences[:5], 3, 5),
            "Evidence from Text": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["because", "shows", "example", "evidence", "data"])][:4],
                2,
                4,
            ),
            "Cautions/Limitations": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["limit", "risk", "bias", "uncertain", "challenge"])][:3],
                2,
                3,
            ),
        }
    else:
        sections = {
            "Executive Summary": [first_line],
            "Key Points": _ensure_range_count(sentences[:5], 3, 5),
            "Method": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["method", "approach", "model", "dataset", "experiment", "procedure"])][:3],
                2,
                3,
            ),
            "Results": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["result", "found", "improve", "accuracy", "outperform", "conclusion"])][:3],
                2,
                3,
            ),
            "Limitations": _ensure_range_count(
                [s for s in sentences if _has_any(s.lower(), ["limit", "future work", "bias", "challenge", "weakness"])][:3],
                2,
                3,
            ),
        }

    return _render_summary(sections, student, age, source_text=text)


def _render_summary(sections: Dict[str, List[str]], simple_text: str, age: int, source_text: str = "") -> str:
    lines: List[str] = []
    for heading, items in sections.items():
        normalized_items = [i.strip() for i in items if i and i.strip()]
        if not normalized_items:
            normalized_items = ["Not clearly stated in the document."]
        lines.append(f"{heading}:")
        lines.extend([f"- {item}" for item in normalized_items])
        lines.append("")

    lines.append("Summary:")
    lines.append(f"- {_build_age_summary(simple_text, source_text, age, min_sentences=3, max_sentences=4)}")
    return "\n".join(lines).strip()


def _chunk_text(text: str, chunk_size: int, max_chunks: int) -> List[str]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []

    chunks = []
    start = 0
    while start < len(compact) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(compact))
        if end < len(compact):
            split_at = compact.rfind(". ", start, end)
            if split_at > start + int(chunk_size * 0.6):
                end = split_at + 1
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def _normalize_summary(summary: str, age: int, template: str, source_text: str = "") -> str:
    text = (summary or "").strip()
    if not text:
        return text

    sections = _extract_sections(text)
    if not sections:
        return _fallback_summary(text, age, template)

    schema = _template_schema(template)
    rendered: Dict[str, List[str]] = {}
    first_fallback = "Not clearly stated in the document."

    for spec in schema:
        name = spec["name"]
        min_count = spec["min"]
        max_count = spec["max"]
        section_items = _get_section_items(sections, [name] + spec.get("aliases", []))

        if max_count == 1:
            head = section_items[0] if section_items else first_fallback
            rendered[name] = [head]
        else:
            rendered[name] = _ensure_range_count(section_items, min_count, max_count)

    rendered = _dedupe_across_sections(rendered)

    primary_head = schema[0]["name"]
    student_lines = _get_section_items(
        sections,
        [f"Summary (Age {age})", "Summary", f"Simple Summary (Age {age})", "Simple Summary", "Explain Like", "Student-Level Explanation"],
    ) or rendered.get(primary_head, [first_fallback])
    student = " ".join(student_lines)

    return _render_summary(rendered, student, age, source_text=source_text)


def _clean_summary_text(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return text

    # Remove markdown emphasis and normalize bullets.
    text = text.replace("**", "")
    text = re.sub(r"^\s*\*\s+", "- ", text, flags=re.MULTILINE)

    # Remove common preface lines before the actual sections.
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not _is_process_note_line(ln)]
    start_idx = 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("Executive Summary:") or stripped.startswith("TLDR:"):
            start_idx = i
            break
    lines = lines[start_idx:]

    # Standardize common section names.
    normalized = []
    for ln in lines:
        s = ln.strip()
        lowered = s.lower()
        if lowered.startswith("rules:"):
            break
        if lowered in {"executive summary:", "executive summary", "tldr:", "tldr"}:
            normalized.append("Executive Summary:")
        elif lowered in {"core idea:", "core idea"}:
            normalized.append("Core Idea:")
        elif lowered in {"overview:", "overview"}:
            normalized.append("Overview:")
        elif lowered in {"key points:", "key points"}:
            normalized.append("Key Points:")
        elif lowered in {"main lessons:", "main lessons"}:
            normalized.append("Main Lessons:")
        elif lowered in {"important points:", "important points"}:
            normalized.append("Important Points:")
        elif lowered in {"method:", "method"}:
            normalized.append("Method:")
        elif lowered in {"author's evidence/arguments:", "authors evidence/arguments:", "author evidence:", "author's evidence:", "author arguments:"}:
            normalized.append("Author's Evidence/Arguments:")
        elif lowered in {"evidence from text:", "evidence from text"}:
            normalized.append("Evidence from Text:")
        elif lowered in {"results:", "results"}:
            normalized.append("Results:")
        elif lowered in {"limitations:", "limitations"}:
            normalized.append("Limitations:")
        elif lowered in {"critiques or weak points:", "weak points:", "critiques:", "criticisms:"}:
            normalized.append("Critiques or Weak Points:")
        elif lowered in {"why it matters:", "why it matters"}:
            normalized.append("Why It Matters:")
        elif lowered in {"cautions/limitations:", "cautions:", "cautions and limitations:"}:
            normalized.append("Cautions/Limitations:")
        elif lowered.startswith("summary") or lowered.startswith("simple summary") or lowered.startswith("student-level explanation") or lowered.startswith("explain like"):
            if not s.endswith(":"):
                s = s + ":"
            normalized.append("Summary:")
        else:
            normalized.append(ln)

    return "\n".join(normalized).strip()


def _is_process_note_line(line: str) -> bool:
    s = (line or "").strip().lower()
    if not s:
        return False
    return any(re.match(pattern, s) for pattern in _PROCESS_NOTE_PATTERNS)


def _dedupe_across_sections(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    seen = set()
    deduped: Dict[str, List[str]] = {}

    for heading, items in sections.items():
        kept: List[str] = []
        for item in items:
            normalized = re.sub(r"\s+", " ", (item or "").strip().lower())
            if not normalized:
                continue
            if normalized in seen and normalized != "not clearly stated in the document.":
                continue
            seen.add(normalized)
            kept.append(item)
        deduped[heading] = kept if kept else ["Not clearly stated in the document."]

    return deduped


def _split_sentences(text: str):
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+", compact)
    clean = [p.strip() for p in parts if len(p.strip()) > 40]
    return clean[:30]


def _has_any(text: str, words):
    return any(w in text for w in words)


def _simplify_for_age(text: str, age: int) -> str:
    base = text.strip()
    if not base:
        return "This document explains a topic and gives some findings."
    if age <= 10:
        return f"In simple words: {base}"
    return base


def _clip_to_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    clean = [p.strip() for p in parts if p.strip()]
    if not clean:
        return "Not clearly stated in the document."
    return " ".join(clean[:max_sentences])


def _build_age_summary(seed_text: str, source_text: str, age: int, min_sentences: int = 3, max_sentences: int = 4) -> str:
    """Build a richer age-adapted summary with a guaranteed sentence range."""
    pieces: List[str] = []

    seed_parts = re.split(r"(?<=[.!?])\s+", (seed_text or "").strip())
    pieces.extend([p.strip() for p in seed_parts if p and p.strip()])

    if len(pieces) < min_sentences:
        src_parts = _split_sentences(source_text)
        for p in src_parts:
            p = p.strip()
            if not p:
                continue
            norm = re.sub(r"\s+", " ", p).lower()
            if any(re.sub(r"\s+", " ", q).lower() == norm for q in pieces):
                continue
            pieces.append(p)
            if len(pieces) >= max_sentences:
                break

    selected = pieces[:max_sentences] if pieces else ["Not clearly stated in the document."]
    merged = " ".join(selected)
    simplified = _simplify_for_age(merged, age)
    final_parts = re.split(r"(?<=[.!?])\s+", simplified)
    final_clean = [p.strip() for p in final_parts if p and p.strip()]

    if len(final_clean) < min_sentences:
        # Ensure minimum detail by appending brief grounded statements.
        defaults = [
            "It explains the main idea clearly.",
            "It gives supporting points from the text.",
            "It also highlights limits or cautions to remember.",
        ]
        for d in defaults:
            final_clean.append(d)
            if len(final_clean) >= min_sentences:
                break

    return " ".join(final_clean[:max_sentences])


def _ensure_exact_count(items: List[str], count: int) -> List[str]:
    clean = [i.strip() for i in items if i and i.strip()]
    if not clean:
        clean = ["Not clearly stated in the document."]
    if len(clean) >= count:
        return clean[:count]
    return clean + ["Not clearly stated in the document."] * (count - len(clean))


def _ensure_range_count(items: List[str], min_count: int, max_count: int) -> List[str]:
    unknown = "Not clearly stated in the document."
    seen = set()
    clean = []
    for item in items or []:
        val = (item or "").strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(val)

    real = [v for v in clean if v.lower() != unknown.lower()]
    out = real[:max_count]

    # Enforce minimum depth for enterprise consistency.
    while len(out) < min_count:
        out.append(unknown)

    if not out:
        return [unknown]

    return out[:max_count]


def _extract_sections(text: str):
    sections = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.endswith(":") and not stripped.startswith("- "):
            current = stripped[:-1].strip()
            sections.setdefault(current, [])
            continue

        if current is None:
            continue

        if stripped.startswith("- "):
            sections[current].append(stripped[2:].strip())
        else:
            sections[current].append(stripped)

    return sections


def _template_schema(template: str) -> List[Dict[str, Any]]:
    if template == TEMPLATE_BOOK:
        return [
            {"name": "Core Idea", "min": 1, "max": 1},
            {"name": "Main Lessons", "min": 3, "max": 5},
            {"name": "Author's Evidence/Arguments", "min": 2, "max": 4},
            {"name": "Critiques or Weak Points", "min": 2, "max": 3},
            {"name": "Why It Matters", "min": 1, "max": 2},
        ]
    if template == TEMPLATE_GENERIC:
        return [
            {"name": "Overview", "min": 1, "max": 1},
            {"name": "Important Points", "min": 3, "max": 5},
            {"name": "Evidence from Text", "min": 2, "max": 4},
            {"name": "Cautions/Limitations", "min": 2, "max": 3},
        ]
    return [
        {"name": "Executive Summary", "min": 1, "max": 1},
        {"name": "Key Points", "min": 3, "max": 5},
        {"name": "Method", "min": 2, "max": 3},
        {"name": "Results", "min": 2, "max": 3},
        {"name": "Limitations", "min": 2, "max": 3},
    ]


def _template_map_sections(template: str) -> List[Tuple[str, str]]:
    if template == TEMPLATE_BOOK:
        return [
            ("Main Lessons", "up to 5 bullets"),
            ("Author's Evidence/Arguments", "up to 4 bullets"),
            ("Critiques or Weak Points", "up to 3 bullets"),
            ("Why It Matters", "up to 2 bullets"),
        ]
    if template == TEMPLATE_GENERIC:
        return [
            ("Important Points", "up to 5 bullets"),
            ("Evidence from Text", "up to 4 bullets"),
            ("Cautions/Limitations", "up to 3 bullets"),
        ]
    return [
        ("Key Points", "up to 5 bullets"),
        ("Method", "up to 3 bullets"),
        ("Results", "up to 3 bullets"),
        ("Limitations", "up to 3 bullets"),
    ]


def _template_instructions(template: str) -> str:
    if template == TEMPLATE_BOOK:
        return "Treat this as a book/chapter style document. Focus on arguments, lessons, and why they matter."
    if template == TEMPLATE_GENERIC:
        return "Treat this as a general document. Use neutral sections and avoid assuming it is a formal research paper."
    return "Treat this as a research-style document with method/findings/limitations where available."


def _template_quality_rules(template: str) -> str:
    if template == TEMPLATE_BOOK:
        return (
            "- In Author's Evidence/Arguments, prioritize concrete supporting details from the text.\n"
            "- In Critiques or Weak Points, focus on weaknesses of the book's claims/coverage, not generic AI warnings.\n"
            "- In Why It Matters, include practical impact for learners/readers.\n"
            "- Prefer chapter/theme-specific claims over generic statements."
        )
    if template == TEMPLATE_RESEARCH:
        return (
            "- In Method, include study type and what was analyzed.\n"
            "- In Results, prioritize explicit findings from the document over generic restatements.\n"
            "- In Results, include at least one concrete metric/benchmark when present.\n"
            "- In Limitations, prefer explicitly stated constraints from the paper."
        )
    return (
        "- In Evidence from Text, use concrete support from the document.\n"
        "- In Cautions/Limitations, avoid generic warnings unless grounded in the source.\n"
        "- Keep examples domain-relevant, not generic textbook language."
    )


def _format_spec_text(template: str, age: int) -> str:
    specs = _template_schema(template)
    lines: List[str] = []
    for spec in specs:
        heading = spec["name"]
        min_count = spec["min"]
        max_count = spec["max"]
        if max_count == 1:
            lines.append(f"{heading}:\n- one short sentence")
        else:
            lines.append(f"{heading}:\n- {min_count} to {max_count} bullets")
    lines.append(f"Summary:\n- 3 to 4 short sentences in age-appropriate language for age {age}")
    return "\n\n".join(lines)


def _normalize_header_key(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    return compact


def _get_section_items(sections: Dict[str, List[str]], names: List[str]) -> List[str]:
    lookup = {_normalize_header_key(k): v for k, v in sections.items()}
    for name in names:
        key = _normalize_header_key(name)
        if key in lookup:
            return lookup[key]
    return []


def _detect_summary_template(text: str) -> str:
    sample = (text or "")[:20000].lower()

    research_keywords = [
        "abstract", "method", "methodology", "experiment", "results", "discussion", "references", "doi", "arxiv",
        "dataset", "baseline", "evaluation", "accuracy", "precision", "recall",
    ]
    book_keywords = [
        "chapter", "preface", "table of contents", "part i", "part ii", "prologue", "epilogue", "author",
    ]

    research_score = sum(sample.count(k) for k in research_keywords)
    book_score = sum(sample.count(k) for k in book_keywords)

    citation_score = len(re.findall(r"\[[0-9]+\]", sample)) + len(re.findall(r"\([a-z][a-z\-]+,\s*\d{4}\)", sample))
    research_score += citation_score

    if research_score >= 8 and research_score > book_score + 2:
        return TEMPLATE_RESEARCH
    if book_score >= 4 and book_score > research_score:
        return TEMPLATE_BOOK
    return TEMPLATE_GENERIC


def _append_topic_suggestions(summary: str, source_text: str, max_topics: int = 8, llm=None) -> str:
    """Append discoverable topics using LLM extraction (with regex fallback)."""
    base = (summary or "").strip()

    # Prefer LLM extraction — much more reliable than regex for PDF books.
    if llm is not None:
        topics = _extract_topics_with_llm(source_text, llm, max_topics=max_topics)
    else:
        topics = []

    # Fallback: regex-based extraction if LLM gave nothing useful.
    if len(topics) < 3:
        regex_topics = _extract_topics_from_source(source_text, max_topics=max_topics)
        seen = {t.lower() for t in topics}
        for t in regex_topics:
            if t.lower() not in seen:
                topics.append(t)
                seen.add(t.lower())
            if len(topics) >= max_topics:
                break

    if not topics:
        return base

    lines = [base, "", "Topics You Can Ask About:"]
    lines.extend([f"- {topic}" for topic in topics])
    lines.append("")
    lines.append("Would you like to know more about any of these topics?")
    return "\n".join(lines).strip()


def _extract_topics_with_llm(text: str, llm, max_topics: int = 8) -> List[str]:
    """Use LLM to extract meaningful chapter/topic names from the document."""
    # Use first 6000 chars — enough to cover table of contents + intro.
    sample = (text or "")[:6000].strip()
    if not sample:
        return []

    prompt = f"""
You are reading a document and need to list the main topics or chapter names a learner could explore.

Document excerpt (first section):
{sample}

Task:
Return EXACTLY {max_topics} short topic labels — chapter names, key themes, or important concepts from this document.

Rules:
- Each topic must be 2 to 8 words.
- Use the actual chapter titles or main themes from the text where present.
- Do NOT include: author names, editor names, publisher info, copyright lines, ISBN, production credits, or page numbers.
- Do NOT include: "Introduction", "Conclusion", "References", "Index", "Appendix".
- Return ONLY a plain numbered list. No explanations.

Example output format:
1. How Predictive AI Works
2. Why AI Predictions Can Fail
3. Fairness and Bias in AI Systems
...
""".strip()

    try:
        response = llm.invoke(prompt).strip()
        topics: List[str] = []
        for line in response.splitlines():
            # Strip leading number/bullet.
            cleaned = re.sub(r'^[\d]+[.)\-\s]+', '', line.strip()).strip()
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip()
            if not cleaned:
                continue
            lower = cleaned.lower()
            if "topic label" in lower or lower.startswith("here are"):
                continue
            if _is_noise_topic(cleaned):
                continue
            word_count = len(cleaned.split())
            if word_count < 2 or word_count > 12:
                continue
            topics.append(cleaned)
            if len(topics) >= max_topics:
                break
        return topics
    except Exception:
        return []


def _extract_topics_from_source(text: str, max_topics: int = 8) -> List[str]:
    """Best-effort extraction of chapter/headline-like topics from raw document text."""
    raw = (text or "")
    if not raw.strip():
        return []

    candidates: List[str] = []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # 1) Prefer explicit chapter/section style lines.
    chapter_re = re.compile(r"^(chapter|section|part)\s+[0-9ivx]+[:\-\.]?\s+(.+)$", re.IGNORECASE)
    heading_re = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\s+)?([A-Za-z][A-Za-z0-9 ,:&'\-/]{3,120})$")

    for ln in lines:
        if len(ln) > 140:
            continue

        m = chapter_re.match(ln)
        if m:
            label = m.group(2).strip()
            if label and not _is_noise_topic(label):
                candidates.append(label)
            continue

        # Upper/title-cased compact lines often represent headings.
        if (ln.isupper() or ln.istitle()) and not ln.endswith(('.', '!', '?')):
            hm = heading_re.match(ln)
            if hm:
                label = hm.group(1).strip()
                if not _is_noise_topic(label):
                    candidates.append(label)

    # 2) Fallback: pick high-information noun-like phrases from sentences.
    if len(candidates) < 3:
        compact = re.sub(r"\s+", " ", raw)
        sentence_parts = re.split(r"(?<=[.!?])\s+", compact)
        for sent in sentence_parts:
            s = sent.strip()
            if len(s) < 30 or len(s) > 140:
                continue
            if any(tok in s.lower() for tok in ["introduction", "conclusion", "references"]):
                continue
            phrase = re.sub(r"^[^A-Za-z]+", "", s)
            phrase = re.sub(r"[.!?]+$", "", phrase).strip()
            if 4 <= len(phrase.split()) <= 10:
                if not _is_noise_topic(phrase):
                    candidates.append(phrase)

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        norm = re.sub(r"\s+", " ", item.strip()).lower()
        if not norm or norm in seen or _is_noise_topic(item):
            continue
        seen.add(norm)
        deduped.append(item.strip())
        if len(deduped) >= max_topics:
            break

    return deduped


def _extract_topics_from_summary(summary_text: str, max_topics: int = 8) -> List[str]:
    """Fallback topic extraction from summary bullets (more semantic, less metadata noise)."""
    sections = _extract_sections(summary_text)
    priority_sections = [
        "Main Lessons",
        "Important Points",
        "Key Points",
        "Author's Evidence/Arguments",
        "Evidence from Text",
        "Why It Matters",
    ]

    topics: List[str] = []
    for sec in priority_sections:
        for item in sections.get(sec, []):
            cleaned = re.sub(r"\s+", " ", (item or "").strip())
            if not cleaned:
                continue
            if _is_noise_topic(cleaned):
                continue
            words = cleaned.split()
            if len(words) > 12:
                cleaned = " ".join(words[:12]).rstrip(",;:")
            topics.append(cleaned)
            if len(topics) >= max_topics:
                return topics

    return topics


def _is_noise_topic(text: str) -> bool:
    s = re.sub(r"\s+", " ", (text or "").strip())
    lower = s.lower()
    if not s:
        return True

    # Remove front-matter and production metadata.
    noise_tokens = [
        "all rights reserved", "editorial", "production", "copyeditor", "jacket",
        "copyright", "isbn", "publisher", "press", "printed", "acknowledgments",
        "table of contents", "contents", "dedication", "index",
    ]
    if any(tok in lower for tok in noise_tokens):
        return True

    # Filter likely person-name lines and single shouty words.
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$", s):
        return True
    if s.isupper() and len(s.split()) <= 2:
        return True
    if len(s.split()) < 2:
        return True

    return False