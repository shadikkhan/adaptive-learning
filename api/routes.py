"""
API Routes for AgeXplain
"""
import asyncio
import json
import time
# from pyexpat.errors import messages
import re
import unicodedata
import uuid
import hashlib
from typing import List, Optional, Dict
# from urllib import request
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from graph import learning_graph
from configs.models import ExplainState
from configs.config import llm, DIFFICULTY_MAP, use_request_llm, LLM_MODEL, LLM_TEMPERATURE
from services.document_service import extract_document_text, summarize_with_fallback
from services.rag_service import get_rag_service
from services.model_provider import RuntimeModelConfig, build_runtime_llm
from services.tts_service import synthesize_tts_mp3, get_audio_dir
from services.json_logger import log_event
from db.database import save_document, get_document, get_document_by_content_hash
import os


# Initialize router (no prefix to match frontend expectations)
router = APIRouter(tags=["explain"])


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
QUIZ_GENERATION_TIMEOUT_SECONDS = int(os.getenv("QUIZ_GENERATION_TIMEOUT_SECONDS", "60"))
QUIZ_SOURCE_MAX_CHUNKS = 12
QUIZ_SOURCE_MAX_CHARS_PER_CHUNK = 450

_NUMBER_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def _normalize_ocr_quiz_text(text: str) -> str:
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Common OCR issue where @ appears in the middle of words (e.g., classi@ers).
    cleaned = re.sub(r"(?<=[A-Za-z])@(?=[A-Za-z])", "fi", cleaned)
    # OCR can drop leading letters and produce @ne for fine.
    cleaned = re.sub(r"\B@(?=[A-Za-z])", "fi", cleaned)

    # Convert patterns like /two.o/zero.o/two.o/one.o into 2021.
    def _replace_number_token(match):
        token = match.group(1).lower()
        return _NUMBER_WORD_TO_DIGIT.get(token, token)

    cleaned = re.sub(r"/(zero|one|two|three|four|five|six|seven|eight|nine)\.o", _replace_number_token, cleaned, flags=re.IGNORECASE)

    # Trim OCR suffix artifacts.
    cleaned = re.sub(r"\.PL\b", ".", cleaned)
    cleaned = re.sub(r"\bJI\b", "", cleaned)
    cleaned = re.sub(r"/[A-Za-z]_", "", cleaned)

    # Normalize broken intra-word hyphen spacing (e.g., re - searchers, two- year, at- tempt).
    cleaned = re.sub(r"\b([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})\b", r"\1\2", cleaned)

    # Repair very common split word artifacts seen in OCR/PDF extraction.
    repairs = {
        "researchers": "researchers",
        "different": "different",
        "attempt": "attempt",
        "unemployed": "unemployed",
        "finetuning": "fine-tuning",
    }
    for bad, good in repairs.items():
        cleaned = re.sub(rf"\b{re.escape(bad)}\b", good, cleaned, flags=re.IGNORECASE)

    # Remove accidental doubled words like "is is".
    cleaned = re.sub(r"\b([A-Za-z]{1,20})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Repair frequent OCR split words observed in source documents.
    cleaned = re.sub(r"\bclassi\s*fication\b", "classification", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdiff\s*er\s*ent\b", "different", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btwo\s*year\b", "two-year", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\blife\s*changing\b", "life-changing", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bprob\s*lems\b", "problems", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhere are\b", "There are", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bstate\s*of\s*the\s*art\b", "state-of-the-art", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\barti\s*ficial\b", "artificial", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhe\s+myth\b", "the myth", cleaned, flags=re.IGNORECASE)

    # Remove stray OCR single-letter inserts inside sentences (except pronoun I).
    cleaned = re.sub(r"(?<=\w)\s+[A-HJ-Z]\s+(?=\w)", " ", cleaned)

    # Normalize common OCR citation artifacts and cramped numbering.
    cleaned = re.sub(r"\b[Ff]igure\s*([A-Za-z]?)\s*(\d+(?:\.\d+)?)", r"Figure \1\2", cleaned)
    cleaned = re.sub(r"\b([A-Za-z]+)(\d{1,3})([A-Za-z]+)\b", r"\1 \2 \3", cleaned)
    cleaned = re.sub(r"[\u201c\u201d\"]\s*,?\s*Q[Ll]\b", "", cleaned)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_quiz_response(response: str, required_count: int) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    if not response:
        return questions

    # Split by question markers in structured-text responses.
    question_blocks = re.split(r'\n(?=Q\d+:)', response)

    for block in question_blocks:
        if not block.strip():
            continue

        try:
            q_match = re.search(r'Q\d+:\s*(.+?)\n', block)
            if not q_match:
                continue
            question_text = q_match.group(1).strip()

            options = {}
            for letter in ['A', 'B', 'C', 'D']:
                opt_match = re.search(rf'{letter}\)\s*(.+?)(?=\n|$)', block)
                if opt_match:
                    options[letter] = opt_match.group(1).strip()

            correct_match = re.search(r'Correct:\s*([A-D])', block)
            if not correct_match:
                continue
            correct = correct_match.group(1).strip()

            exp_match = re.search(r'Explanation:\s*(.+?)(?=Q\d+:|$)', block, re.DOTALL)
            explanation = exp_match.group(1).strip() if exp_match else "No explanation provided."

            if question_text and len(options) == 4 and correct:
                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct": correct,
                    "explanation": explanation
                })
        except Exception:
            continue

    # JSON fallback
    if len(questions) < required_count:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            try:
                json_questions = json.loads(response[start_idx:end_idx])
                if isinstance(json_questions, list):
                    for q in json_questions:
                        if isinstance(q, dict):
                            questions.append(q)
            except Exception:
                pass

    return questions


def _clip_chunk_for_quiz(text: str, max_chars: int) -> str:
    cleaned = _normalize_ocr_quiz_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    clipped = cleaned[:max_chars]
    # Prefer ending at sentence boundary to avoid cut-off fragments.
    last_sentence = max(clipped.rfind("."), clipped.rfind("?"), clipped.rfind("!"))
    if last_sentence >= int(max_chars * 0.6):
        return clipped[:last_sentence + 1].strip()

    # Fallback: end at word boundary.
    last_space = clipped.rfind(" ")
    if last_space >= int(max_chars * 0.7):
        return clipped[:last_space].strip()
    return clipped.strip()


def _sanitize_quiz_question(question: Dict[str, str]) -> Dict[str, str]:
    options = question.get("options") or {}
    return {
        "question": _normalize_ocr_quiz_text(question.get("question", "")),
        "options": {k: _normalize_ocr_quiz_text(v) for k, v in options.items()},
        "correct": question.get("correct", ""),
        "explanation": _normalize_ocr_quiz_text(question.get("explanation", "")),
    }


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def _tokens(text: str) -> set:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "have", "has", "were", "was", "are",
        "into", "about", "what", "when", "where", "which", "their", "they", "them", "been", "can",
        "will", "would", "could", "should", "only", "than", "then", "also", "some", "more", "most",
    }
    parts = _normalize_for_match(text).split()
    return {p for p in parts if len(p) >= 4 and p not in stop}


def _has_source_overlap(explanation: str, source_blocks: List[str], min_hits: int = 3) -> bool:
    exp_tokens = _tokens(explanation)
    if not exp_tokens:
        return False
    src_tokens = _tokens(" ".join(source_blocks))
    return len(exp_tokens.intersection(src_tokens)) >= min_hits


def _is_generic_supported_stem(question_text: str) -> bool:
    q = (question_text or "").strip().lower()
    return bool(re.match(r"^(which statement|which option|what statement).*(supported|source passage)", q))


def _is_doc_grounded_question(question: Dict[str, str], source_blocks: List[str]) -> bool:
    question_text = (question.get("question") or "").strip()
    options = question.get("options") or {}
    correct = (question.get("correct") or "").strip().upper()
    explanation = (question.get("explanation") or "").strip()

    if correct not in {"A", "B", "C", "D"}:
        return False
    if len(options) != 4 or correct not in options:
        return False
    if len(explanation) < 24:
        return False

    # Avoid repetitive generic stems that read like evaluator artifacts.
    if _is_generic_supported_stem(question_text):
        return False

    # Avoid weak pronoun-only explanations like "This was first demonstrated..."
    if re.match(r"^(this|that|it)\b", explanation.lower()) and len(_tokens(explanation)) < 8:
        return False

    # Explanation should contain at least one meaningful token from the correct option.
    correct_tokens = _tokens(options.get(correct, ""))
    exp_tokens = _tokens(explanation)
    if correct_tokens and not correct_tokens.intersection(exp_tokens):
        return False

    # Explanation must overlap with source passage vocabulary.
    if not _has_source_overlap(explanation, source_blocks, min_hits=3):
        return False

    # Question text itself should reference source concepts, not generic rubric wording.
    q_tokens = _tokens(question_text)
    src_tokens = _tokens(" ".join(source_blocks))
    if q_tokens and len(q_tokens.intersection(src_tokens)) < 2:
        return False

    return True


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    for p in parts:
        candidate = re.sub(r"^[^A-Za-z0-9]+", "", p.strip())
        candidate = _normalize_ocr_quiz_text(candidate)
        if len(candidate) >= 30:
            return candidate
    return _normalize_ocr_quiz_text(re.sub(r"^[^A-Za-z0-9]+", "", (text or "").strip()))


def _best_support_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    candidates = []

    for p in parts:
        c = _normalize_ocr_quiz_text(re.sub(r"^[^A-Za-z0-9]+", "", p.strip()))
        if len(c) < 35:
            continue
        # Skip OCR-heavy/citation fragments that create unreadable quiz explanations.
        if re.search(r"\bfigure\s*[A-Za-z]?\d", c, flags=re.IGNORECASE):
            continue
        if re.search(r"[A-Za-z]{2,}\d{1,}[A-Za-z]{1,}", c):
            continue
        if re.search(r"\bQ[Ll]\b", c):
            continue
        words = c.split()
        score = 0
        if 8 <= len(words) <= 34:
            score += 2
        if re.search(r"\d", c):
            score += 2
        if re.search(r"\b[A-Z][a-z]{2,}\b", c):
            score += 1
        if re.match(r"^(there are a few|note that|this|that|it)\b", c, flags=re.IGNORECASE):
            score -= 2
        if re.search(r"\bfigure\s*[A-Za-z]?\d", c, flags=re.IGNORECASE):
            score -= 1
        if re.search(r"[A-Za-z]{2,}\d{2,}[A-Za-z]{2,}", c):
            score -= 1
        if "?" in c:
            score -= 1
        candidates.append((score, c))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    return _first_sentence(text)


def _build_source_fallback_question(topic: str, source_blocks: List[str], idx: int) -> Dict[str, object]:
    source = source_blocks[idx % len(source_blocks)] if source_blocks else topic
    sentence = _best_support_sentence(_normalize_ocr_quiz_text(source))
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    if len(sentence) > 170:
        sentence = sentence[:167].rstrip() + "..."

    # Build domain-agnostic distractors by applying controlled contradiction patterns.
    no_numbers = re.sub(r"\d+([.,]\d+)?", "", sentence).strip()
    absolute_claim = re.sub(r"\b(some|many|often|can|may|might)\b", "always", no_numbers, flags=re.IGNORECASE)
    if absolute_claim == no_numbers:
        absolute_claim = f"{no_numbers} This statement is always true with no exceptions."

    impossible_certainty = re.sub(
        r"\b(limited|uncertain|challenge|risk|error|bias|variation|approximate)\b",
        "guaranteed",
        sentence,
        flags=re.IGNORECASE,
    )
    if impossible_certainty == sentence:
        impossible_certainty = "The document claims complete certainty and zero trade-offs."

    unsupported_scope = "The document provides no direct evidence for a specific factual claim in this section."

    question_templates = [
        'According to the source passage about "{topic}", which claim is supported?',
        'Based on the passage, what is explicitly stated about "{topic}"?',
        'Which option best matches the source passage discussion of "{topic}"?',
        'From the provided text on "{topic}", which statement is accurate?',
        'What does the source passage directly say about "{topic}"?',
    ]
    q_text = question_templates[idx % len(question_templates)].format(topic=(topic or "this topic"))

    return {
        "question": q_text,
        "options": {
            "A": sentence,
            "B": absolute_claim,
            "C": impossible_certainty,
            "D": unsupported_scope,
        },
        "correct": "A",
        "explanation": sentence,
    }


# ---- Sample Topics Data ----
TOPICS_DATA = {
    "Science": [
        "Photosynthesis",
        "Solar System",
        "Water Cycle",
        "Gravity",
        "Electricity"
    ],
    "Math": [
        "Fractions",
        "Multiplication",
        "Geometry",
        "Percentages",
        "Algebra Basics"
    ],
    "History": [
        "Ancient Egypt",
        "World War II",
        "Renaissance",
        "Industrial Revolution",
        "American Revolution"
    ],
    "Technology": [
        "How the Internet Works",
        "Artificial Intelligence",
        "Computers",
        "Smartphones",
        "Coding Basics"
    ]
}


# ---- Request/Response Models ----

class LearnerProfileRequest(BaseModel):
    age: int

class MessageRequest(BaseModel):
    role: str
    content: str


class ModelConfigRequest(BaseModel):
    provider: str = "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class ExplainRequest(BaseModel):
    model_config = {"populate_by_name": True}

    topic: str
    age: int
    context: Optional[str] = ""
    doc_id: Optional[str] = None
    user_answer: Optional[str] = None
    profession: Optional[str] = None
    expertise_level: Optional[str] = None
    area_of_interest: Optional[str] = None
    include_examples: bool = True
    include_questions: bool = True
    force_new_topic: bool = False
    client_trace_id: Optional[str] = None
    llm_config: Optional[ModelConfigRequest] = None
    client_model_config: Optional[ModelConfigRequest] = Field(None, alias="model_config")

    def resolved_llm_config(self) -> Optional[ModelConfigRequest]:
        return self.client_model_config or self.llm_config

class LegacyExplainRequest(BaseModel):
    user_input: str
    messages: List[MessageRequest] = []
    learner: LearnerProfileRequest
    intent: Optional[str] = "new_question"
    user_answer: Optional[str] = None

class QuizGenerateRequest(BaseModel):
    model_config = {"populate_by_name": True}

    topic: str
    age: int
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = "medium"
    doc_id: Optional[str] = None
    profession: Optional[str] = None
    expertise_level: Optional[str] = None
    area_of_interest: Optional[str] = None
    client_trace_id: Optional[str] = None
    llm_config: Optional[ModelConfigRequest] = None
    client_model_config: Optional[ModelConfigRequest] = Field(None, alias="model_config")

    def resolved_llm_config(self) -> Optional[ModelConfigRequest]:
        return self.client_model_config or self.llm_config


class DocumentSummaryRequest(BaseModel):
    doc_id: str
    age: int
    profession: Optional[str] = None
    expertise_level: Optional[str] = None
    area_of_interest: Optional[str] = None
    client_trace_id: Optional[str] = None
    llm_config: Optional[ModelConfigRequest] = None


class DocumentAskRequest(BaseModel):
    doc_id: str
    question: str
    age: int
    client_trace_id: Optional[str] = None
    profession: Optional[str] = None
    expertise_level: Optional[str] = None
    area_of_interest: Optional[str] = None
    include_examples: bool = True
    include_questions: bool = True
    llm_config: Optional[ModelConfigRequest] = None


class ValidateModelRequest(BaseModel):
    client_trace_id: Optional[str] = None
    llm_config: Optional[ModelConfigRequest] = None
    prompt: Optional[str] = "Reply with exactly OK"


def _resolve_runtime_llm(llm_config: Optional[ModelConfigRequest]):
    runtime_config = RuntimeModelConfig(
        provider=(llm_config.provider if llm_config else "ollama"),
        model=(llm_config.model if llm_config else None),
        api_key=(llm_config.api_key if llm_config else None),
        base_url=(llm_config.base_url if llm_config else None),
    )
    try:
        return build_runtime_llm(runtime_config, default_model=LLM_MODEL, default_temperature=LLM_TEMPERATURE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _llm_cfg_summary(llm_config: Optional[ModelConfigRequest]) -> str:
    if not llm_config:
        return "provider=ollama model=<default> base_url=<default> api_key_set=False"
    provider = llm_config.provider or "ollama"
    model = llm_config.model or "<default>"
    base_url = llm_config.base_url or "<default>"
    has_key = bool((llm_config.api_key or "").strip())
    return f"provider={provider} model={model} base_url={base_url} api_key_set={has_key}"


def _llm_cfg_fields(llm_config: Optional[ModelConfigRequest]) -> Dict[str, object]:
    if not llm_config:
        return {
            "llm_provider": "ollama",
            "llm_model": "<default>",
            "llm_base_url": "<default>",
            "llm_api_key_set": False,
        }
    return {
        "llm_provider": llm_config.provider or "ollama",
        "llm_model": llm_config.model or "<default>",
        "llm_base_url": llm_config.base_url or "<default>",
        "llm_api_key_set": bool((llm_config.api_key or "").strip()),
    }


def _explain_request_summary(request: ExplainRequest) -> str:
    return (
        f"topic={request.topic!r} topic_chars={len(request.topic or '')} "
        f"age={request.age} doc_id={request.doc_id} "
        f"profession={request.profession!r} expertise_level={request.expertise_level!r} "
        f"area_of_interest={request.area_of_interest!r} "
        f"include_examples={request.include_examples} include_questions={request.include_questions} "
        f"force_new_topic={request.force_new_topic} context_chars={len(request.context or '')} "
        f"user_answer_present={bool((request.user_answer or '').strip())} "
        f"client_trace_id={request.client_trace_id or '<none>'}"
    )


def _quiz_request_summary(request: QuizGenerateRequest) -> str:
    return (
        f"topic={request.topic!r} age={request.age} num_questions={request.num_questions} "
        f"difficulty={request.difficulty} doc_id={request.doc_id} "
        f"profession={request.profession!r} expertise_level={request.expertise_level!r} "
        f"area_of_interest={request.area_of_interest!r} "
        f"client_trace_id={request.client_trace_id or '<none>'}"
    )


def _validate_request_summary(request: ValidateModelRequest) -> str:
    return (
        f"prompt_chars={len((request.prompt or '').strip())} "
        f"client_trace_id={request.client_trace_id or '<none>'}"
    )


def _doc_summary_request_summary(request: DocumentSummaryRequest) -> str:
    return (
        f"doc_id={request.doc_id} age={request.age} "
        f"client_trace_id={request.client_trace_id or '<none>'}"
    )


def _doc_ask_request_summary(request: DocumentAskRequest) -> str:
    return (
        f"doc_id={request.doc_id} age={request.age} question_chars={len(request.question or '')} "
        f"profession={request.profession!r} expertise_level={request.expertise_level!r} "
        f"area_of_interest={request.area_of_interest!r} include_examples={request.include_examples} "
        f"include_questions={request.include_questions} client_trace_id={request.client_trace_id or '<none>'}"
    )


def _model_config_from_form(
    model_provider: str,
    model_name: Optional[str],
    model_api_key: Optional[str],
    model_base_url: Optional[str],
) -> ModelConfigRequest:
    return ModelConfigRequest(
        provider=(model_provider or "ollama").strip().lower(),
        model=(model_name or None),
        api_key=(model_api_key or None),
        base_url=(model_base_url or None),
    )


# ---- API Endpoints ----

@router.get("/topics")
async def get_topics():
    """
    Get available topic packs for learning.
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event("topics.start", trace_id=trace_id)
    try:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("topics.end", trace_id=trace_id, packs=len(TOPICS_DATA), elapsed_ms=elapsed_ms)
        return TOPICS_DATA
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("topics.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(exc))
        raise


@router.post("/quiz/generate")
async def generate_quiz(request: QuizGenerateRequest):
    """
    Generate a quiz on a given topic with multiple choice questions.
    """
    quiz_trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    llm_cfg = request.resolved_llm_config()
    log_event(
        "quiz.generate.start",
        trace_id=quiz_trace_id,
        topic=request.topic,
        age=request.age,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        doc_id=request.doc_id or "<none>",
        profession=request.profession or "<none>",
        expertise_level=request.expertise_level or "<none>",
        area_of_interest=request.area_of_interest or "<none>",
        client_trace_id=request.client_trace_id or "<none>",
        **_llm_cfg_fields(llm_cfg),
    )
    try:
        runtime_llm = _resolve_runtime_llm(llm_cfg)
        difficulty_level = DIFFICULTY_MAP.get(request.difficulty, "moderate difficulty with some critical thinking")
        profession = (request.profession or "").strip()
        expertise_level = (request.expertise_level or "").strip()
        area_of_interest = (request.area_of_interest or "").strip()
        learner_profile = (
            "Learner profile:\n"
            f"- Profession: {profession or 'Not provided'}\n"
            f"- Expertise level: {expertise_level or 'Not provided'}\n"
            f"- Area of interest: {area_of_interest or 'Not provided'}"
        )
        interest_rule = (
            f"12. If helpful, lightly frame wording/examples using '{area_of_interest}' as a familiar context, "
            "but NEVER change the core topic or factual answer. If it does not fit naturally, ignore it."
            if area_of_interest
            else "12. Keep wording neutral unless a learner area of interest is provided."
        )

        doc_context = ""
        source_section = ""
        source_blocks: List[str] = []
        verbatim_chunks: List[str] = []
        if request.doc_id:
            doc = get_document(request.doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")

            # Get verbatim chunks already stored in RAG index — actual sentences from the document.
            # Sample evenly across all chunks so we cover the whole paper, not just one section.
            rag_svc = get_rag_service()
            if request.doc_id not in rag_svc.indices:
                rag_svc.index_document(request.doc_id, (doc.get("text") or "").strip())

            all_chunks = rag_svc.indices[request.doc_id].chunks if request.doc_id in rag_svc.indices else []
            if all_chunks:
                # Use topic-focused retrieval so quiz questions stay on the requested topic.
                rag_index = rag_svc.indices[request.doc_id]
                query = request.topic or "key concepts"
                retrieved = rag_index.retrieve(query, top_k=QUIZ_SOURCE_MAX_CHUNKS)
                if retrieved:
                    sampled = [chunk for chunk, _score in retrieved]
                else:
                    # Fallback: evenly-spaced sampling if retrieval returns nothing
                    n_chunks = len(all_chunks)
                    step = max(1, n_chunks // QUIZ_SOURCE_MAX_CHUNKS)
                    sampled = [all_chunks[i] for i in range(0, n_chunks, step)][:QUIZ_SOURCE_MAX_CHUNKS]
                verbatim_chunks = [
                    _clip_chunk_for_quiz(chunk, QUIZ_SOURCE_MAX_CHARS_PER_CHUNK)
                    for chunk in sampled
                    if chunk.strip()
                ]
                source_blocks = verbatim_chunks[:]
            else:
                # Fallback: use raw text
                doc_text = (doc.get("text") or "").strip()
                doc_context = _normalize_ocr_quiz_text(doc_text[:10000])
                source_blocks = [doc_context] if doc_context else []

        if verbatim_chunks or doc_context:
            # Build source passages numbered list from verbatim chunk text
            if verbatim_chunks:
                passages = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(verbatim_chunks))
                source_section = f"SOURCE PASSAGES (verbatim from the document):\n{passages}"
            else:
                source_section = f"DOCUMENT TEXT:\n{doc_context}"

            prompt = f"""You are writing a quiz focused on the topic: "{request.topic}". Every question and every answer must come DIRECTLY from the source passages below.

{learner_profile}

RULES (follow exactly):
1. The quiz topic is "{request.topic}". Only write questions relevant to this topic.
2. Read the source passages below word by word.
3. Write questions that ask about facts explicitly stated in those passages AND related to the topic.
4. The correct answer must be a fact directly stated in the passages.
5. Wrong answers (distractors) must be plausible but NOT from the passages.
6. Do NOT use any outside knowledge or prior beliefs about any term or name.
7. Treat every term in the passages as a technical/domain concept only.
8. Clean obvious OCR noise in your wording (e.g., classi@ers -> classifiers, /two.o/zero.o/two.o/one.o -> 2021).
9. Difficulty: {difficulty_level}
10. Do NOT use generic stems like "Which statement is explicitly supported by the provided source passage?".
11. Prefer concrete chapter-specific questions (definitions, claims, results, examples, comparisons).
12. Age-appropriate language:
    - If age <= 7: very short questions, everyday words only, no technical terms.
    - If 8 <= age <= 10: simple words, one technical term max.
    - If 11 <= age <= 14: school-level phrasing, conceptual questions OK.
    - If 15 <= age <= 18: domain vocabulary welcome, cause-effect or comparison questions OK.
    - If age >= 19: adult-level language, precise terminology welcome.
    - If expertise is Beginner: prefer recall or definitional questions regardless of age.
    - If expertise is Advanced and age >= 19: prefer analytical or application questions.
{interest_rule}

{source_section}

Create exactly {request.num_questions} questions in this EXACT format:

Q1: [Question about a specific fact in the passages]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct: [A, B, C, or D]
Explanation: [A clear, readable supporting sentence that includes key words from the correct option, not just pronouns like "this"]

Continue for all {request.num_questions} questions."""
        else:
            prompt = f"""Create {request.num_questions} quiz questions about {request.topic} for age {request.age}.

{learner_profile}

Rules:
- Difficulty: {difficulty_level}
- Keep all questions on topic.
- If area of interest is provided, you may lightly use it as a familiar context in wording only; do not change facts.
- Age-appropriate language:
    - If age <= 7: very short questions, everyday words only, no technical terms.
    - If 8 <= age <= 10: simple words, one technical term max.
    - If 11 <= age <= 14: school-level phrasing, conceptual questions OK.
    - If 15 <= age <= 18: domain vocabulary welcome, cause-effect or comparison questions OK.
    - If age >= 19: adult-level language, precise terminology welcome.
    - If expertise is Beginner: prefer recall or definitional questions regardless of age.
    - If expertise is Advanced and age >= 19: prefer analytical or application questions.

For each question, write in this EXACT format:

Q1: [Question text]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
Correct: [A, B, C, or D]
Explanation: [Why this answer is correct]

Q2: [Question text]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
Correct: [A, B, C, or D]
Explanation: [Why this answer is correct]

Continue for all {request.num_questions} questions."""
        
        try:
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(runtime_llm.invoke, prompt),
                timeout=QUIZ_GENERATION_TIMEOUT_SECONDS,
            )
            response = _normalize_ocr_quiz_text((raw_response or "").strip())
        except asyncio.TimeoutError:
            return {
                "trace_id": quiz_trace_id,
                "client_trace_id": request.client_trace_id,
                "topic": request.topic,
                "questions": [{
                    "question": f"What would you like to learn first about {request.topic}?",
                    "options": {
                        "A": "Core concepts",
                        "B": "Important terms",
                        "C": "Real-world examples",
                        "D": "All of these"
                    },
                    "correct": "D",
                    "explanation": "Quiz generation timed out. This fallback lets you continue without waiting indefinitely."
                }]
            }
        log_event("quiz.generate.llm_response", trace_id=quiz_trace_id, preview=response[:500])
        
        # Parse LLM output (structured text first, then JSON fallback).
        questions = _parse_quiz_response(response, request.num_questions)

        if questions:
            questions = [_sanitize_quiz_question(q) for q in questions]

        # For document quizzes, enforce grounding so weak/ambiguous items are not returned.
        if request.doc_id and source_blocks:
            grounded = [q for q in questions if _is_doc_grounded_question(q, source_blocks)]
            questions = grounded

            # If we are missing items, run one focused retry generation before using deterministic fallback.
            if len(questions) < request.num_questions:
                missing = request.num_questions - len(questions)
                existing_q_text = "\n".join(
                    f"- {q.get('question', '')}" for q in questions if q.get("question")
                )
                retry_prompt = f"""Generate exactly {missing} additional multiple-choice quiz questions from the source passages below.

Topic: {request.topic}
Difficulty: {difficulty_level}
{learner_profile}

Rules:
1. Questions must be directly supported by the source passages.
2. Questions must be specific to topic "{request.topic}".
3. Do not repeat any existing questions.
4. Do not use generic stems like "Which statement is explicitly supported by the provided source passage?".
5. Explanation must be one clear sentence grounded in passage wording.
6. If area of interest is provided, you may use it only as light framing. Do not change core facts.
7. Age-appropriate language:
    - If age <= 7: very short questions, everyday words only, no technical terms.
    - If 8 <= age <= 10: simple words, one technical term max.
    - If 11 <= age <= 14: school-level phrasing, conceptual questions OK.
    - If 15 <= age <= 18: domain vocabulary welcome, cause-effect or comparison questions OK.
    - If age >= 19: adult-level language, precise terminology welcome.
    - If expertise is Beginner: prefer recall or definitional questions regardless of age.
    - If expertise is Advanced and age >= 19: prefer analytical or application questions.

Existing questions (do not repeat):
{existing_q_text or "- none"}

{source_section}

Output EXACT format:
Q1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct: [A, B, C, or D]
Explanation: [Grounded supporting sentence]
"""
                try:
                    retry_raw = await asyncio.wait_for(
                        asyncio.to_thread(runtime_llm.invoke, retry_prompt),
                        timeout=max(20, QUIZ_GENERATION_TIMEOUT_SECONDS // 2),
                    )
                    retry_text = _normalize_ocr_quiz_text((retry_raw or "").strip())
                    retry_questions = _parse_quiz_response(retry_text, missing)
                    retry_questions = [_sanitize_quiz_question(q) for q in retry_questions]
                    retry_grounded = [q for q in retry_questions if _is_doc_grounded_question(q, source_blocks)]

                    seen = {_normalize_for_match(q.get("question", "")) for q in questions}
                    for q in retry_grounded:
                        key = _normalize_for_match(q.get("question", ""))
                        if key and key not in seen:
                            questions.append(q)
                            seen.add(key)
                            if len(questions) >= request.num_questions:
                                break
                except Exception as retry_error:
                    log_event("quiz.generate.retry_skipped", level="WARNING", trace_id=quiz_trace_id, error=str(retry_error))

            while len(questions) < request.num_questions:
                questions.append(_build_source_fallback_question(request.topic, source_blocks, len(questions)))
        
        if len(questions) == 0:
            # Last resort: create a simple question
            questions = [{
                "question": f"What is an important concept related to {request.topic}?",
                "options": {
                    "A": "Option A - Please try a different topic",
                    "B": "Option B - The AI couldn\'t generate proper questions",
                    "C": "Option C - Please try again",
                    "D": "Option D - Check your connection"
                },
                "correct": "A",
                "explanation": f"The system had trouble generating questions about {request.topic}. Please try a different topic or try again."
            }]
        
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("quiz.generate.end", trace_id=quiz_trace_id, questions=len(questions[:request.num_questions]), elapsed_ms=elapsed_ms)
        return {
            "trace_id": quiz_trace_id,
            "client_trace_id": request.client_trace_id,
            "topic": request.topic,
            "questions": questions[:request.num_questions]
        }
            
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("quiz.generate.error", level="ERROR", trace_id=quiz_trace_id, elapsed_ms=elapsed_ms, error=str(e))
        # Return a fallback question instead of error
        return {
            "trace_id": quiz_trace_id,
            "client_trace_id": request.client_trace_id,
            "topic": request.topic,
            "questions": [{
                "question": f"What would you like to learn about {request.topic}?",
                "options": {
                    "A": "The basics and fundamentals",
                    "B": "Advanced concepts",
                    "C": "Real-world applications",
                    "D": "All of the above"
                },
                "correct": "D",
                "explanation": "All aspects of learning are important! The quiz generator encountered an issue, but you can continue learning by asking questions in the chat."
            }]
        }


@router.post("/models/validate")
async def validate_model(request: ValidateModelRequest):
    """Validate that the selected model config can produce a response."""
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event(
        "models.validate.start",
        trace_id=trace_id,
        prompt_chars=len((request.prompt or "").strip()),
        client_trace_id=request.client_trace_id or "<none>",
        **_llm_cfg_fields(request.llm_config),
    )
    try:
        runtime_llm = _resolve_runtime_llm(request.llm_config)
        probe_prompt = (request.prompt or "Reply with exactly OK").strip()
        raw_response = await asyncio.wait_for(
            asyncio.to_thread(runtime_llm.invoke, probe_prompt),
            timeout=25,
        )
        response = (raw_response or "").strip()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("models.validate.end", trace_id=trace_id, ok=True, preview_chars=len(response[:120]), elapsed_ms=elapsed_ms)
        return {
            "ok": True,
            "trace_id": trace_id,
            "client_trace_id": request.client_trace_id,
            "provider": request.llm_config.provider if request.llm_config else "ollama",
            "model": request.llm_config.model if request.llm_config else LLM_MODEL,
            "preview": response[:120],
        }
    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("models.validate.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error="timeout")
        raise HTTPException(status_code=504, detail="Model request timed out") from exc
    except HTTPException:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("models.validate.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error="http_exception")
        raise
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("models.validate.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    age: int = Form(10),
    profession: Optional[str] = Form(None),
    expertise_level: Optional[str] = Form(None),
    area_of_interest: Optional[str] = Form(None),
    llm_provider: str = Form("ollama"),
    llm_name: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    client_trace_id: Optional[str] = Form(None),
):
    """
    Upload a document and return an age-adapted summary.
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event(
        "documents.upload.start",
        trace_id=trace_id,
        filename=file.filename,
        age=age,
        profession=profession or "<none>",
        expertise_level=expertise_level or "<none>",
        area_of_interest=area_of_interest or "<none>",
        size_bytes=getattr(file, "size", None),
        client_trace_id=client_trace_id or "<none>",
        **_llm_cfg_fields(_model_config_from_form(llm_provider, llm_name, llm_api_key, llm_base_url)),
    )
    try:
        model_config = _model_config_from_form(
            model_provider=llm_provider,
            model_name=llm_name,
            model_api_key=llm_api_key,
            model_base_url=llm_base_url,
        )
        runtime_llm = _resolve_runtime_llm(model_config)

        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Max size is 20MB")

        extracted_text, parser = extract_document_text(file.filename or "document", file.content_type or "", payload)
        cleaned_text = (extracted_text or "").strip()
        if len(cleaned_text) < 80:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough readable text from this file"
            )

        # Deduplication: reuse existing doc if exact same file content already in DB.
        content_hash = hashlib.sha256(payload).hexdigest()
        filename = file.filename or "document"
        existing = get_document_by_content_hash(content_hash)
        if existing:
            doc_id = existing["doc_id"]
            rag_service = get_rag_service()
            if doc_id not in rag_service.indices:
                rag_service.index_document(doc_id, existing["text"])
            summary, summary_mode = summarize_with_fallback(
                existing["text"],
                age,
                runtime_llm,
                profession=profession or "",
                expertise_level=expertise_level or "",
                area_of_interest=area_of_interest or "",
            )
            audio_url = None
            try:
                audio_path = await asyncio.to_thread(synthesize_tts_mp3, summary)
                if audio_path:
                    audio_url = f"/audio/{audio_path.name}"
            except Exception as tts_exc:
                log_event("documents.upload.tts_error", level="WARNING", trace_id=trace_id, error=str(tts_exc))
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            log_event("documents.upload.end", trace_id=trace_id, reused=True, doc_id=doc_id, elapsed_ms=elapsed_ms)
            return {
                "trace_id": trace_id,
                "client_trace_id": client_trace_id,
                "doc_id": doc_id,
                "filename": filename,
                "parser": existing["parser"],
                "summary_mode": summary_mode,
                "char_count": len(existing["text"]),
                "chunk_count": len(rag_service.indices[doc_id].chunks) if doc_id in rag_service.indices else 0,
                "summary": summary,
                "audio_url": audio_url,
                "reused": True,
            }

        summary, summary_mode = summarize_with_fallback(
            cleaned_text,
            age,
            runtime_llm,
            profession=profession or "",
            expertise_level=expertise_level or "",
            area_of_interest=area_of_interest or "",
        )
        audio_url = None
        try:
            audio_path = await asyncio.to_thread(synthesize_tts_mp3, summary)
            if audio_path:
                audio_url = f"/audio/{audio_path.name}"
        except Exception as tts_exc:
            log_event("documents.upload.tts_error", level="WARNING", trace_id=trace_id, error=str(tts_exc))

        doc_id = str(uuid.uuid4())
        save_document(doc_id, filename, cleaned_text, parser, content_hash=content_hash)

        # Index document for RAG
        rag_service = get_rag_service()
        chunk_count = rag_service.index_document(doc_id, cleaned_text)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.upload.end", trace_id=trace_id, reused=False, doc_id=doc_id, chunk_count=chunk_count, elapsed_ms=elapsed_ms)

        return {
            "trace_id": trace_id,
            "client_trace_id": client_trace_id,
            "doc_id": doc_id,
            "filename": file.filename,
            "parser": parser,
            "summary_mode": summary_mode,
            "char_count": len(cleaned_text),
            "chunk_count": chunk_count,
            "summary": summary,
            "audio_url": audio_url,
        }
    except HTTPException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.upload.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(exc.detail))
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.upload.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/summarize")
async def summarize_document(request: DocumentSummaryRequest):
    """
    Re-summarize an already uploaded document, optionally for a different age.
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event(
        "documents.summarize.start",
        trace_id=trace_id,
        doc_id=request.doc_id,
        age=request.age,
        profession=request.profession or "<none>",
        expertise_level=request.expertise_level or "<none>",
        area_of_interest=request.area_of_interest or "<none>",
        client_trace_id=request.client_trace_id or "<none>",
        **_llm_cfg_fields(request.llm_config),
    )
    doc = get_document(request.doc_id)
    if not doc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.summarize.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error="document_not_found")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        runtime_llm = _resolve_runtime_llm(request.llm_config)
        summary, summary_mode = summarize_with_fallback(
            doc["text"],
            request.age,
            runtime_llm,
            profession=request.profession or "",
            expertise_level=request.expertise_level or "",
            area_of_interest=request.area_of_interest or "",
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.summarize.end", trace_id=trace_id, mode=summary_mode, elapsed_ms=elapsed_ms)
        return {
            "trace_id": trace_id,
            "client_trace_id": request.client_trace_id,
            "doc_id": request.doc_id,
            "filename": doc["filename"],
            "summary_mode": summary_mode,
            "summary": summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.summarize.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/ask")
async def ask_document(request: DocumentAskRequest):
    """
    Ask a question about an uploaded document using the same tutor flow.
    Returns explanation, example, question prompt, and source citations.
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event(
        "documents.ask.start",
        trace_id=trace_id,
        doc_id=request.doc_id,
        age=request.age,
        question_chars=len(request.question or ""),
        profession=request.profession or "<none>",
        expertise_level=request.expertise_level or "<none>",
        area_of_interest=request.area_of_interest or "<none>",
        include_examples=request.include_examples,
        include_questions=request.include_questions,
        client_trace_id=request.client_trace_id or "<none>",
        **_llm_cfg_fields(request.llm_config),
    )
    doc = get_document(request.doc_id)
    if not doc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.ask.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error="document_not_found")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        runtime_llm = _resolve_runtime_llm(request.llm_config)
        rag_service = get_rag_service()
        if request.doc_id not in rag_service.indices:
            rag_service.index_document(request.doc_id, doc["text"])

        initial_state = {
            "user_input": request.question,
            "messages": [{"role": "user", "content": request.question}],
            "learner": {
                "age": request.age,
                "difficulty": "medium",
                "profession": request.profession,
                "expertise_level": request.expertise_level,
                "area_of_interest": request.area_of_interest,
            },
            "intent": "document_question",
            "doc_id": request.doc_id,
            "include_examples": request.include_examples,
            "include_questions": request.include_questions,
            "force_new_topic": False,
            "retrieved_context": None,
            "rag_sources": None,
            "simplified_explanation": None,
            "example": None,
            "safe_text": None,
            "safety_checked_text": None,
            "thought_question": None,
            "quiz_question": None,
            "quiz_options": None,
            "correct_answer": None,
            "user_answer": None,
            "score": None,
            "final_output": None,
            "feedback": None,
        }

        with use_request_llm(runtime_llm):
            result = await learning_graph.ainvoke(initial_state)
        final_output = result.get("final_output") or {}
        sources = result.get("rag_sources") or []

        explanation = final_output.get("explanation") or result.get("safe_text") or result.get("simplified_explanation") or ""
        example = final_output.get("example") or result.get("example") or ""
        think_question = final_output.get("think_question") or result.get("thought_question") or ""

        # Preserve legacy answer field for compatibility with existing clients.
        answer = explanation

        audio_url = None
        tts_payload = "\n\n".join(
            f"{label}: {value}" for label, value in [
                ("Explanation", explanation),
                ("Example", example),
                ("Question", think_question),
            ] if (value or "").strip()
        )
        if tts_payload:
            try:
                audio_path = await asyncio.to_thread(synthesize_tts_mp3, tts_payload)
                if audio_path:
                    audio_url = f"/audio/{audio_path.name}"
            except Exception as tts_exc:
                log_event("documents.ask.tts_error", level="WARNING", trace_id=trace_id, error=str(tts_exc))

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            "documents.ask.end",
            trace_id=trace_id,
            intent=final_output.get("intent", "document_question"),
            source_count=len(sources),
            elapsed_ms=elapsed_ms,
        )

        return {
            "trace_id": trace_id,
            "client_trace_id": request.client_trace_id,
            "doc_id": request.doc_id,
            "question": request.question,
            "answer": answer,
            "intent": final_output.get("intent", "document_question"),
            "explanation": explanation,
            "example": example,
            "think_question": think_question,
            "audio_url": audio_url,
            "sources": sources,
            "source_count": len(sources),
        }
    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("documents.ask.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain/stream")
async def explain_stream(request: ExplainRequest):
    """
    Stream the explanation response in real-time as the graph processes through nodes.
    This endpoint uses Server-Sent Events (SSE) to stream updates as the LangGraph
    processes through each agent node.
    
    Frontend-compatible version that accepts {topic, age, context}
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    llm_cfg = request.resolved_llm_config()
    log_event(
        "explain.stream.start",
        trace_id=trace_id,
        topic=request.topic,
        topic_chars=len(request.topic or ""),
        age=request.age,
        doc_id=request.doc_id or "<none>",
        profession=request.profession or "<none>",
        expertise_level=request.expertise_level or "<none>",
        area_of_interest=request.area_of_interest or "<none>",
        include_examples=request.include_examples,
        include_questions=request.include_questions,
        force_new_topic=request.force_new_topic,
        context_chars=len(request.context or ""),
        user_answer_present=bool((request.user_answer or "").strip()),
        client_trace_id=request.client_trace_id or "<none>",
        **_llm_cfg_fields(llm_cfg),
    )
    try:
        runtime_llm = _resolve_runtime_llm(llm_cfg)
        # Parse context into messages if provided
        messages = []
        if request.context and not request.force_new_topic:
    # Always parse the full context first (do NOT collapse to only Question)
           context_lines = request.context.strip().split('\n\n')
           for line in context_lines:
              if line.startswith('User: '):
                 messages.append({"role": "user", "content": line[6:]})
              elif (
                  line.startswith('Explanation: ')
                  or line.startswith('Example: ')
                  or line.startswith('Question: ')
                  or line.startswith('Feedback: ')
               ):
            # Combine assistant sections into one assistant message block
                  if not messages or messages[-1]["role"] != "assistant":
                     messages.append({"role": "assistant", "content": line})
                  else:
                      messages[-1]["content"] += "\n" + line
        
     
        current_question = None
        # Prefer the LAST question from parsed assistant messages
        for msg in reversed(messages):
          if msg.get("role") != "assistant":
            continue
          content = msg.get("content", "")
          if "Question:" in content:
             current_question = content.rsplit("Question:", 1)[1].strip()
             break

# Fallback: parse from raw context using last occurrence
        if not request.force_new_topic and not current_question and request.context and "Question:" in request.context:
            current_question = request.context.rsplit("Question:", 1)[1].strip()

# Trim accidental extra content after the question
        if current_question:
            for marker in ["\nUser:", "\nExplanation:", "\nExample:", "\nFeedback:"]:
                 if marker in current_question:
                     current_question = current_question.split(marker, 1)[0].strip() 
   
        log_event(
            "explain.stream.context",
            trace_id=trace_id,
            current_question=current_question,
            context_chars=len(request.context or ""),
            user_answer_present=bool((request.user_answer or "").strip()),
            parsed_messages=len(messages),
        )

        # Add current user input
        messages.append({"role": "user", "content": request.topic})
        
        # Convert request to ExplainState
        initial_state = {
            "user_input": request.topic,
            "messages": messages,
            "learner": {
                "age": request.age,
                "difficulty": "medium",
                "profession": request.profession,
                "expertise_level": request.expertise_level,
                "area_of_interest": request.area_of_interest,
            },
            "intent": "new_question",
            "doc_id": request.doc_id,
            "include_examples": request.include_examples,
            "include_questions": request.include_questions,
            "force_new_topic": request.force_new_topic,
            "model_provider": llm_cfg.provider if llm_cfg else "ollama",
            "model_name": llm_cfg.model if llm_cfg else LLM_MODEL,
            "retrieved_context": None,
            "rag_sources": None,
            "simplified_explanation": None,
            "example": None,
            "safety_checked_text": None,
            "thought_question": None,
            "current_question": current_question, 
            "quiz_question": None,
            "quiz_options": None,
            "correct_answer": None,
            "user_answer": request.user_answer,
            "score": None,
            "final_output": None,
            "feedback": None,
        }
         
        async def event_generator():
            """Generate Server-Sent Events from the graph stream"""
            current_section = None
            sections = {
                "Explanation": "",
                "Example": "",
                "Question": "",
                "Feedback": ""
            }
            
            try:
                # Stream events from the graph
                resolved_intent = initial_state.get("intent", "new_question")
                # Set the request-scoped LLM inside the generator so it is active
                # in the same async context where the graph nodes execute.
                with use_request_llm(runtime_llm):
                    async for event in learning_graph.astream(initial_state):
                        for node_name, node_output in event.items():
                            if node_name == "infer_intent" and "intent" in node_output:
                                resolved_intent = node_output["intent"]

                            # For all later nodes, use the resolved intent (not stale initial intent)
                            current_intent = node_output.get("intent", resolved_intent)
                            log_event(
                                "explain.stream.node",
                                level="DEBUG",
                                trace_id=trace_id,
                                node=node_name,
                                resolved_intent=resolved_intent,
                                node_intent=node_output.get("intent"),
                            )
                            if node_name == "safety" and "safe_text" in node_output:
                                if current_intent not in ["answer", "quiz"]:
                                    current_section = "Explanation"
                                    yield f"data: {json.dumps({'type': 'section', 'section': 'Explanation'})}\n\n"
                                    text = node_output.get("safe_text") or ""
                                    sections["Explanation"] = text
                                    yield f"data: {json.dumps({'type': 'update', 'section': 'Explanation', 'text': text})}\n\n"

                                    # If safety disables examples/questions (strict mode), clear previously streamed text.
                                    if node_output.get("include_examples") is False:
                                        sections["Example"] = ""
                                        yield f"data: {json.dumps({'type': 'update', 'section': 'Example', 'text': ''})}\n\n"
                                    if node_output.get("include_questions") is False:
                                        sections["Question"] = ""
                                        yield f"data: {json.dumps({'type': 'update', 'section': 'Question', 'text': ''})}\n\n"

                            if node_name == "simplify" and "simplified_explanation" in node_output:
                                if current_intent not in ["answer", "quiz"]:
                                    current_section = "Explanation"
                                    yield f"data: {json.dumps({'type': 'section', 'section': 'Explanation'})}\n\n"
                                    text = node_output["simplified_explanation"]
                                    sections["Explanation"] = text
                                    yield f"data: {json.dumps({'type': 'update', 'section': 'Explanation', 'text': text})}\n\n"
                            # current_section = "Explanation"
                            # yield f"data: {json.dumps({'type': 'section', 'section': 'Explanation'})}\n\n"
                            
                            # Send the full content
                            # text = node_output["simplified_explanation"]
                            # sections["Explanation"] = text
                            # yield f"data: {json.dumps({'type': 'update', 'section': 'Explanation', 'text': text})}\n\n"
                        
                            elif node_name == "generate_example" and "example" in node_output:
                                if current_intent not in ["answer", "quiz"]:
                                    current_section = "Example"
                                    yield f"data: {json.dumps({'type': 'section', 'section': 'Example'})}\n\n"
                                    text = node_output["example"]
                                    sections["Example"] = text
                                    yield f"data: {json.dumps({'type': 'update', 'section': 'Example', 'text': text})}\n\n"
                            # current_section = "Explanation"
                            # current_section = "Example"
                            # yield f"data: {json.dumps({'type': 'section', 'section': 'Example'})}\n\n"
                            
                            # text = node_output["example"]
                            # sections["Example"] = text
                            # yield f"data: {json.dumps({'type': 'update', 'section': 'Example', 'text': text})}\n\n"
                        
                            elif node_name == "think" and "thought_question" in node_output:
                                if current_intent not in ["answer", "quiz"]:
                                    current_section = "Question"
                                    yield f"data: {json.dumps({'type': 'section', 'section': 'Question'})}\n\n"
                                    text = node_output["thought_question"]
                                    sections["Question"] = text
                                    yield f"data: {json.dumps({'type': 'update', 'section': 'Question', 'text': text})}\n\n"
    
                            # current_section = "Question"
                            # yield f"data: {json.dumps({'type': 'section', 'section': 'Question'})}\n\n"
                            
                            # text = node_output["thought_question"]
                            # sections["Question"] = text
                            # yield f"data: {json.dumps({'type': 'update', 'section': 'Question', 'text': text})}\n\n"
                        
                            elif node_name == "answer_feedback" and "feedback" in node_output:
                                current_section = "Feedback"
                                yield f"data: {json.dumps({'type': 'section', 'section': 'Feedback'})}\n\n"
                                text = node_output["feedback"]
                                sections["Feedback"] = text
                                yield f"data: {json.dumps({'type': 'update', 'section': 'Feedback', 'text': text})}\n\n"
                
                # Send completion signal
                tts_payload = "\n\n".join(
                    f"{label}: {value}" for label, value in sections.items() if (value or "").strip()
                )
                if tts_payload:
                    try:
                        audio_path = await asyncio.to_thread(synthesize_tts_mp3, tts_payload)
                        if audio_path:
                            audio_event = {
                                "type": "audio",
                                "url": f"/audio/{audio_path.name}",
                                "trace_id": trace_id,
                                "client_trace_id": request.client_trace_id,
                            }
                            yield f"data: {json.dumps(audio_event)}\n\n"
                    except Exception as tts_exc:
                        log_event("explain.stream.tts_error", level="ERROR", trace_id=trace_id, error=str(tts_exc))

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("explain.stream.end", trace_id=trace_id, elapsed_ms=elapsed_ms)
                yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id, 'client_trace_id': request.client_trace_id})}\n\n"
                
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("explain.stream.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
                error_event = {
                    "type": "error",
                    "error": str(e),
                    "trace_id": trace_id,
                    "client_trace_id": request.client_trace_id,
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("explain.stream.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain(request: LegacyExplainRequest):
    """
    Non-streaming endpoint that returns the complete result after processing.
    """
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    log_event(
        "explain.start",
        trace_id=trace_id,
        user_input_chars=len(request.user_input or ""),
        messages=len(request.messages or []),
        learner_age=request.learner.age,
        intent=request.intent,
    )
    try:
        # Convert request to ExplainState
        initial_state = {
            "user_input": request.user_input,
            "messages": [
                {"role": msg.role, "content": msg.content} 
                for msg in request.messages
            ],
            "learner": {"age": request.learner.age, "difficulty": "medium"},
            "intent": request.intent,
            "simplified_explanation": None,
            "example": None,
            "safe_text": None,
            "safety_checked_text": None,
            "thought_question": None,
            "quiz_question": None,
            "quiz_options": None,
            "correct_answer": None,
            "user_answer": request.user_answer,
            "score": None,
            "final_output": None,
            "feedback": None,
        }

        # Invoke the graph and wait for completion
        result = await learning_graph.ainvoke(initial_state)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("explain.end", trace_id=trace_id, elapsed_ms=elapsed_ms)
        return {
            "success": True,
            "trace_id": trace_id,
            "data": result
        }
    
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("explain.error", level="ERROR", trace_id=trace_id, elapsed_ms=elapsed_ms, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.options("/audio/{file_name}")
async def audio_options(file_name: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
    )


@router.get("/audio/{file_name}")
async def get_audio(file_name: str):
    trace_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    safe_name = os.path.basename(file_name)
    audio_path = get_audio_dir() / safe_name
    log_event("audio.get.start", trace_id=trace_id, file=safe_name)

    if not audio_path.exists() or not audio_path.is_file():
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event("audio.get.error", level="ERROR", trace_id=trace_id, file=safe_name, elapsed_ms=elapsed_ms, error="not_found")
        raise HTTPException(status_code=404, detail="Audio file not found")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log_event("audio.get.end", trace_id=trace_id, file=safe_name, elapsed_ms=elapsed_ms)
    return FileResponse(
        str(audio_path),
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
        },
    )


# ---- Visualization Endpoints ----

@router.get("/visualize/graph")
async def visualize_graph():
    """
    Get the agent workflow graph visualization as Mermaid syntax.
    """
    try:
        mermaid_syntax = learning_graph.get_graph().draw_mermaid()
        
        return {
            "success": True,
            "mermaid": mermaid_syntax,
            "agents": [
                {"name": "infer_intent", "description": "Intent Classification"},
                {"name": "simplify", "description": "Age-Adaptive Explanation"},
                {"name": "generate_example", "description": "Example Generation"},
                {"name": "think", "description": "Reflection Question"},
                {"name": "quiz", "description": "Quiz Generation"},
                {"name": "save_answer", "description": "Answer Storage"},
                {"name": "evaluate_answer", "description": "Answer Evaluation"},
                {"name": "answer_feedback", "description": "Answer Feedback"},
                {"name": "safety", "description": "Safety Validation"},
                {"name": "format", "description": "Response Formatting"}
            ],
            "workflows": {
                "teaching": "infer_intent → simplify → generate_example → safety → think → format",
                "quiz_generation": "infer_intent → quiz → safety → format",
                "answer_evaluation": "infer_intent → save_answer → evaluate_answer → answer_feedback → safety → format"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/graph/html", response_class=HTMLResponse)
async def visualize_graph_html():
    """
    Get an interactive HTML visualization of the agent workflow.
    """
    try:
        mermaid_syntax = learning_graph.get_graph().draw_mermaid()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgeXplain - Agent Workflow</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .mermaid {{
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .legend {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .agent-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .agent-item {{
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .agent-name {{
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AgeXplain Multi-Agent System</h1>
        <p class="subtitle">Interactive Agent Workflow Visualization</p>
        
        <div class="mermaid">
{mermaid_syntax}
        </div>
        
        <div class="legend">
            <h3>🔧 Agent Descriptions</h3>
            <div class="agent-list">
                <div class="agent-item">
                    <div class="agent-name">1. infer_intent</div>
                    <div>Classifies user intent</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">2. simplify</div>
                    <div>Age-appropriate explanations</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">3. generate_example</div>
                    <div>Real-world examples</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">4. think</div>
                    <div>Reflective questions</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">5. quiz</div>
                    <div>Quiz generation</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">6. save_answer</div>
                    <div>Answer storage</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">7. evaluate_answer</div>
                    <div>Answer scoring</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">8. answer_feedback</div>
                    <div>Quiz feedback</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">9. safety</div>
                    <div>Content validation</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">10. format</div>
                    <div>Response formatting</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                curve: 'basis',
                useMaxWidth: true
            }}
        }});
    </script>
</body>
</html>"""
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/graph/ascii")
async def visualize_graph_ascii():
    """
    Get a simple ASCII representation of the agent workflow.
    """
    try:
        ascii_graph = learning_graph.get_graph().draw_ascii()
        
        return {
            "success": True,
            "ascii": ascii_graph,
            "note": "This is a simplified ASCII representation of the workflow"
        }
    except Exception as e:
        # Fallback if ASCII generation fails
        return {
            "success": True,
            "ascii": """
Agent Workflow (Simplified):

START
  ↓
[infer_intent] ──┬→ [quiz] → [safety] → [format] → END
                 │
                 ├→ [save_answer] → [evaluate_answer] → [answer_feedback] → [safety] → [format] → END
                 │
                 └→ [simplify] → [generate_example] → [safety] → [think] → [format] → END
            """,
            "note": "Fallback ASCII representation"
        }
