"""
AgeXplain – Streamlit UI
Mirrors the React frontend, calling the Python backend directly (no HTTP API).
"""

import sys
import os

# ── Make sure the streamlit/ directory itself is on sys.path so that all
#    backend packages (graph, agents, configs, …) can be imported directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import asyncio
import hashlib
import uuid
import re

import streamlit as st
from dotenv import load_dotenv

# Load .env from the streamlit directory (same as backend/.env – copy it there)
load_dotenv(os.path.join(_HERE, ".env"))

# ── Backend imports (all unmodified) ──────────────────────────────────────────
from db.database import init_db, save_document, get_document, get_document_by_content_hash
from services.rag_service import get_rag_service
from services.document_service import extract_document_text, summarize_with_fallback
from services.model_provider import RuntimeModelConfig, build_runtime_llm
from configs.config import DIFFICULTY_MAP, use_request_llm, LLM_MODEL, LLM_TEMPERATURE
from graph import learning_graph
from api.routes import TOPICS_DATA, _parse_quiz_response, _sanitize_quiz_question

# ── One-time DB / RAG initialisation ─────────────────────────────────────────
@st.cache_resource
def _init_backend():
    init_db()
    from services.rag_service import reload_indices_from_db
    reload_indices_from_db()
    from services.tts_service import cleanup_audio_files
    cleanup_audio_files(force=True)

_init_backend()

# ── Constants ─────────────────────────────────────────────────────────────────
PROFESSION_OPTIONS = [
    "Business (Sales, Marketing, Finance)",
    "Creative and Media",
    "Data and Research",
    "Education (Teacher, Trainer)",
    "Government and Public Service",
    "Healthcare (Doctor, Nurse)",
    "Management (Team Lead, Manager)",
    "Operations and Skilled Trades",
    "Other",
    "Student",
    "Technology (Software Engineer, IT)",
]

AREA_OF_INTEREST_OPTIONS = sorted([
    "General", "Everyday Life", "School & Exams", "Career & Workplace",
    "Business", "Technology", "Engineering", "Finance", "Healthcare",
    "Law & Policy", "Environment", "Society & Community", "Sports",
    "Soccer", "Cricket", "Basketball", "Music & Arts",
    "Movies & Storytelling", "Gaming", "Research",
])

PROVIDER_OPTIONS = ["local", "claude", "copilot", "gemini", "openai"]

KNOWN_TOPICS = {
    "photosynthesis", "solar system", "water cycle", "gravity", "electricity",
    "fractions", "multiplication", "geometry", "percentages", "algebra basics",
    "ancient egypt", "world war ii", "renaissance", "industrial revolution",
    "american revolution", "how the internet works", "artificial intelligence",
    "computers", "smartphones", "coding basics",
}

# ── Session state initialisation ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "chats": [],
        "active_chat_id": None,
        "last_question": "",
        # settings – these are read by widgets via key=, only set once here
        "age": 10,
        "profession": "Student",
        "expertise_level": "Beginner",
        "area_of_interest": "General",
        "include_examples": False,
        "include_questions": False,
        "model_provider": "local",
        "model_api_key": "",
        "model_validation_result": None,
        "selected_pack": "",
        "quiz_difficulty": "medium",
        # action flags set by sidebar/pill buttons, consumed by render_main
        "_send_text": None,        # text to send in this rerun
        "_upload_done": set(),     # set of file names already processed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Utility helpers ───────────────────────────────────────────────────────────

def _active_chat():
    cid = st.session_state.active_chat_id
    return next((c for c in st.session_state.chats if c["id"] == cid), None)


def _resolve_runtime_llm():
    provider = st.session_state.get("model_provider", "local")
    api_key = (st.session_state.get("model_api_key") or "").strip() or None
    cfg = RuntimeModelConfig(
        provider=provider if provider != "local" else "ollama",
        model=None,
        api_key=api_key,
        base_url=None,
    )
    return build_runtime_llm(cfg, default_model=LLM_MODEL, default_temperature=LLM_TEMPERATURE)


def _new_chat_id():
    return str(uuid.uuid4())


def _build_context(messages):
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"User: {m['text']}")
        elif m["role"] == "assistant" and m.get("sections"):
            s = m["sections"]
            sub = []
            if s.get("Explanation"):
                sub.append(f"Explanation: {s['Explanation']}")
            if s.get("Example"):
                sub.append(f"Example: {s['Example']}")
            if s.get("Question"):
                sub.append(f"Question: {s['Question']}")
            if s.get("Feedback"):
                sub.append(f"Feedback: {s['Feedback']}")
            if sub:
                parts.append("\n".join(sub))
    return "\n\n".join(parts)


def _looks_like_answer(text):
    value = (text or "").strip()
    if not value or value.endswith("?"):
        return False
    lower = value.lower()
    if re.match(r"^(because|i think|it is|it was|they are|they were|yes|no|my answer|answer:)\b", lower):
        return True
    return len(lower.split()) <= 18


def _run_async(coro):
    """Run a coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── Core backend calls ────────────────────────────────────────────────────────

def call_explain(topic, age, context, doc_id, user_answer,
                 profession, expertise_level, area_of_interest,
                 include_examples, include_questions, force_new_topic):
    """Call the learning graph and return sections dict."""
    runtime_llm = _resolve_runtime_llm()

    messages = []
    if context and not force_new_topic:
        for line in context.strip().split("\n\n"):
            if line.startswith("User: "):
                messages.append({"role": "user", "content": line[6:]})
            elif any(line.startswith(p) for p in ("Explanation: ", "Example: ", "Question: ", "Feedback: ")):
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": line})
                else:
                    messages[-1]["content"] += "\n" + line

    current_question = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "Question:" in msg.get("content", ""):
            current_question = msg["content"].rsplit("Question:", 1)[1].strip()
            break
    if not current_question and context and "Question:" in context and not force_new_topic:
        current_question = context.rsplit("Question:", 1)[1].strip()
    if current_question:
        for marker in ["\nUser:", "\nExplanation:", "\nExample:", "\nFeedback:"]:
            if marker in current_question:
                current_question = current_question.split(marker, 1)[0].strip()

    messages.append({"role": "user", "content": topic})

    initial_state = {
        "user_input": topic,
        "messages": messages,
        "learner": {
            "age": age,
            "difficulty": "medium",
            "profession": profession,
            "expertise_level": expertise_level,
            "area_of_interest": area_of_interest,
        },
        "intent": "new_question",
        "doc_id": doc_id,
        "include_examples": include_examples,
        "include_questions": include_questions,
        "force_new_topic": force_new_topic,
        "model_provider": st.session_state.get("model_provider", "local") if st.session_state.get("model_provider") != "local" else "ollama",
        "model_name": LLM_MODEL,
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
        "user_answer": user_answer,
        "score": None,
        "final_output": None,
        "feedback": None,
    }

    async def _run():
        with use_request_llm(runtime_llm):
            return await learning_graph.ainvoke(initial_state)

    result = _run_async(_run())
    final_output = result.get("final_output") or {}

    return {
        "Explanation": final_output.get("explanation") or result.get("safe_text") or result.get("simplified_explanation") or "",
        "Example": final_output.get("example") or result.get("example") or "",
        "Question": final_output.get("think_question") or result.get("thought_question") or "",
        "Feedback": final_output.get("feedback") or result.get("feedback") or "",
    }


def call_upload_document(file_bytes, filename, content_type, age,
                          profession, expertise_level, area_of_interest):
    runtime_llm = _resolve_runtime_llm()
    if not file_bytes:
        raise ValueError("Empty file")
    if len(file_bytes) > 20 * 1024 * 1024:
        raise ValueError("File too large (max 20 MB)")

    extracted_text, parser = extract_document_text(filename, content_type, file_bytes)
    cleaned_text = (extracted_text or "").strip()
    if len(cleaned_text) < 80:
        raise ValueError("Could not extract enough readable text from this file")

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = get_document_by_content_hash(content_hash)

    if existing:
        doc_id = existing["doc_id"]
        rag_service = get_rag_service()
        if doc_id not in rag_service.indices:
            rag_service.index_document(doc_id, existing["text"])
        summary, _ = summarize_with_fallback(
            existing["text"], age, runtime_llm,
            profession=profession or "", expertise_level=expertise_level or "",
            area_of_interest=area_of_interest or "",
        )
        return {"doc_id": doc_id, "filename": filename, "summary": summary}

    summary, _ = summarize_with_fallback(
        cleaned_text, age, runtime_llm,
        profession=profession or "", expertise_level=expertise_level or "",
        area_of_interest=area_of_interest or "",
    )
    doc_id = str(uuid.uuid4())
    save_document(doc_id, filename, cleaned_text, parser, content_hash=content_hash)
    get_rag_service().index_document(doc_id, cleaned_text)
    return {"doc_id": doc_id, "filename": filename, "summary": summary}


def call_generate_quiz(topic, age, num_questions, difficulty, doc_id,
                        profession, expertise_level, area_of_interest):
    from api.routes import (
        _normalize_ocr_quiz_text, _is_doc_grounded_question,
        _build_source_fallback_question, QUIZ_SOURCE_MAX_CHUNKS,
        QUIZ_SOURCE_MAX_CHARS_PER_CHUNK, _clip_chunk_for_quiz,
    )
    runtime_llm = _resolve_runtime_llm()
    difficulty_level = DIFFICULTY_MAP.get(difficulty, "moderate difficulty with some critical thinking")
    learner_profile = (
        "Learner profile:\n"
        f"- Profession: {profession or 'Not provided'}\n"
        f"- Expertise level: {expertise_level or 'Not provided'}\n"
        f"- Area of interest: {area_of_interest or 'Not provided'}"
    )

    source_blocks, verbatim_chunks, source_section, doc_context = [], [], "", ""

    if doc_id:
        doc = get_document(doc_id)
        if not doc:
            raise ValueError("Document not found")
        rag_svc = get_rag_service()
        if doc_id not in rag_svc.indices:
            rag_svc.index_document(doc_id, (doc.get("text") or "").strip())
        all_chunks = rag_svc.indices[doc_id].chunks if doc_id in rag_svc.indices else []
        if all_chunks:
            retrieved = rag_svc.indices[doc_id].retrieve(topic or "key concepts", top_k=QUIZ_SOURCE_MAX_CHUNKS)
            sampled = [c for c, _ in retrieved] if retrieved else all_chunks[::max(1, len(all_chunks) // QUIZ_SOURCE_MAX_CHUNKS)][:QUIZ_SOURCE_MAX_CHUNKS]
            verbatim_chunks = [_clip_chunk_for_quiz(c, QUIZ_SOURCE_MAX_CHARS_PER_CHUNK) for c in sampled if c.strip()]
            source_blocks = verbatim_chunks[:]
        else:
            doc_context = _normalize_ocr_quiz_text((doc.get("text") or "").strip()[:10000])
            source_blocks = [doc_context] if doc_context else []

    if verbatim_chunks:
        passages = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(verbatim_chunks))
        source_section = f"SOURCE PASSAGES (verbatim from the document):\n{passages}"
    elif doc_context:
        source_section = f"DOCUMENT TEXT:\n{doc_context}"

    if source_section:
        prompt = (
            f'You are writing a quiz focused on the topic: "{topic}". Every question must come DIRECTLY from the source passages.\n\n'
            f'{learner_profile}\n\nRULES:\n1. Only write questions relevant to this topic.\n'
            f'2. Correct answers must be facts directly stated in the passages.\n'
            f'3. Difficulty: {difficulty_level}\n\n{source_section}\n\n'
            f'Create exactly {num_questions} questions in this EXACT format:\n\n'
            f'Q1: [Question]\nA) [Option]\nB) [Option]\nC) [Option]\nD) [Option]\n'
            f'Correct: [A, B, C, or D]\nExplanation: [A clear supporting sentence]\n\nContinue for all {num_questions} questions.'
        )
    else:
        prompt = (
            f'Create {num_questions} quiz questions about {topic} for age {age}.\n\n'
            f'{learner_profile}\n\nRules:\n- Difficulty: {difficulty_level}\n\n'
            f'For each question use this EXACT format:\n\n'
            f'Q1: [Question text]\nA) [First option]\nB) [Second option]\nC) [Third option]\nD) [Fourth option]\n'
            f'Correct: [A, B, C, or D]\nExplanation: [Why this answer is correct]\n\nContinue for all {num_questions} questions.'
        )

    raw = runtime_llm.invoke(prompt)
    response = _normalize_ocr_quiz_text((raw or "").strip())
    questions = [_sanitize_quiz_question(q) for q in _parse_quiz_response(response, num_questions)]

    if doc_id and source_blocks:
        questions = [q for q in questions if _is_doc_grounded_question(q, source_blocks)]
        while len(questions) < num_questions:
            questions.append(_build_source_fallback_question(topic, source_blocks, len(questions)))

    if not questions:
        questions = [{
            "question": f"What is an important concept related to {topic}?",
            "options": {"A": "The basics", "B": "Advanced concepts", "C": "Real-world applications", "D": "All of the above"},
            "correct": "D", "explanation": "All aspects are important!",
        }]
    return questions[:num_questions]


def call_validate_model():
    try:
        response = _resolve_runtime_llm().invoke("Reply with exactly OK")
        return True, f"Connected. Model responded: {(response or '').strip()[:80]}"
    except Exception as exc:
        return False, str(exc)


# ── Action handlers ────────────────────────────────────────────────────────────

def handle_send(text):
    text = text.strip()
    if not text:
        return

    is_known = text.lower() in KNOWN_TOPICS
    force_fresh = is_known  # pill/pack clicks set _send_text and pass force_fresh via flag below

    chat_id = st.session_state.active_chat_id
    if not chat_id:
        chat_id = _new_chat_id()
        st.session_state.chats.insert(0, {
            "id": chat_id, "topic": text,
            "messages": [], "doc_id": None, "docs": [],
            "quiz_state": None, "show_quiz_setup": False,
            "quiz_topic": "", "quiz_doc_id": None,
        })
        st.session_state.active_chat_id = chat_id

    chat = next(c for c in st.session_state.chats if c["id"] == chat_id)

    # Check if this is a "force fresh" send (set by topic pill / pack button)
    force_fresh = force_fresh or st.session_state.pop("_force_fresh", False)
    if force_fresh:
        st.session_state.last_question = ""
        chat["topic"] = text

    chat["messages"].append({"role": "user", "text": text})

    context = "" if force_fresh else _build_context(chat["messages"][:-1])
    pending_q = "" if force_fresh else st.session_state.last_question
    user_answer_hint = text if (pending_q and _looks_like_answer(text)) else None

    with st.spinner("Thinking…"):
        try:
            sections = call_explain(
                topic=text, age=st.session_state.age,
                context=context, doc_id=chat.get("doc_id"),
                user_answer=user_answer_hint,
                profession=st.session_state.profession,
                expertise_level=st.session_state.expertise_level,
                area_of_interest=st.session_state.area_of_interest,
                include_examples=st.session_state.include_examples,
                include_questions=st.session_state.include_questions,
                force_new_topic=force_fresh,
            )
            if sections.get("Question"):
                st.session_state.last_question = sections["Question"]
            chat["messages"].append({"role": "assistant", "sections": sections})
        except Exception as exc:
            chat["messages"].append({"role": "assistant", "sections": {}, "error": str(exc)})


def handle_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    content_type = uploaded_file.type or ""

    # Guard: skip if already processed this file in this session
    done_set = st.session_state._upload_done
    file_key = f"{filename}:{hashlib.md5(file_bytes).hexdigest()[:8]}"
    if file_key in done_set:
        return
    done_set.add(file_key)

    chat_id = st.session_state.active_chat_id
    if not chat_id:
        chat_id = _new_chat_id()
        st.session_state.chats.insert(0, {
            "id": chat_id, "topic": f"Document: {filename}",
            "messages": [], "doc_id": None, "docs": [],
            "quiz_state": None, "show_quiz_setup": False,
            "quiz_topic": "", "quiz_doc_id": None,
        })
        st.session_state.active_chat_id = chat_id

    chat = next(c for c in st.session_state.chats if c["id"] == chat_id)

    with st.spinner(f"Uploading and summarizing {filename}…"):
        try:
            result = call_upload_document(
                file_bytes=file_bytes, filename=filename, content_type=content_type,
                age=st.session_state.age, profession=st.session_state.profession,
                expertise_level=st.session_state.expertise_level,
                area_of_interest=st.session_state.area_of_interest,
            )
            doc_id = result["doc_id"]
            chat["doc_id"] = doc_id
            existing_ids = {d["doc_id"] for d in chat.get("docs", [])}
            if doc_id not in existing_ids:
                chat.setdefault("docs", []).append({"doc_id": doc_id, "filename": filename})
            chat["messages"].append({
                "role": "assistant",
                "text": f"Document Summary: {filename}\n\n{result['summary']}",
            })
        except Exception as exc:
            chat["messages"].append({"role": "assistant", "text": f"Upload failed: {exc}"})


def handle_generate_quiz(chat):
    quiz_topic = (chat.get("quiz_topic") or "").strip()
    if not quiz_topic:
        st.error("Please enter a topic for the quiz!")
        return
    with st.spinner("Generating quiz…"):
        try:
            questions = call_generate_quiz(
                topic=quiz_topic, age=st.session_state.age, num_questions=5,
                difficulty=st.session_state.quiz_difficulty,
                doc_id=chat.get("quiz_doc_id") or chat.get("doc_id"),
                profession=st.session_state.profession,
                expertise_level=st.session_state.expertise_level,
                area_of_interest=st.session_state.area_of_interest,
            )
            chat["quiz_state"] = {
                "topic": quiz_topic, "questions": questions,
                "current_index": 0, "score": 0, "answers": [],
                "showing_feedback": False, "current_is_correct": None, "completed": False,
            }
            chat["show_quiz_setup"] = False
        except Exception as exc:
            st.error(f"Failed to generate quiz: {exc}")


# ── CSS ────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap');

/* ── Reset & tokens ── */
:root {
  --bg-main:   #f4f7f8;
  --bg-panel:  #fdfefe;
  --bg-strong: #edf3f3;
  --bg-chat:   #ffffff;
  --brand:     #0f766e;
  --brand-h:   #0b5f59;
  --text-s:    #0f1722;
  --text-b:    #2a3b49;
  --text-m:    #6b7d8c;
  --line-s:    #d9e2e6;
  --line-st:   #b9c8cf;
  --shadow-s:  0 8px 22px rgba(22,40,54,.08);
  --shadow-p:  0 14px 30px rgba(16,40,56,.12);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background:
    radial-gradient(circle at 10% 12%, #d5ece8 0%, transparent 36%),
    radial-gradient(circle at 88% 15%, #cfe9e4 0%, transparent 34%),
    linear-gradient(155deg, #f8fbfb 0%, #eef4f4 100%) !important;
  color: var(--text-b) !important;
}

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }

/* ── Left sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-strong) 100%) !important;
  border-right: 1px solid var(--line-s) !important;
  box-shadow: var(--shadow-s) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* sidebar headings */
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-size: .75rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  color: var(--text-m) !important;
  margin: 14px 0 8px !important;
  font-weight: 700 !important;
}

/* ── Sidebar "new chat" button ── */
section[data-testid="stSidebar"] .stButton:first-of-type > button {
  background: linear-gradient(135deg, var(--brand) 0%, #109e93 100%) !important;
  color: #fff !important;
  border: 1px solid #0d5f59 !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  font-size: .92rem !important;
  transition: transform .14s, box-shadow .14s !important;
}
section[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 18px rgba(10,113,104,.24) !important;
}

/* ── Chat item buttons in sidebar ── */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  color: var(--text-b) !important;
  font-size: .86rem !important;
  text-align: left !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  transition: background .14s, border-color .14s !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button:hover {
  background: #f2f7f8 !important;
  border-color: #d3e0e3 !important;
}

/* ── Sidebar selects / inputs ── */
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stTextInput input {
  border-radius: 10px !important;
  border: 1px solid var(--line-st) !important;
  color: var(--text-b) !important;
  font-size: .88rem !important;
  background: #fff !important;
}
section[data-testid="stSidebar"] label { color: var(--text-s) !important; font-weight: 600 !important; font-size: .84rem !important; }

/* ── Main area panels ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
  background: var(--bg-chat);
  border-radius: 16px;
  border: 1px solid var(--line-s);
  box-shadow: var(--shadow-p);
}

/* ── Message bubbles ── */
.msg-user-wrap {
  display: flex; justify-content: flex-end; margin: 10px 0 10px 20%;
}
.msg-user-inner {
  background: #dff3ee;
  border: 1px solid #bfe2d8;
  border-radius: 8px;
  padding: 6px 12px;
  color: #133946;
  font-size: .93rem;
  line-height: 1.5;
  word-break: break-word;
  position: relative;
}
.msg-who-user {
  font-size: 11px; font-weight: 600; color: #6b7280;
  text-align: right; margin-bottom: 3px;
}

.msg-assistant-wrap {
  margin: 10px 20% 10px 0;
}
.msg-assistant-inner {
  background: #f3f8f8;
  border: 1px solid #d8e6e4;
  border-radius: 8px;
  padding: 8px 10px;
  color: #1f3746;
  font-size: .93rem;
  word-break: break-word;
}
.msg-who-asst {
  font-size: 12px; font-weight: 600; color: #5d7280; margin-bottom: 4px;
}

/* ── Section blocks inside assistant message ── */
.sec-block {
  white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;
  line-height: 1.55; margin-bottom: 8px; padding: 8px;
  border-radius: 6px; background: #fdfeff; font-size: .9rem;
}
.sec-explanation { border-left: 3px solid #10b981; }
.sec-example     { border-left: 3px solid #fbbf24; }
.sec-question    { border-left: 3px solid #2563eb; }
.sec-feedback    { border-left: 3px solid #8b5cf6; background: #f3f4f6; }

.sec-title {
  font-size: 11px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; margin-bottom: 4px;
}
.sec-title-explanation { color: #065f46; }
.sec-title-example     { color: #92400e; }
.sec-title-question    { color: #1e40af; }
.sec-title-feedback    { color: #6d28d9; }

/* ── Error bubble ── */
.msg-error {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 12px 16px; border-radius: 8px;
  background: rgba(239,68,68,.08); border: 1px solid rgba(239,68,68,.3);
  color: #dc2626; font-size: .92rem; margin: 8px 0;
}

/* ── Topic pills ── */
.topic-pill-btn {
  background: #fff !important;
  border: 1.5px solid #38bdf8 !important;
  color: #0284c7 !important;
  border-radius: 20px !important;
  padding: 5px 14px !important;
  font-size: 13px !important;
  cursor: pointer !important;
  transition: background .15s, color .15s, transform .1s !important;
  font-family: inherit !important;
}
.topic-pill-btn:hover {
  background: #0ea5e9 !important; color: #fff !important;
  border-color: #0ea5e9 !important; transform: translateY(-1px) !important;
}
.summary-topic-box {
  background: #f0f9ff; border: 1px solid #bae6fd;
  border-radius: 10px; padding: 14px 16px; margin-top: 8px;
}
.summary-topic-box strong { display: block; font-size: 13px; color: #0369a1; margin-bottom: 10px; }
.summary-topic-prompt { font-size: 12px; color: #64748b; font-style: italic; margin: 8px 0 0; }

/* ── Right panel (settings column) ── */
.right-panel {
  background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-strong) 100%);
  border: 1px solid var(--line-s);
  border-radius: 16px;
  box-shadow: var(--shadow-s);
  padding: 16px;
  height: 100%;
  overflow-y: auto;
}
.right-panel h4 {
  font-size: .75rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--text-m); margin: 14px 0 8px; font-weight: 700;
}

/* ── Primary action buttons ── */
button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--brand) 0%, #109e93 100%) !important;
  border: 1px solid #0a5b56 !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-weight: 700 !important;
  transition: transform .14s, box-shadow .14s !important;
}
button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 16px rgba(16,125,117,.3) !important;
}

/* ── Quiz progress bar ── */
.quiz-bar-wrap {
  width: 100%; height: 8px; border-radius: 999px;
  background: #e8f1f6; overflow: hidden; margin: 8px 0 16px;
}
.quiz-bar-fill {
  height: 100%; border-radius: inherit;
  background: linear-gradient(90deg, #0f766e 0%, #18a59a 100%);
  transition: width 220ms ease;
}

/* ── Quiz option buttons ── */
.quiz-opt {
  display: flex; align-items: flex-start; gap: 10px;
  width: 100%; text-align: left; padding: 12px 13px;
  border-radius: 10px; border: 1px solid #cfe0e7;
  background: #fbfdff; color: #1f3746;
  font-size: .92rem; line-height: 1.45;
  cursor: pointer; transition: all .2s; margin-bottom: 10px;
}
.quiz-opt:hover { border-color: #4e95b7; background: #f2fbff; transform: translateX(2px); }
.quiz-opt-key {
  width: 24px; height: 24px; border-radius: 999px;
  flex: 0 0 24px; display: inline-flex; align-items: center;
  justify-content: center; background: #eaf4f8; color: #2f4c5c;
  font-weight: 700; font-size: .78rem; margin-top: 1px;
}
.quiz-opt-correct { background: #e9faf3 !important; border-color: #10b981 !important; }
.quiz-opt-incorrect { background: #fff0f0 !important; border-color: #ef4444 !important; }
.quiz-opt-correct .quiz-opt-key { background: #d1fae5; color: #065f46; }
.quiz-opt-incorrect .quiz-opt-key { background: #fee2e2; color: #991b1b; }

.quiz-feedback-correct { background:#f0fbf6; border:1px solid #bde9d7; color:#135445;
  border-radius:10px; padding:12px 13px; margin-top:12px; font-size:.9rem; line-height:1.45; }
.quiz-feedback-incorrect { background:#fff5f5; border:1px solid #f6c8c8; color:#7f1d1d;
  border-radius:10px; padding:12px 13px; margin-top:12px; font-size:.9rem; line-height:1.45; }

/* ── Quiz results ── */
.quiz-score-card {
  padding: 22px 24px;
  background: linear-gradient(135deg, #0f766e 0%, #109e93 100%);
  color: #fff; border-radius: 12px;
  font-size: 1.75rem; font-weight: 700;
  text-align: center; border: 1px solid #0a5b56;
  margin: 14px 0 20px;
}
.quiz-score-pct { font-size: .52em; margin-top: 8px; font-weight: 700; }
.review-card {
  background: #fff; border: 1px solid #d8e6e4;
  border-radius: 10px; padding: 14px 16px;
  margin-bottom: 12px; line-height: 1.5; color: #1f3746;
}
.review-card-correct  { border-left: 4px solid #10b981; }
.review-card-incorrect { border-left: 4px solid #ef4444; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
  background: #fcfefe !important;
  border: 1px solid var(--line-st) !important;
  border-radius: 14px !important;
  font-size: .95rem !important;
  color: var(--text-b) !important;
  font-family: inherit !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: #2b8b83 !important;
  box-shadow: 0 0 0 4px rgba(43,139,131,.15) !important;
}
[data-testid="stChatInputContainer"] {
  background: linear-gradient(180deg, #fdfefe 0%, #f3f8f8 100%) !important;
  border-top: 1px solid var(--line-s) !important;
}

/* ── Model validation ── */
.val-ok  { color: #065f46; font-size: .84rem; margin-top: 4px; background: #d1fae5; padding: 6px 10px; border-radius: 8px; }
.val-err { color: #991b1b; font-size: .84rem; margin-top: 4px; background: #fee2e2; padding: 6px 10px; border-radius: 8px; }

/* ── Dividers ── */
hr { border-color: var(--line-s) !important; margin: 12px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #c4d4d8; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _parse_summary_topics(text):
    HEADER = "Topics You Can Ask About:"
    idx = text.find(HEADER)
    if idx == -1:
        return text, [], ""
    before = text[:idx].rstrip()
    rest = text[idx + len(HEADER):]
    topics, after_lines = [], []
    in_topics = True
    for ln in rest.split("\n"):
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped == "Would you like to know more about any of these topics?":
            in_topics = False
            continue
        if in_topics and stripped.startswith("- "):
            topics.append(stripped[2:].strip())
        else:
            in_topics = False
            after_lines.append(stripped)
    return before, topics, "\n".join(after_lines)


def _esc(text: str) -> str:
    """Minimal HTML escaping for raw text inside HTML blocks."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_message(msg, msg_idx):
    """Render one chat message with React-matching bubble styles."""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user-wrap">'
            f'<div class="msg-user-inner">'
            f'<div class="msg-who-user">You</div>'
            f'{_esc(msg["text"])}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    # assistant — error
    if msg.get("error"):
        st.markdown(
            f'<div class="msg-error">⚠ {_esc(msg["error"])}</div>',
            unsafe_allow_html=True,
        )
        return

    sections = msg.get("sections")
    raw_text = msg.get("text", "")

    # assistant — structured sections
    if sections:
        blocks = ['<div class="msg-who-asst">Assistant</div>']
        if sections.get("Feedback"):
            blocks.append(
                f'<div class="sec-block sec-feedback">'
                f'<div class="sec-title sec-title-feedback">Feedback</div>'
                f'{_esc(sections["Feedback"])}'
                f'</div>'
            )
        else:
            if sections.get("Explanation"):
                blocks.append(
                    f'<div class="sec-block sec-explanation">'
                    f'<div class="sec-title sec-title-explanation">Explanation</div>'
                    f'{_esc(sections["Explanation"])}'
                    f'</div>'
                )
            if sections.get("Example"):
                blocks.append(
                    f'<div class="sec-block sec-example">'
                    f'<div class="sec-title sec-title-example">Example</div>'
                    f'{_esc(sections["Example"])}'
                    f'</div>'
                )
            if sections.get("Question"):
                blocks.append(
                    f'<div class="sec-block sec-question">'
                    f'<div class="sec-title sec-title-question">Question to Think About</div>'
                    f'{_esc(sections["Question"])}'
                    f'</div>'
                )
        if len(blocks) > 1:
            inner = "".join(blocks)
            st.markdown(
                f'<div class="msg-assistant-wrap"><div class="msg-assistant-inner">{inner}</div></div>',
                unsafe_allow_html=True,
            )
        return

    # assistant — raw text, possibly with topic pills
    if raw_text:
        if "Topics You Can Ask About:" in raw_text:
            before, topics, after = _parse_summary_topics(raw_text)
            st.markdown(
                f'<div class="msg-assistant-wrap"><div class="msg-assistant-inner">'
                f'<div class="msg-who-asst">Assistant</div>'
                f'<pre style="white-space:pre-wrap;font-family:inherit;margin:0;color:#1f3746">{_esc(before)}</pre>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if topics:
                st.markdown(
                    '<div class="summary-topic-box"><strong>Topics You Can Ask About:</strong>',
                    unsafe_allow_html=True,
                )
                pill_cols = st.columns(min(len(topics), 4))
                for i, t in enumerate(topics):
                    with pill_cols[i % len(pill_cols)]:
                        if st.button(t, key=f"pill_{msg_idx}_{i}", use_container_width=True):
                            st.session_state._force_fresh = True
                            handle_send(t)
                            st.rerun()
                st.markdown(
                    '<p class="summary-topic-prompt">Click a topic or type your own question below.</p></div>',
                    unsafe_allow_html=True,
                )
            if after:
                st.markdown(
                    f'<div class="msg-assistant-wrap"><div class="msg-assistant-inner">'
                    f'<div class="sec-block sec-explanation">{_esc(after)}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="msg-assistant-wrap"><div class="msg-assistant-inner">'
                f'<div class="msg-who-asst">Assistant</div>'
                f'{_esc(raw_text)}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def render_quiz_setup(chat):
    st.subheader("Quiz Setup")
    docs = chat.get("docs") or (
        [{"doc_id": chat["doc_id"], "filename": "Uploaded document"}] if chat.get("doc_id") else []
    )
    is_doc_chat = bool(docs)

    if is_doc_chat:
        st.caption("Questions will be generated from the uploaded document.")
        if len(docs) > 1:
            doc_names = {d["doc_id"]: d["filename"] for d in docs}
            chat["quiz_doc_id"] = st.selectbox(
                "Select document", list(doc_names.keys()),
                format_func=lambda k: doc_names[k],
                key="quiz_doc_sel",
            )
        st.text_input("Focus area (optional)", key="_qt_input",
                      value=chat.get("quiz_topic", ""),
                      on_change=lambda: chat.update({"quiz_topic": st.session_state._qt_input}))
    else:
        st.text_input("Quiz topic", key="_qt_input",
                      value=chat.get("quiz_topic", ""),
                      on_change=lambda: chat.update({"quiz_topic": st.session_state._qt_input}))
        # Also sync immediately for button click
        chat["quiz_topic"] = st.session_state.get("_qt_input", chat.get("quiz_topic", ""))

    st.selectbox("Difficulty", ["easy", "medium", "hard"],
                 index=["easy", "medium", "hard"].index(st.session_state.quiz_difficulty),
                 key="quiz_diff_sel",
                 on_change=lambda: st.session_state.update({"quiz_difficulty": st.session_state.quiz_diff_sel}))

    can_gen = is_doc_chat or bool((chat.get("quiz_topic") or "").strip())
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Quiz", type="primary", disabled=not can_gen, key="btn_gen_quiz"):
            chat["quiz_topic"] = st.session_state.get("_qt_input", chat.get("quiz_topic", ""))
            st.session_state.quiz_difficulty = st.session_state.get("quiz_diff_sel", st.session_state.quiz_difficulty)
            handle_generate_quiz(chat)
            st.rerun()
    with col2:
        if st.button("Cancel", key="btn_cancel_quiz"):
            chat["show_quiz_setup"] = False
            st.rerun()


def render_quiz(chat):
    qs = chat["quiz_state"]

    # ── Results screen ──
    if qs.get("completed"):
        total = len(qs["questions"])
        pct = round(qs["score"] / total * 100) if total else 0
        st.markdown(
            f'<div class="quiz-score-card">{qs["score"]}/{total}'
            f'<div class="quiz-score-pct">{pct}% correct</div></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Review")
        for i, ans in enumerate(qs["answers"]):
            card_cls = "review-card review-card-correct" if ans["isCorrect"] else "review-card review-card-incorrect"
            icon = "✓" if ans["isCorrect"] else "✗"
            exp_text = f"<br><span style='color:#315060;font-size:.88rem'>{_esc(ans.get('explanation',''))}</span>" if ans.get("explanation") else ""
            ca = f"<br><strong style='color:#991b1b'>Correct: {_esc(ans['correctAnswer'])}</strong>" if not ans["isCorrect"] else ""
            st.markdown(
                f'<div class="{card_cls}"><strong>Q{i+1}: {_esc(ans["question"])}</strong><br>'
                f'Your answer: <strong>{_esc(ans["userAnswer"])}</strong> {icon}{ca}{exp_text}</div>',
                unsafe_allow_html=True,
            )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Try Again", type="primary", key="quiz_retry"):
                chat["show_quiz_setup"] = True
                chat["quiz_state"] = None
                st.rerun()
        with c2:
            if st.button("Back to Explain", key="quiz_back"):
                chat["quiz_state"] = None
                chat["show_quiz_setup"] = False
                st.rerun()
        return

    # ── Active question ──
    q = qs["questions"][qs["current_index"]]
    total = len(qs["questions"])
    pct_fill = round((qs["current_index"]) / total * 100)
    answered = len(qs["answers"])

    # Header card
    st.markdown(
        f'<div style="background:#fff;border:1px solid #d7e7e5;border-radius:12px;'
        f'padding:16px 18px;box-shadow:0 3px 14px rgba(20,58,80,.06);margin-bottom:14px">'
        f'<h2 style="margin:0 0 10px;color:#1f3746;font-size:1.1rem">Quiz: {_esc(qs["topic"])}</h2>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'color:#5a6e7b;font-size:.84rem;font-weight:600;margin-bottom:8px">'
        f'<span>Question {qs["current_index"] + 1} of {total}</span>'
        f'<span style="border:1px solid #b7ddd8;border-radius:999px;padding:3px 9px;'
        f'background:#f2fbf8;color:#21665d">Score: {qs["score"]}/{answered if answered else "–"}</span>'
        f'</div>'
        f'<div class="quiz-bar-wrap"><div class="quiz-bar-fill" style="width:{pct_fill}%"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Question card
    st.markdown(
        f'<div style="background:#fff;border:1px solid #d7e7e5;border-radius:12px;'
        f'padding:18px;box-shadow:0 3px 14px rgba(20,58,80,.06)">'
        f'<p style="font-size:.76rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;'
        f'color:#5f7482;margin:0 0 6px">Question</p>'
        f'<h3 style="font-size:1.1rem;font-weight:700;line-height:1.45;margin:0 0 18px;color:#1f3746">'
        f'{_esc(q["question"])}'
        f'</h3>',
        unsafe_allow_html=True,
    )

    if not qs["showing_feedback"]:
        for key, val in q["options"].items():
            if st.button(
                f"{key})  {val}",
                key=f"opt_{qs['current_index']}_{key}",
                use_container_width=True,
            ):
                is_correct = key == q["correct"]
                qs["answers"].append({
                    "question": q["question"],
                    "userAnswer": key,
                    "correctAnswer": q["correct"],
                    "isCorrect": is_correct,
                    "explanation": q.get("explanation", ""),
                })
                qs["score"] += 1 if is_correct else 0
                qs["showing_feedback"] = True
                qs["current_is_correct"] = is_correct
                st.rerun()
    else:
        last_answer = qs["answers"][-1]["userAnswer"] if qs["answers"] else None
        for key, val in q["options"].items():
            is_right = key == q["correct"]
            was_picked = key == last_answer
            if is_right:
                extra = "quiz-opt-correct"
            elif was_picked:
                extra = "quiz-opt-incorrect"
            else:
                extra = ""
            key_cls = ("quiz-opt-key " + extra).strip()
            st.markdown(
                f'<div class="quiz-opt {extra}">'
                f'<span class="{key_cls}">{key}</span>'
                f'<span class="quiz-opt-text">{_esc(val)}</span></div>',
                unsafe_allow_html=True,
            )

        if qs["current_is_correct"]:
            st.markdown(
                '<div class="quiz-feedback-correct">✓ Correct!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="quiz-feedback-incorrect">✗ Not quite! '
                f'Correct answer: <strong>{_esc(q["correct"])}</strong></div>',
                unsafe_allow_html=True,
            )
        if q.get("explanation"):
            st.caption(q["explanation"])

        btn_label = "Next Question →" if qs["current_index"] < total - 1 else "View Results"
        if st.button(btn_label, type="primary", key="quiz_next"):
            if qs["current_index"] < total - 1:
                qs["current_index"] += 1
                qs["showing_feedback"] = False
                qs["current_is_correct"] = None
            else:
                qs["completed"] = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """Left sidebar — chat history & file upload only."""
    with st.sidebar:
        st.markdown("## 🧠 AgeXplain")

        # ── New chat ──
        if st.button("＋ New Chat", use_container_width=True, key="btn_new_chat"):
            st.session_state.active_chat_id = None
            st.session_state.last_question = ""
            st.rerun()

        # ── Chat history ──
        st.markdown("### Chats")
        if st.session_state.chats:
            if st.button("Clear All", key="btn_clear_all"):
                st.session_state.chats = []
                st.session_state.active_chat_id = None
                st.rerun()

        if not st.session_state.chats:
            st.markdown('<p style="font-size:13px;color:#7a8d99;margin:4px 0">No chats yet.</p>', unsafe_allow_html=True)

        for c in st.session_state.chats:
            is_active = c["id"] == st.session_state.active_chat_id
            label = ("▶ " if is_active else "") + c["topic"][:36]
            col_chat, col_del = st.columns([6, 1])
            with col_chat:
                if st.button(label, key=f"chat_{c['id']}", use_container_width=True):
                    st.session_state.active_chat_id = c["id"]
                    st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{c['id']}"):
                    st.session_state.chats = [x for x in st.session_state.chats if x["id"] != c["id"]]
                    if st.session_state.active_chat_id == c["id"]:
                        st.session_state.active_chat_id = None
                    st.rerun()

        # ── File upload ──
        st.divider()
        st.markdown("### Upload Document")
        uploaded = st.file_uploader(
            "PDF / DOCX / TXT / MD",
            type=["pdf", "docx", "txt", "md", "csv", "html"],
            key="sidebar_uploader",
            label_visibility="visible",
        )
        if uploaded:
            handle_upload(uploaded)
            st.rerun()


def render_right_panel():
    """Right settings panel — rendered as a Streamlit column."""
    st.markdown(
        '<div class="right-panel">',
        unsafe_allow_html=True,
    )

    # ── Model ──
    st.markdown('<h4 style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:0 0 6px;font-weight:700">Model</h4>', unsafe_allow_html=True)
    st.selectbox("Provider", PROVIDER_OPTIONS,
                 index=PROVIDER_OPTIONS.index(st.session_state.model_provider),
                 key="model_provider",
                 label_visibility="collapsed")
    if st.session_state.model_provider != "local":
        st.text_input("API Key", type="password", key="model_api_key",
                      placeholder="API Key…", label_visibility="collapsed")

    if st.button("Test Connection", key="btn_validate", use_container_width=True):
        ok, msg = call_validate_model()
        st.session_state.model_validation_result = {"ok": ok, "message": msg}
    if st.session_state.model_validation_result:
        r = st.session_state.model_validation_result
        css = "val-ok" if r["ok"] else "val-err"
        st.markdown(f'<p class="{css}">{_esc(r["message"])}</p>', unsafe_allow_html=True)

    st.divider()

    # ── Difficulty / Age ──
    st.markdown('<h4 style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:0 0 4px;font-weight:700">Explain Age</h4>', unsafe_allow_html=True)
    st.slider("Age", 5, 35, key="age", label_visibility="collapsed")

    st.divider()

    # ── Profile ──
    st.markdown('<h4 style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:0 0 4px;font-weight:700">Profile</h4>', unsafe_allow_html=True)
    st.selectbox("Profession", PROFESSION_OPTIONS,
                 index=PROFESSION_OPTIONS.index(st.session_state.profession),
                 key="profession", label_visibility="visible")
    st.selectbox("Expertise Level", ["Beginner", "Intermediate", "Advanced"],
                 index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.expertise_level),
                 key="expertise_level", label_visibility="visible")
    st.selectbox("Example Context", AREA_OF_INTEREST_OPTIONS,
                 index=AREA_OF_INTEREST_OPTIONS.index(
                     st.session_state.area_of_interest
                     if st.session_state.area_of_interest in AREA_OF_INTEREST_OPTIONS else "General"
                 ),
                 key="area_of_interest", label_visibility="visible")

    st.divider()

    # ── Output ──
    st.markdown('<h4 style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:0 0 4px;font-weight:700">Output</h4>', unsafe_allow_html=True)
    st.checkbox("Include examples", key="include_examples")
    st.checkbox("Include follow-up questions", key="include_questions")

    st.divider()

    # ── Topic Packs ──
    st.markdown('<h4 style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:0 0 4px;font-weight:700">Topic Packs</h4>', unsafe_allow_html=True)
    pack_names = [""] + sorted(TOPICS_DATA.keys())
    st.selectbox("Pack", pack_names,
                 index=pack_names.index(st.session_state.selected_pack)
                 if st.session_state.selected_pack in pack_names else 0,
                 key="selected_pack",
                 label_visibility="collapsed")
    if st.session_state.selected_pack:
        for t in TOPICS_DATA[st.session_state.selected_pack]:
            if st.button(t, key=f"pack_{t}", use_container_width=True):
                st.session_state._force_fresh = True
                handle_send(t)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

def render_main():
    chat = _active_chat()

    # ── Full-width overlays (quiz setup / active quiz) ──
    if chat and chat.get("show_quiz_setup"):
        col_chat, col_right = st.columns([3, 1], gap="medium")
        with col_chat:
            render_quiz_setup(chat)
        with col_right:
            render_right_panel()
        return

    if chat and chat.get("quiz_state"):
        col_chat, col_right = st.columns([3, 1], gap="medium")
        with col_chat:
            render_quiz(chat)
        with col_right:
            render_right_panel()
        return

    # ── Normal layout: chat area | right settings panel ──
    col_chat, col_right = st.columns([3, 1], gap="medium")

    with col_chat:
        # Message list
        if not chat:
            st.markdown(
                '<p style="color:#7a8d99;text-align:center;margin-top:80px;font-size:.95rem">'
                'Start a conversation — type a question or pick a topic from the right panel.</p>',
                unsafe_allow_html=True,
            )
        else:
            for i, msg in enumerate(chat["messages"]):
                render_message(msg, i)

        # ── Quiz launch button (visible mid-chat) ──
        if chat and chat.get("messages") and not chat.get("quiz_state") and not chat.get("show_quiz_setup"):
            st.markdown('<div style="display:flex;justify-content:flex-end;margin-top:8px">', unsafe_allow_html=True)
            if st.button("🎯 Start Quiz", key="btn_launch_quiz"):
                qt = chat.get("topic", "")
                if chat.get("doc_id"):
                    qt = re.sub(r"^Document:\s*", "", qt, flags=re.IGNORECASE)
                    qt = re.sub(r"\.(pdf|docx|txt|md|csv|html?)$", "", qt, flags=re.IGNORECASE).strip()
                    if not qt:
                        qt = "the uploaded document"
                chat["show_quiz_setup"] = True
                chat["quiz_topic"] = qt
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Chat input ──
        prompt = st.chat_input("Type your message…", key="chat_input_box")
        if prompt:
            handle_send(prompt)
            st.rerun()

    with col_right:
        render_right_panel()


# ── Main entry ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgeXplain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_inject_css()
render_sidebar()
render_main()



