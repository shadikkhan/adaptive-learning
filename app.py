"""AgeXplain – Streamlit UI"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hashlib, uuid, re, json, asyncio, threading, io
from urllib.parse import quote
import streamlit as st
from dotenv import load_dotenv
try:
    from streamlit_pills import pills
except Exception:
    pills = None
load_dotenv(os.path.join(_HERE, ".env"))

TOPICS_DATA = {
    "Science":    ["photosynthesis","solar system","water cycle","gravity","electricity"],
    "Math":       ["fractions","multiplication","geometry","percentages","algebra basics"],
    "History":    ["ancient egypt","world war ii","renaissance","industrial revolution","american revolution"],
    "Technology": ["how the internet works","artificial intelligence","computers","smartphones","coding basics"],
}
PROFESSION_OPTIONS = [
    "Business (Sales, Marketing, Finance)","Creative and Media","Data and Research",
    "Education (Teacher, Trainer)","Government and Public Service","Healthcare (Doctor, Nurse)",
    "Management (Team Lead, Manager)","Operations and Skilled Trades","Other","Student",
    "Technology (Software Engineer, IT)",
]
AREA_OPTIONS = sorted([
    "General","Everyday Life","School & Exams","Career & Workplace","Business","Technology",
    "Engineering","Finance","Healthcare","Law & Policy","Environment","Society & Community",
    "Sports","Soccer","Cricket","Basketball","Music & Arts","Movies & Storytelling","Gaming","Research",
])
ENABLE_LOCAL = os.getenv("ENABLE_LOCAL_PROVIDER","true").strip().lower() in {"1","true","yes","on"}
PROVIDERS = ["claude","copilot","gemini","openai"]
if ENABLE_LOCAL: PROVIDERS.insert(3,"local")
PLABELS = {"claude":"Claude","copilot":"Copilot","gemini":"Gemini","local":"Local (Ollama)","openai":"OpenAI"}

def _init():
    D = dict(chats=[],active_chat_id=None,last_question="",age=10,
             profession="Student",expertise_level="Beginner",area_of_interest="General",
             include_examples=False,include_questions=False,model_provider="claude",
             _model_provider_selected="claude",
             model_api_key="",model_validation_result=None,selected_pack="",
             _api_keys_by_provider={},
             quiz_difficulty="medium",_ic=0,_upload_done=set(),
             _last_selected_pack="",_last_pack_topic="",_pending_pack_topic="",
             _pending_prompt_text="",_db_ready=False,
             _last_mic_hash="",_last_mic_text="",_last_mic_prompt_text="",
             _is_generating=False,_pending_send_text="",_pending_send_force_fresh=False,
             _pending_include_examples=False,_pending_include_questions=False)
    for k,v in D.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

def _chat():
    cid=st.session_state.active_chat_id
    return next((c for c in st.session_state.chats if c["id"]==cid),None)

def _active_provider() -> str:
    # Prefer stable session state over widget state to avoid rerun-order drift.
    p = (st.session_state.get("model_provider") or st.session_state.get("_model_provider_selected") or "").strip().lower()
    if p in PROVIDERS:
        return p
    sel = (st.session_state.get("model_provider_widget") or "").strip().lower()
    if sel in PROVIDERS:
        return sel
    return "local"

def _active_api_key(provider: str | None = None) -> str:
    p = (provider or _active_provider() or "").strip().lower()
    byp = st.session_state.get("_api_keys_by_provider") or {}
    mapped = (byp.get(p) or "").strip()
    if mapped:
        return mapped
    ui_key = (st.session_state.get("model_api_key") or "").strip()
    if ui_key:
        return ui_key
    env_map = {
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "copilot": ["GITHUB_TOKEN", "COPILOT_API_KEY"],
    }
    for name in env_map.get(p, []):
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return ""

def _new_chat(topic="New Chat"):
    cid=str(uuid.uuid4())
    st.session_state.chats.insert(0,dict(id=cid,topic=topic,messages=[],
        doc_id=None,docs=[],quiz_state=None,show_quiz_setup=False,quiz_topic="",quiz_doc_id=None))
    st.session_state.active_chat_id=cid
    return cid

def _err():
    p=_active_provider()
    if not p: return "Select a model provider in the right panel."
    if p!="local" and not _active_api_key(p):
        return f"Enter API key for '{p}' in the right panel."
    return ""

def _esc(t): return (t or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
def _esc_pre(t): return (t or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _ensure_db():
    if not st.session_state.get("_db_ready"):
        from db.database import init_db
        from services.rag_service import reload_indices_from_db
        init_db()
        reload_indices_from_db()
        st.session_state._db_ready = True

def _build_llm():
    from services.model_provider import RuntimeModelConfig, build_runtime_llm
    from configs.config import LLM_MODEL, LLM_TEMPERATURE
    p = _active_provider()
    k = _active_api_key(p)
    cfg = RuntimeModelConfig(provider=p, api_key=k or None)
    return build_runtime_llm(cfg, default_model=LLM_MODEL, default_temperature=LLM_TEMPERATURE)

def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box = {"result": None, "error": None}

    def _runner():
        try:
            box["result"] = asyncio.run(coro)
        except Exception as exc:
            box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if box["error"] is not None:
        raise box["error"]
    return box["result"]

def _llm_request_config():
    from api.routes import ModelConfigRequest
    provider = _active_provider()
    api_key = _active_api_key(provider) or None
    return ModelConfigRequest(provider=provider, api_key=api_key)

def _chat_messages_for_graph(chat):
    out = []
    for m in (chat or {}).get("messages", []):
        role = m.get("role")
        if role == "user":
            txt = (m.get("text") or "").strip()
            if txt:
                out.append({"role": "user", "content": txt})
            continue
        if role != "assistant":
            continue
        secs = m.get("sections") or {}
        if secs:
            parts = []
            if secs.get("Explanation"):
                parts.append(f"Explanation: {secs['Explanation']}")
            if secs.get("Example"):
                parts.append(f"Example: {secs['Example']}")
            if secs.get("Question"):
                parts.append(f"Question: {secs['Question']}")
            if secs.get("Feedback"):
                parts.append(f"Feedback: {secs['Feedback']}")
            txt = "\n".join(parts).strip()
        else:
            txt = (m.get("text") or m.get("error") or "").strip()
        if txt:
            out.append({"role": "assistant", "content": txt})
    return out

def _tts_payload_from_sections(sections):
    if not sections:
        return ""
    parts = []
    if sections.get("Explanation"):
        parts.append(sections["Explanation"])
    if sections.get("Example"):
        parts.append(f"Example: {sections['Example']}")
    if sections.get("Question"):
        parts.append(f"Question: {sections['Question']}")
    if sections.get("Feedback"):
        parts.append(f"Feedback: {sections['Feedback']}")
    return "\n\n".join([p for p in parts if p]).strip()

def _synthesize_audio(text):
    payload = (text or "").strip()
    if not payload:
        return None
    try:
        from services.tts_service import synthesize_tts_mp3
        audio_path = synthesize_tts_mp3(payload)
        return str(audio_path) if audio_path else None
    except Exception:
        return None

def _transcribe_audio_input(audio_file):
    if audio_file is None:
        return None
    try:
        import speech_recognition as sr
    except Exception:
        return None

    try:
        audio_bytes = audio_file.getvalue()
        if not audio_bytes:
            return None
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash == st.session_state.get("_last_mic_hash"):
            return st.session_state.get("_last_mic_text") or None

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        transcript = (recognizer.recognize_google(audio_data) or "").strip()
        st.session_state._last_mic_hash = audio_hash
        st.session_state._last_mic_text = transcript
        return transcript or None
    except Exception:
        return None

def _chat_context_for_stream(chat):
    lines = []
    for m in (chat or {}).get("messages", []):
        role = m.get("role")
        if role == "user":
            txt = (m.get("text") or "").strip()
            if txt:
                lines.append(f"User: {txt}")
            continue
        if role != "assistant":
            continue
        secs = m.get("sections") or {}
        if secs.get("Explanation"):
            lines.append(f"Explanation: {secs['Explanation']}")
        if secs.get("Example"):
            lines.append(f"Example: {secs['Example']}")
        if secs.get("Question"):
            lines.append(f"Question: {secs['Question']}")
        if secs.get("Feedback"):
            lines.append(f"Feedback: {secs['Feedback']}")
    return "\n\n".join(lines)

def _resolve_audio_event_path(audio_url):
    url = (audio_url or "").strip()
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if os.path.exists(url):
        return url
    if url.startswith("/audio/"):
        from services.tts_service import get_audio_dir
        name = os.path.basename(url)
        path = get_audio_dir() / name
        if path.exists():
            return str(path)
        backend_base = (os.getenv("BACKEND_URL") or "http://localhost:8001").strip().rstrip("/")
        return f"{backend_base}{url}"
    return None

def _render_audio_if_available(audio_path):
    if not audio_path or audio_path == "None":
        return
    resolved = _resolve_audio_event_path(audio_path)
    if not resolved and os.path.exists(audio_path):
        resolved = audio_path
    if resolved:
        try:
            st.audio(resolved, format="audio/mp3")
        except Exception:
            st.caption("Audio unavailable for this response.")

def _queue_send(text, force_fresh=False):
    payload = (text or "").strip()
    if not payload:
        return
    st.session_state._pending_send_text = payload
    st.session_state._pending_send_force_fresh = bool(force_fresh)
    # Snapshot checkbox values NOW so render-cycle ordering can't stale them
    st.session_state._pending_include_examples = bool(st.session_state.get("include_examples", False))
    st.session_state._pending_include_questions = bool(st.session_state.get("include_questions", False))
    st.session_state._is_generating = True

def _render_live_sections(placeholders, sections):
    feedback = (sections.get("Feedback") or "").strip()
    if feedback:
        placeholders["explanation"].empty()
        placeholders["example"].empty()
        placeholders["question"].empty()
        placeholders["feedback"].markdown(
            f'<div class="ma"><div class="ma-i"><div class="ma-w">Assistant</div>'
            f'<div class="sb sb-f"><span class="st2 st-f">Feedback</span>{_esc(feedback)}</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    placeholders["feedback"].empty()
    exp = (sections.get("Explanation") or "").strip()
    ex = (sections.get("Example") or "").strip()
    q = (sections.get("Question") or "").strip()
    if exp:
        placeholders["explanation"].markdown(
            f'<div class="ma"><div class="ma-i"><div class="ma-w">Assistant</div>'
            f'<div class="sb sb-e"><span class="st2 st-e">Explanation</span>{_esc(exp)}</div></div></div>',
            unsafe_allow_html=True,
        )
    if ex:
        placeholders["example"].markdown(
            f'<div class="ma"><div class="ma-i"><div class="sb sb-ex">'
            f'<span class="st2 st-ex">Example</span>{_esc(ex)}</div></div></div>',
            unsafe_allow_html=True,
        )
    if q:
        placeholders["question"].markdown(
            f'<div class="ma"><div class="ma-i"><div class="sb sb-q">'
            f'<span class="st2 st-q">Question to Think About</span>{_esc(q)}</div></div></div>',
            unsafe_allow_html=True,
        )

def call_explain_stream(topic, **kw):
    from api.routes import ExplainRequest, explain_stream

    chat = kw.get("chat")
    request = ExplainRequest(
        topic=topic,
        age=kw.get("age", st.session_state.age),
        context=kw.get("context") or _chat_context_for_stream(chat),
        doc_id=kw.get("doc_id"),
        user_answer=kw.get("user_answer"),
        profession=st.session_state.profession,
        expertise_level=st.session_state.expertise_level,
        area_of_interest=st.session_state.area_of_interest,
        include_examples=bool(st.session_state.get("_pending_include_examples", st.session_state.get("include_examples", False))),
        include_questions=bool(st.session_state.get("_pending_include_questions", st.session_state.get("include_questions", False))),
        force_new_topic=bool(kw.get("force_new_topic", False)),
        llm_config=_llm_request_config(),
    )

    response = _run_async(explain_stream(request))
    sections = {"Explanation": "", "Example": "", "Question": "", "Feedback": ""}
    audio_path = None
    on_update = kw.get("on_update")

    async def _consume_stream():
        nonlocal audio_path
        buffer = ""
        async for chunk in response.body_iterator:
            text = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
            buffer += text
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                data_lines = []
                for ln in frame.splitlines():
                    if ln.startswith("data:"):
                        data_lines.append(ln[5:].strip())
                if not data_lines:
                    continue
                try:
                    payload = json.loads("\n".join(data_lines))
                except Exception:
                    continue

                evt_type = payload.get("type")
                if evt_type == "update":
                    section = payload.get("section")
                    if section in sections:
                        sections[section] = payload.get("text") or ""
                        if callable(on_update):
                            on_update(dict(sections))
                elif evt_type == "audio":
                    audio_path = _resolve_audio_event_path(payload.get("url"))
                elif evt_type == "error":
                    err = (payload.get("error") or "Stream error").strip()
                    has_partial = any((sections.get(k) or "").strip() for k in sections)
                    if has_partial:
                        continue
                    raise RuntimeError(err)

    _run_async(_consume_stream())
    return sections, audio_path

def call_explain(topic, **kw):
    from graph import learning_graph
    from configs.config import use_request_llm
    _ensure_db()
    runtime_llm = _build_llm()
    chat = kw.get("chat")
    force_new_topic = bool(kw.get("force_new_topic", False))
    messages = kw.get("messages") or _chat_messages_for_graph(chat)
    if not messages:
        messages = [{"role": "user", "content": topic}]
    state = {
        "user_input": topic,
        "messages": messages,
        "learner": {
            "age": kw.get("age", st.session_state.age),
            "difficulty": "medium",
            "profession": st.session_state.profession,
            "expertise_level": st.session_state.expertise_level,
            "area_of_interest": st.session_state.area_of_interest,
        },
        "intent": "new_question",
        "doc_id": kw.get("doc_id"),
        "include_examples": st.session_state.include_examples,
        "include_questions": st.session_state.include_questions,
        "force_new_topic": force_new_topic,
        "retrieved_context": None, "rag_sources": None,
        "simplified_explanation": None, "example": None,
        "safe_text": None, "safety_checked_text": None,
        "thought_question": None, "quiz_question": None,
        "quiz_options": None, "correct_answer": None,
        "user_answer": None, "score": None,
        "final_output": None, "feedback": None,
    }
    with use_request_llm(runtime_llm):
        result = learning_graph.invoke(state)
    fo = result.get("final_output") or {}
    return {
        "Explanation": fo.get("explanation") or result.get("safe_text") or result.get("simplified_explanation") or "",
        "Example": fo.get("example") or result.get("example") or "",
        "Question": fo.get("think_question") or result.get("thought_question") or "",
        "Feedback": result.get("feedback") or "",
    }

def call_upload(fb, fn, **kw):
    from services.document_service import extract_document_text, summarize_with_fallback
    from services.rag_service import get_rag_service
    from db.database import save_document, get_document_by_content_hash
    import hashlib as _hlib
    _ensure_db()
    runtime_llm = _build_llm()
    content_hash = _hlib.sha256(fb).hexdigest()
    existing = get_document_by_content_hash(content_hash)
    if existing:
        doc_id = existing["doc_id"]
        rag = get_rag_service()
        if doc_id not in rag.indices:
            rag.index_document(doc_id, existing["text"])
        summary, _ = summarize_with_fallback(
            existing["text"], st.session_state.age, runtime_llm,
            profession=st.session_state.profession,
            expertise_level=st.session_state.expertise_level,
            area_of_interest=st.session_state.area_of_interest,
        )
        return {"doc_id": doc_id, "filename": fn, "summary": summary}
    extracted, parser = extract_document_text(fn, "", fb)
    cleaned = (extracted or "").strip()
    if len(cleaned) < 80:
        raise ValueError("Could not extract enough readable text from this file")
    summary, _ = summarize_with_fallback(
        cleaned, st.session_state.age, runtime_llm,
        profession=st.session_state.profession,
        expertise_level=st.session_state.expertise_level,
        area_of_interest=st.session_state.area_of_interest,
    )
    doc_id = str(uuid.uuid4())
    save_document(doc_id, fn, cleaned, parser, content_hash=content_hash)
    get_rag_service().index_document(doc_id, cleaned)
    return {"doc_id": doc_id, "filename": fn, "summary": summary}

def call_quiz(topic, num, **kw):
    from api.routes import QuizGenerateRequest, generate_quiz
    _ensure_db()
    req = QuizGenerateRequest(
        topic=topic,
        age=st.session_state.age,
        num_questions=num,
        difficulty=kw.get("difficulty", st.session_state.quiz_difficulty),
        doc_id=kw.get("doc_id"),
        profession=st.session_state.profession,
        expertise_level=st.session_state.expertise_level,
        area_of_interest=st.session_state.area_of_interest,
        llm_config=_llm_request_config(),
    )
    data = _run_async(generate_quiz(req)) or {}
    questions = data.get("questions") or []
    return questions[:num] if questions else [{
        "question": f"What is an important concept in {topic}?",
        "options": {"A": "Core concepts", "B": "Advanced topics", "C": "Real-world use", "D": "All of these"},
        "correct": "D", "explanation": f"All aspects of {topic} are important.",
    }]

def call_validate():
    try:
        from api.routes import ValidateModelRequest, validate_model
        req = ValidateModelRequest(llm_config=_llm_request_config(), prompt="Reply with exactly OK")
        result = _run_async(validate_model(req)) or {}
        return bool(result.get("ok")), (result.get("preview") or "Connected").strip()[:120]
    except Exception as e:
        return False, str(e)

def do_send(text,force_fresh=False,live_target=None):
    text=text.strip()
    if not text: return
    if not st.session_state.active_chat_id: _new_chat(text)
    chat=next(c for c in st.session_state.chats if c["id"]==st.session_state.active_chat_id)
    e=_err()
    if e: chat["messages"].append({"role":"assistant","error":e,"sections":None}); return
    if force_fresh: st.session_state.last_question=""; chat["topic"]=text
    chat["messages"].append({"role":"user","text":text})
    live_box = live_target or st.container()
    live_placeholders = {}
    with live_box:
        live_placeholders = {
            "explanation": st.empty(),
            "example": st.empty(),
            "question": st.empty(),
            "feedback": st.empty(),
        }
    with st.spinner("Thinking…"):
        try:
            s, audio_path = call_explain_stream(
                topic=text,
                age=st.session_state.age,
                doc_id=chat.get("doc_id"),
                chat=chat,
                force_new_topic=force_fresh,
                on_update=lambda sections: _render_live_sections(live_placeholders, sections),
            )
            # Keep the same streamed bubble visible until rerun so there is no visual gap.
            _render_live_sections(live_placeholders, s)
            if s.get("Question"): st.session_state.last_question=s["Question"]
            if not audio_path:
                audio_path = _synthesize_audio(_tts_payload_from_sections(s))
            chat["messages"].append({"role":"assistant","sections":s,"text":"","audio_path":audio_path})
        except Exception as ex:
            chat["messages"].append({"role":"assistant","sections":None,"error":str(ex)})

def paste_to_prompt(text):
    text=(text or "").strip()
    if not text:
        return
    st.session_state._pending_prompt_text = text

def do_upload(f):
    fb=f.read(); fn=f.name
    key=f"{fn}:{hashlib.md5(fb).hexdigest()[:8]}"
    if key in st.session_state._upload_done: return
    st.session_state._upload_done.add(key)
    if not st.session_state.active_chat_id: _new_chat(f"Document: {fn}")
    chat=next(c for c in st.session_state.chats if c["id"]==st.session_state.active_chat_id)
    with st.spinner(f"Uploading {fn}…"):
        try:
            r=call_upload(fb,fn); doc_id=r["doc_id"]
            chat["doc_id"]=doc_id
            if doc_id not in {d["doc_id"] for d in chat.get("docs",[])}:
                chat.setdefault("docs",[]).append({"doc_id":doc_id,"filename":fn})
            audio_path=_synthesize_audio(r.get("summary") or "")
            chat["messages"].append({"role":"assistant","sections":None,
                "text":f"Document Summary: {fn}\n\n{r['summary']}","audio_path":audio_path})
        except Exception as ex:
            chat["messages"].append({"role":"assistant","sections":None,"text":f"Upload failed: {ex}"})

def do_quiz(chat):
    qt=(chat.get("quiz_topic") or "").strip()
    docs=chat.get("docs") or ([{"doc_id":chat["doc_id"],"filename":"Uploaded document"}] if chat.get("doc_id") else [])
    if not qt and not docs: st.error("Enter a topic."); return
    with st.spinner("Generating quiz…"):
        try:
            qs=call_quiz(topic=qt or "the document",num=5,
                difficulty=st.session_state.quiz_difficulty,
                area_of_interest=st.session_state.area_of_interest,
                doc_id=chat.get("quiz_doc_id") or chat.get("doc_id"))
            chat["quiz_state"]=dict(topic=qt or "Document Quiz",questions=qs,current_index=0,
                score=0,answers=[],showing_feedback=False,current_is_correct=None,completed=False)
            chat["show_quiz_setup"]=False
        except Exception as ex: st.error(f"Failed: {ex}")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — scoped carefully, no broad overrides that fight Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css():
    css_path = os.path.join(_HERE, "css", "styles.css")
    try:
        with open(css_path, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {css_path}")

def _inject_js():
    st.markdown("""<script>
(function(){
  // JS runs inside a Streamlit iframe; DOM elements injected via st.markdown
  // also live inside iframes. Walk all frames to find our elements.
  function getDoc(){
    // Try parent frames first (script may run in a nested iframe)
    try{ if(window.parent && window.parent.document) return window.parent.document; }catch(_e){}
    return document;
  }

  /* ── 1. White cards ── */
  function fixCards(){
    var D=getDoc();
    // Page background — teal green
    var app=D.querySelector('[data-testid="stApp"]');
    if(app) app.style.background='#0d6b63';
    var vc=D.querySelector('[data-testid="stAppViewContainer"]');
    if(vc) vc.style.background='#0d6b63';
    var cols=D.querySelectorAll('[data-testid="column"]');
    // All panels: white box on teal background
        [0,1,2].forEach(function(i){
      if(!cols[i]) return;
      var w=cols[i].querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
      if(!w) return;
      w.style.background='#ffffff';
      w.style.borderRadius='18px';
      w.style.border='none';
      w.style.overflow='hidden';
            if(i===1){
                w.style.boxShadow='0 10px 32px rgba(0,0,0,0.28), 0 32px 72px rgba(0,0,0,0.36)';
            } else {
                w.style.boxShadow='0 8px 24px rgba(0,0,0,0.22), 0 24px 60px rgba(0,0,0,0.30)';
            }
    });
  }

  /* ── 2. New Chat button — teal gradient ── */
  function fixNewChat(){
    var D=getDoc();
    D.querySelectorAll('button').forEach(function(b){
      if((b.innerText||'').includes('Start New Chat')){
        b.style.background='linear-gradient(135deg,#0f766e 0%,#109e93 100%)';
        b.style.color='#fff';
        b.style.border='1px solid #0d5f59';
        b.style.borderRadius='12px';
        b.style.fontWeight='800';
        b.style.width='100%';
        b.style.padding='8px 16px';
        b.style.boxShadow='0 4px 14px rgba(15,118,110,.28)';
        b.style.fontFamily='Manrope,sans-serif';
        b.style.letterSpacing='.02em';
        b.style.marginBottom='4px';
      }
    });
  }


  /* ── 4. Collapse left-sidebar chat row gaps ── */
  function fixChatGaps(){
    var D=getDoc();
    var cols=D.querySelectorAll('[data-testid="column"]');
    if(!cols[0]) return;
    var leftCol=cols[0];
    // Zero gap on every stVerticalBlock inside left column
    leftCol.querySelectorAll('[data-testid="stVerticalBlock"]').forEach(function(vb){
      vb.style.gap='0';
      vb.style.rowGap='0';
    });
    // Walk every node and collapse all margins/padding/minHeight
    leftCol.querySelectorAll('[data-testid="element-container"],[data-testid="stHorizontalBlock"]').forEach(function(el){
      el.style.margin='0';
      el.style.padding='0';
      el.style.minHeight='0';
      if(el.parentElement){
        el.parentElement.style.margin='0';
        el.parentElement.style.padding='0';
        el.parentElement.style.minHeight='0';
      }
    });
    // Active state: buttons whose label starts with ►
    leftCol.querySelectorAll('button').forEach(function(b){
      var txt=(b.textContent||b.innerText||'').trim();
      if(txt.charAt(0)==='►'){
        b.style.background='#e2efee';
        b.style.borderColor='#9cc5c1';
        b.style.color='#114745';
        b.style.fontWeight='700';
      }
    });
  }


    /* ── 3. Composer action buttons row — left attach + right grouped quiz/mic/send ── */
  function fixComposerBtns(){
    var D=getDoc();
    // Find the cmp-bar
    var bars=D.querySelectorAll('.cmp-bar');
    bars.forEach(function(bar){
            // Outer row: attach + textarea + actions
      var outerRow=bar.querySelector('[data-testid="stHorizontalBlock"]');
      if(outerRow){
        outerRow.style.gap='4px';
        outerRow.style.alignItems='center';
                // Attach column (1st) and actions column (3rd) fixed widths
        var outerCols=outerRow.children;
                if(outerCols[0]){ outerCols[0].style.flex='0 0 36px'; outerCols[0].style.maxWidth='36px'; outerCols[0].style.padding='0'; }
                if(outerCols[2]){ outerCols[2].style.flex='0 0 116px'; outerCols[2].style.maxWidth='116px'; outerCols[2].style.padding='0'; }
      }
            // Inner actions row: quiz + mic + send
      var allRows=bar.querySelectorAll('[data-testid="stHorizontalBlock"]');
      allRows.forEach(function(row){
        if(row===outerRow) return; // skip outer
                // Do not reshape uploader internals.
                if(row.querySelector('[data-testid="stFileUploader"]')) return;
        row.style.gap='0';
        row.style.alignItems='center';
        row.style.height='36px';
        var cols=row.querySelectorAll(':scope > [data-testid="column"]');
        cols.forEach(function(col,idx){
          col.style.flex='0 0 36px';
          col.style.maxWidth='36px';
          col.style.minWidth='0';
          col.style.padding='0';
          col.style.height='36px';
          var btn=col.querySelector('button');
          if(btn){
            btn.style.height='36px';
            btn.style.width='36px';
            btn.style.minWidth='36px';
            btn.style.maxWidth='36px';
            btn.style.padding='0';
            btn.style.display='flex';
            btn.style.alignItems='center';
            btn.style.justifyContent='center';
            // Capsule: round outer corners only
            var n=cols.length;
            if(idx===0)         btn.style.borderRadius='9px 0 0 9px';
            else if(idx===n-1)  btn.style.borderRadius='0 9px 9px 0';
            else                btn.style.borderRadius='0';
            if(idx>0) btn.style.borderLeft='none';
          }
        });
      });
    });
    // Send button teal highlight
    var sendBtn=D.querySelector('#b_send');
    if(!sendBtn){
      D.querySelectorAll('button').forEach(function(b){
        if((b.getAttribute('title')||'').indexOf('Send')!==-1) sendBtn=b;
      });
    }
    if(sendBtn){
      sendBtn.style.background='linear-gradient(135deg,#0f766e,#109e93)';
      sendBtn.style.color='#fff';
      sendBtn.style.border='1.5px solid #0a5b56';
      sendBtn.style.boxShadow='0 3px 10px rgba(15,118,110,.28)';
    }
  }

    /* ── 5. Live dictation via Web Speech API ── */
    function wireLiveDictation(){
        // Search all accessible frames for our elements
        function findInFrames(selector){
            var frames=[window, window.parent, window.top];
            for(var i=0;i<frames.length;i++){
                try{
                    var el=frames[i].document.querySelector(selector);
                    if(el) return el;
                    // Also search iframes within that frame
                    var iframes=frames[i].document.querySelectorAll('iframe');
                    for(var j=0;j<iframes.length;j++){
                        try{
                            var el2=iframes[j].contentDocument&&iframes[j].contentDocument.querySelector(selector);
                            if(el2) return el2;
                        }catch(_e){}
                    }
                }catch(_e){}
            }
            return null;
        }

        var recCtor=window.SpeechRecognition||window.webkitSpeechRecognition||
                    (window.parent&&(window.parent.SpeechRecognition||window.parent.webkitSpeechRecognition));
        
        if(!window.__axDictation){
            window.__axDictation={active:false,recognizer:null,baseText:'',finalText:''};
        }
        var S=window.__axDictation;

        function getPromptBox(){
            return findInFrames('.cmp-bar textarea[placeholder="Type your message…"]') ||
                   findInFrames('textarea[placeholder="Type your message…"]');
        }

        function getMicBtn(){
            return findInFrames('#ax-mic-btn');
        }

        function setMicVisual(active){
            var btn=getMicBtn();
            if(!btn) return;
            if(active){
                btn.style.background='linear-gradient(135deg,#b91c1c,#dc2626)';
                btn.style.color='#fff';
                btn.style.border='1.5px solid #991b1b';
                btn.style.boxShadow='0 3px 10px rgba(185,28,28,.35)';
            } else {
                btn.style.background='#fff';
                btn.style.color='';
                btn.style.border='1px solid #c5d6d4';
                btn.style.boxShadow='none';
            }
        }

        function pushToPrompt(value){
            var box=getPromptBox();
            if(!box) return;
            box.value=value;
            box.dispatchEvent(new Event('input',{bubbles:true}));
        }

        function startDictation(){
            if(!recCtor){
                alert('Live dictation is not supported in this browser. Use Chrome or Edge.');
                return;
            }
            var box=getPromptBox();
            if(!box) return;

            if(!S.recognizer){
                S.recognizer=new recCtor();
                S.recognizer.continuous=true;
                S.recognizer.interimResults=true;
                S.recognizer.lang='en-US';

                S.recognizer.onresult=function(event){
                    var interim='';
                    for(var i=event.resultIndex;i<event.results.length;i++){
                        var txt=(event.results[i][0]&&event.results[i][0].transcript)||'';
                        if(event.results[i].isFinal) S.finalText+=txt+' ';
                        else interim+=txt;
                    }
                    pushToPrompt((S.baseText+S.finalText+interim).trim());
                };

                S.recognizer.onend=function(){
                    if(S.active){
                        try{ S.recognizer.start(); }catch(_e){}
                    }
                };

                S.recognizer.onerror=function(event){
                    console.log('Speech recognition error:', event.error);
                    if(S.active){
                        S.active=false;
                        setMicVisual(false);
                    }
                };
            }

            S.baseText=((box.value||'').trim() ? (box.value||'').trim()+' ' : '');
            S.finalText='';
            S.active=true;
            setMicVisual(true);
            try{ S.recognizer.start(); }catch(_e){ console.log('Failed to start:', _e); }
        }

        function stopDictation(){
            S.active=false;
            setMicVisual(false);
            if(S.recognizer){
                try{ S.recognizer.stop(); }catch(_e){}
            }
        }

        function toggleDictation(){
            if(S.active) stopDictation();
            else startDictation();
        }

        // Event delegation: bind at document level across frames (survives rerenders)
        if(!window.__axDictationDelegated){
            var frames=[window, window.parent, window.top];
            for(var fi=0;fi<frames.length;fi++){
                try{
                    frames[fi].document.addEventListener('click', function(ev){
                        var btn=ev.target;
                        if(btn && btn.id==='ax-mic-btn'){
                            ev.preventDefault();
                            ev.stopPropagation();
                            toggleDictation();
                        }
                    }, true);
                }catch(_e){}
            }
            window.__axDictationDelegated=true;
        }

        setMicVisual(!!S.active);
    }

  /* Run everything together */
    function runAll(){
        fixCards();
        fixNewChat();
        fixComposerBtns();
        fixChatGaps();
    }

  // Run immediately, then at intervals to catch rerenders
  runAll();
  setTimeout(runAll, 150);
  setTimeout(runAll, 400);
  setTimeout(runAll, 900);
  setTimeout(runAll, 2000);
  // Keep running every 500ms to handle Streamlit rerenders
  setInterval(runAll, 500);

    /* ── 6. Receive live dictation transcript from mic component iframe ── */
    if(!window.__axMsgListener){
        window.__axMsgListener=true;
        var _micBase='';

        function _findPromptBox(){
            var sels=[
                'textarea[placeholder="Type your message\u2026"]',
                'textarea[placeholder="Type your message..."]',
                '.cmp-bar textarea'
            ];
            var roots=[window, window.parent, window.top];
            for(var i=0;i<roots.length;i++){
                try{
                    var doc=roots[i].document;
                    for(var s=0;s<sels.length;s++){
                        var hit=doc.querySelector(sels[s]);
                        if(hit) return hit;
                    }
                    var ifrs=doc.querySelectorAll('iframe');
                    for(var j=0;j<ifrs.length;j++){
                        try{
                            var idoc=ifrs[j].contentDocument;
                            if(!idoc) continue;
                            for(var k=0;k<sels.length;k++){
                                var ih=idoc.querySelector(sels[k]);
                                if(ih) return ih;
                            }
                        }catch(_e){}
                    }
                }catch(_e){}
            }
            return null;
        }

        function _onDictationMessage(ev){
            if(!ev.data) return;
            if(ev.data.type==='ax-dictation-start'){
                var box0=_findPromptBox();
                _micBase=box0 ? ((box0.value||'').trim() ? box0.value.trim()+' ' : '') : '';
                return;
            }
            if(ev.data.type==='ax-dictation'){
                var box=_findPromptBox();
                if(!box) return;
                var nv=(_micBase+(ev.data.text||'')).trim();
                var win=(box.ownerDocument && box.ownerDocument.defaultView) || window;
                var nativeSetter=Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype,'value').set;
                nativeSetter.call(box, nv);
                box.dispatchEvent(new Event('input',{bubbles:true}));
            }
        }

        var _wins=[window, window.parent, window.top];
        for(var wi=0;wi<_wins.length;wi++){
            try{ _wins[wi].addEventListener('message', _onDictationMessage); }catch(_e){}
        }
    }

})();
</script>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_topics(text):
    H="Topics You Can Ask About:"
    idx=text.find(H)
    if idx==-1: return text,[],""
    before=text[:idx].rstrip(); rest=text[idx+len(H):]
    topics,after,inn=[],[],True
    for ln in rest.split("\n"):
        s=ln.strip()
        if not s: continue
        if s=="Would you like to know more about any of these topics?": inn=False; continue
        if inn and s.startswith("- "): topics.append(s[2:].strip())
        else: inn=False; after.append(s)
    return before,topics,"\n".join(after)

def render_msg(msg,idx):
    if msg["role"]=="user":
        st.markdown(f'<div class="mu"><div class="mu-i"><div class="mu-w">You</div>{_esc(msg.get("text",""))}</div></div>',unsafe_allow_html=True); return
    if msg.get("error"):
        st.markdown(f'<div class="ma"><div class="merr">⚠&nbsp;{_esc(msg["error"])}</div></div>',unsafe_allow_html=True); return
    secs=msg.get("sections"); raw=msg.get("text","")
    if secs:
        B=['<div class="ma-w">Assistant</div>']
        if secs.get("Feedback"):
            B.append(f'<div class="sb sb-f"><span class="st2 st-f">Feedback</span>{_esc(secs["Feedback"])}</div>')
        else:
            if secs.get("Explanation"): B.append(f'<div class="sb sb-e"><span class="st2 st-e">Explanation</span>{_esc(secs["Explanation"])}</div>')
            if secs.get("Example"):     B.append(f'<div class="sb sb-ex"><span class="st2 st-ex">Example</span>{_esc(secs["Example"])}</div>')
            if secs.get("Question"):    B.append(f'<div class="sb sb-q"><span class="st2 st-q">Question to Think About</span>{_esc(secs["Question"])}</div>')
        if len(B)>1: st.markdown(f'<div class="ma"><div class="ma-i">{"".join(B)}</div></div>',unsafe_allow_html=True)
        if msg.get("audio_path"):
            _render_audio_if_available(msg.get("audio_path"))
        return
    if not raw: return
    if "Topics You Can Ask About:" in raw:
        before,topics,after=_parse_topics(raw)
        st.markdown(f'<div class="ma"><div class="ma-i"><div class="ma-w">Assistant</div><pre style="white-space:pre-wrap;font-family:\'Manrope\',sans-serif;margin:0;color:#1f3746;font-size:.9rem">{_esc_pre(before)}</pre></div></div>',unsafe_allow_html=True)
        if topics:
            st.markdown('<div class="tbox"><strong>Topics You Can Ask About:</strong>',unsafe_allow_html=True)
            st.markdown('<div class="pill-row">',unsafe_allow_html=True)
            pc=st.columns(min(len(topics),4))
            for i,t in enumerate(topics):
                with pc[i%len(pc)]:
                    if st.button(t,key=f"pill_{idx}_{i}",use_container_width=True): _queue_send(t,force_fresh=True); st.rerun()
            st.markdown('</div><p>Click a topic or type your own question below.</p></div>',unsafe_allow_html=True)
        if after: st.markdown(f'<div class="ma"><div class="ma-i"><div class="sb sb-e">{_esc(after)}</div></div></div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ma"><div class="ma-i"><div class="ma-w">Assistant</div><div style="white-space:pre-wrap;line-height:1.55;font-size:.9rem">{_esc(raw)}</div></div></div>',unsafe_allow_html=True)
        if msg.get("audio_path"):
            _render_audio_if_available(msg.get("audio_path"))

def render_quiz_setup(chat):
    docs=chat.get("docs") or ([{"doc_id":chat["doc_id"],"filename":"Uploaded document"}] if chat.get("doc_id") else [])
    is_doc=bool(docs)
    st.markdown('<h2 style="margin:0 0 12px;font-size:1rem;font-weight:700;color:#0f1722">📝 Quiz Setup</h2>',unsafe_allow_html=True)
    if is_doc:
        st.caption("Questions from uploaded document.")
        qt=st.text_input("Focus area (optional)",value=chat.get("quiz_topic",""),key="qs_topic")
    else:
        qt=st.text_input("Quiz Topic *",value=chat.get("quiz_topic",""),placeholder="e.g. photosynthesis, planets",key="qs_topic")
        st.caption(f"Chat: {chat.get('topic','')}")
    chat["quiz_topic"]=qt or ""
    di=st.select_slider("Difficulty",["easy","medium","hard"],value=st.session_state.quiz_difficulty,key="qs_diff")
    st.session_state.quiz_difficulty=di
    bc={"easy":("#d1fae5","#065f46"),"medium":("#fef3c7","#92400e"),"hard":("#fee2e2","#991b1b")}
    bg,fg=bc[di]
    st.markdown(f'<span style="display:inline-block;padding:3px 12px;border-radius:999px;font-size:.76rem;font-weight:700;background:{bg};color:{fg}">{di}</span>',unsafe_allow_html=True)
    st.slider("Age",5,35,value=st.session_state.age,key="qs_age")
    st.session_state.age=st.session_state.qs_age
    st.selectbox("Profession",PROFESSION_OPTIONS,index=PROFESSION_OPTIONS.index(st.session_state.profession) if st.session_state.profession in PROFESSION_OPTIONS else 0,key="qs_profession")
    st.session_state.profession=st.session_state.qs_profession
    st.selectbox("Expertise",["Beginner","Intermediate","Advanced"],index=["Beginner","Intermediate","Advanced"].index(st.session_state.expertise_level) if st.session_state.expertise_level in ["Beginner","Intermediate","Advanced"] else 0,key="qs_expertise")
    st.session_state.expertise_level=st.session_state.qs_expertise
    st.selectbox("Example Context",AREA_OPTIONS,index=AREA_OPTIONS.index(st.session_state.area_of_interest) if st.session_state.area_of_interest in AREA_OPTIONS else 0,key="qs_area")
    can=is_doc or bool((chat.get("quiz_topic") or "").strip())
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="q-secondary">',unsafe_allow_html=True)
        if st.button("Cancel",key="qs_cancel",use_container_width=True): chat["show_quiz_setup"]=False; st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="q-primary">',unsafe_allow_html=True)
        if st.button("Generate Quiz (5 questions)",key="qs_gen",disabled=not can,use_container_width=True): do_quiz(chat); st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

def render_quiz(chat):
    qs=chat["quiz_state"]
    if qs.get("completed"):
        tot=len(qs["questions"]); pct=round(qs["score"]/tot*100) if tot else 0
        st.markdown(f'<div class="qscore">{qs["score"]}/{tot}<div class="qpct">{pct}%</div></div>',unsafe_allow_html=True)
        st.markdown('<h3 style="color:#1f3746;margin:0 0 6px;font-size:.9rem">Review</h3>',unsafe_allow_html=True)
        for i,a in enumerate(qs["answers"]):
            cls="rev rev-ok" if a["isCorrect"] else "rev rev-no"; icon="✓" if a["isCorrect"] else "✗"
            ca=f'<br><strong style="color:#991b1b">Correct: {_esc(a["correctAnswer"])}</strong>' if not a["isCorrect"] else ""
            cx=f'<br><span class="qctx"><strong>Context:</strong> {_esc(a.get("context",""))}</span>' if a.get("context") else ""
            ex=f'<br><span style="color:#315060;font-size:.86rem">{_esc(a.get("explanation",""))}</span>' if a.get("explanation") else ""
            st.markdown(f'<div class="{cls}"><strong>Q{i+1}:</strong> {_esc(a["question"])}<br>Your answer: <strong>{_esc(a["userAnswer"])}</strong> {icon}{ca}{cx}{ex}</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="q-primary">',unsafe_allow_html=True)
            if st.button("Try Again",key="qr_retry",use_container_width=True): chat["show_quiz_setup"]=True; chat["quiz_state"]=None; st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="q-secondary">',unsafe_allow_html=True)
            if st.button("Back to Explain",key="qr_back",use_container_width=True): chat["quiz_state"]=None; chat["show_quiz_setup"]=False; st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)
        return
    q=qs["questions"][qs["current_index"]]; tot=len(qs["questions"])
    qctx=(q.get("example_context") or q.get("context") or q.get("area_of_interest") or "").strip()
    pct=round(qs["current_index"]/tot*100); ac=len(qs["answers"])
    st.markdown(f'<div style="background:#fff;border:1px solid #d7e7e5;border-radius:12px;padding:10px 12px;margin-bottom:8px;box-shadow:0 2px 10px rgba(20,58,80,.06)"><h2 style="margin:0 0 5px;color:#1f3746;font-size:.92rem;font-weight:700">Quiz: {_esc(qs["topic"])}</h2><div style="display:flex;align-items:center;justify-content:space-between;color:#5a6e7b;font-size:.76rem;font-weight:600;margin-bottom:4px"><span>Q {qs["current_index"]+1}/{tot}</span><span style="border:1px solid #b7ddd8;border-radius:999px;padding:1px 7px;background:#f2fbf8;color:#21665d">Score: {qs["score"]}/{ac if ac else "–"}</span></div><div class="qbar"><div class="qfill" style="width:{pct}%"></div></div></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="background:#fff;border:1px solid #d7e7e5;border-radius:12px;padding:12px;box-shadow:0 2px 10px rgba(20,58,80,.06)"><p style="font-size:.66rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#5f7482;margin:0 0 4px">Choose one answer</p><h3 style="font-size:.9rem;font-weight:700;line-height:1.35;margin:0 0 8px;color:#1f3746">{_esc(q["question"])}</h3>',unsafe_allow_html=True)
    if qctx:
        st.markdown(f'<div class="qctx"><strong>Context:</strong> {_esc(qctx)}</div>',unsafe_allow_html=True)
    if not qs["showing_feedback"]:
        for k,v in q["options"].items():
            if st.button(f"{k})  {v}",key=f"qo_{qs['current_index']}_{k}",use_container_width=True):
                ic=(k==q["correct"]); qs["answers"].append({"question":q["question"],"userAnswer":k,"correctAnswer":q["correct"],"isCorrect":ic,"explanation":q.get("explanation",""),"context":qctx})
                qs["score"]+=1 if ic else 0; qs["showing_feedback"]=True; qs["current_is_correct"]=ic; st.rerun()
    else:
        la=qs["answers"][-1]["userAnswer"] if qs["answers"] else None
        for k,v in q["options"].items():
            ir=k==q["correct"]; wp=k==la; cls="qc" if ir else ("qi" if wp else "")
            st.markdown(f'<div class="qopt {cls}"><span class="qopt-key">{k}</span><span>{_esc(v)}</span></div>',unsafe_allow_html=True)
        fbc="qfb-ok" if qs["current_is_correct"] else "qfb-no"
        fbt="✓ Correct!" if qs["current_is_correct"] else f'✗ Not quite! Correct: <strong>{_esc(q["correct"])}</strong>'
        exp=f'<p style="margin:6px 0 0">{_esc(q.get("explanation",""))}</p>' if q.get("explanation") else ""
        st.markdown(f'<div class="{fbc}">{fbt}{exp}</div>',unsafe_allow_html=True)
        lbl="Next Question →" if qs["current_index"]<tot-1 else "View Results"
        st.markdown('<div class="q-primary">',unsafe_allow_html=True)
        if st.button(lbl,key="q_next",use_container_width=True):
            if qs["current_index"]<tot-1: qs["current_index"]+=1; qs["showing_feedback"]=False; qs["current_is_correct"]=None
            else: qs["completed"]=True
            st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

def render_left():
    if st.button("＋  Start New Chat",use_container_width=True,key="btn_new"):
        st.session_state.active_chat_id=None; st.session_state.last_question=""; st.rerun()
    hL,hR=st.columns([3,2.4],gap="small")
    with hL: st.markdown('<p style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6b7d8c;margin:8px 0 5px">Chats</p>',unsafe_allow_html=True)
    with hR:
        if st.session_state.chats:
            if st.button("Clear All",key="btn_clear",use_container_width=False): st.session_state.chats=[]; st.session_state.active_chat_id=None; st.rerun()
    if not st.session_state.chats:
        st.markdown('<p style="font-size:13px;color:#7a8d99;margin:3px 0">No chat history yet.</p>',unsafe_allow_html=True)
    for c in st.session_state.chats:
        active=c["id"]==st.session_state.active_chat_id
        cL,cR=st.columns([6,1],gap="small")
        with cL:
            label=("► " if active else "")+c["topic"][:38]
            if st.button(label,key=f"c_{c['id']}",use_container_width=True): st.session_state.active_chat_id=c["id"]; st.rerun()
        with cR:
            if st.button("×",key=f"d_{c['id']}"):
                st.session_state.chats=[x for x in st.session_state.chats if x["id"]!=c["id"]]
                if st.session_state.active_chat_id==c["id"]: st.session_state.active_chat_id=None
                st.rerun()

def render_right():
    first=[True]
    def H(t):
        mt="0" if first[0] else "6px"; first[0]=False
        st.markdown(f'<p style="font-size:.64rem;font-weight:700;letter-spacing:.10em;text-transform:uppercase;color:#6b7d8c;margin:{mt} 0 2px;line-height:1">{t}</p>',unsafe_allow_html=True)
    H("Model")
    if st.session_state.get("_model_provider_selected") not in PROVIDERS:
        st.session_state._model_provider_selected=PROVIDERS[0]
    if st.session_state.get("model_provider_widget") not in PROVIDERS:
        st.session_state.model_provider_widget=st.session_state._model_provider_selected
    st.selectbox("Provider",PROVIDERS,format_func=lambda p:PLABELS.get(p,p),key="model_provider_widget",
        label_visibility="collapsed")
    sel=st.session_state.model_provider_widget
    if sel!=st.session_state._model_provider_selected:
        # Persist current provider key before switching.
        prev = st.session_state._model_provider_selected
        byp = st.session_state.get("_api_keys_by_provider") or {}
        prev_key = (st.session_state.get("model_api_key") or "").strip()
        if prev in PROVIDERS and prev_key:
            byp[prev] = prev_key
        st.session_state._model_provider_selected=sel
        st.session_state.model_provider=sel
        # Restore selected provider key (if any).
        st.session_state.model_api_key=(byp.get(sel) or "").strip()
        st.session_state._api_keys_by_provider=byp
        st.session_state.model_validation_result=None
    else:
        st.session_state.model_provider=st.session_state._model_provider_selected
    p=(st.session_state.get("model_provider") or "").lower()
    if p and p!="local":
        byp = st.session_state.get("_api_keys_by_provider") or {}
        if not (st.session_state.get("model_api_key") or "").strip() and (byp.get(p) or "").strip():
            st.session_state.model_api_key = (byp.get(p) or "").strip()
        st.text_input("API Key",type="password",key="model_api_key",placeholder="Enter API key",label_visibility="collapsed")
        # Keep a per-provider copy so token survives reruns/topic clicks.
        byp = st.session_state.get("_api_keys_by_provider") or {}
        cur_key = (st.session_state.get("model_api_key") or "").strip()
        if cur_key:
            byp[p] = cur_key
            st.session_state._api_keys_by_provider = byp
    elif p=="local": st.markdown('<p style="font-size:.74rem;color:#6b7280;margin:0 0 2px;font-style:italic">Local — no key needed.</p>',unsafe_allow_html=True)
    if st.button("Test Model Connection",key="btn_val",use_container_width=True):
        with st.spinner("Testing connection…"):
            ok,msg=call_validate()
        st.session_state.model_validation_result={"ok":ok,"message":msg}
    if st.session_state.model_validation_result:
        r=st.session_state.model_validation_result
        st.markdown(f'<p class="{"val-ok" if r["ok"] else "val-err"}">{_esc(r["message"])}</p>',unsafe_allow_html=True)
    H("Difficulty"); st.slider("Age",5,35,key="age",label_visibility="visible")
    H("Profile")
    st.selectbox("Profession",PROFESSION_OPTIONS,key="profession",label_visibility="visible")
    st.selectbox("Expertise Level",["Beginner","Intermediate","Advanced"],key="expertise_level",label_visibility="visible")
    st.selectbox("Example Context",AREA_OPTIONS,key="area_of_interest",label_visibility="visible")
    st.markdown('<p style="font-size:.70rem;color:#6b7d8c;margin:0 0 3px;font-style:italic">Examples only; topic facts unchanged.</p>',unsafe_allow_html=True)
    H("Output")
    c1,c2=st.columns(2,gap="small")
    with c1: st.checkbox("Include examples",key="include_examples")
    with c2: st.checkbox("Include questions",key="include_questions")
    H("Topic Packs")
    packs=["— choose a pack —"]+sorted(TOPICS_DATA.keys())
    st.selectbox("Pack",packs,key="selected_pack",label_visibility="collapsed")
    if st.session_state.selected_pack and st.session_state.selected_pack != "— choose a pack —":
        if st.session_state._last_selected_pack != st.session_state.selected_pack:
            st.session_state._last_selected_pack = st.session_state.selected_pack
            st.session_state._last_pack_topic = ""
            st.session_state._pending_pack_topic = ""
        topics=TOPICS_DATA[st.session_state.selected_pack]
        use_button_fallback = pills is None
        if pills is not None:
            try:
                chosen = pills("Subtopics", topics, index=None, clearable=True)
            except TypeError:
                # Older streamlit-pills may select first option by default; use fallback selectbox instead.
                chosen = None
                use_button_fallback = True
            except Exception:
                chosen = None
                use_button_fallback = True
            if chosen:
                if chosen != st.session_state._last_pack_topic:
                    st.session_state._pending_pack_topic = chosen
                    st.session_state._last_pack_topic = chosen
                    paste_to_prompt(chosen)
                    st.rerun()
        if use_button_fallback:
            t_opts=["— select subtopic —"]+topics
            chosen_fb=st.selectbox("Subtopic",t_opts,key=f"subtopic_{st.session_state.selected_pack}",label_visibility="collapsed")
            if chosen_fb and chosen_fb != "— select subtopic —":
                if chosen_fb != st.session_state._last_pack_topic:
                    st.session_state._pending_pack_topic = chosen_fb
                    st.session_state._last_pack_topic = chosen_fb
                    paste_to_prompt(chosen_fb)
                    st.rerun()
            elif chosen_fb == "— select subtopic —":
                st.session_state._pending_pack_topic = ""
                st.session_state._last_pack_topic = ""

def render_center(chat):
    if not st.session_state.get("_db_ready"):
        with st.spinner("Initializing backend services... please wait"):
            try:
                _ensure_db()
            except Exception as ex:
                st.error(f"Backend initialization failed: {ex}")
                return
        st.caption("Backend initialized. You can start typing now.")

    if chat and chat.get("show_quiz_setup"):
        st.markdown('<div class="quiz-pane" style="padding:18px 18px 20px">',unsafe_allow_html=True)
        render_quiz_setup(chat)
        st.markdown('</div>',unsafe_allow_html=True)
        return
    if chat and chat.get("quiz_state"):
        st.markdown('<div class="quiz-pane" style="padding:14px 18px 20px">',unsafe_allow_html=True)
        render_quiz(chat)
        st.markdown('</div>',unsafe_allow_html=True)
        return

    pending_preview = (st.session_state.get("_pending_send_text") or "").strip()

    msg_box=st.container(height=760,border=False)
    with msg_box:
        if not chat or not chat.get("messages"):
            st.markdown('<p style="color:#6b7d8c;text-align:center;margin-top:80px;font-size:.95rem">Select a topic or type a question to begin.</p>',unsafe_allow_html=True)
        else:
            for i,m in enumerate(chat["messages"]): render_msg(m,i)
        if pending_preview:
            render_msg({"role":"user","text":pending_preview}, -1)

    if st.session_state.get("_pending_send_text"):
        send_text = st.session_state._pending_send_text
        send_force = bool(st.session_state.get("_pending_send_force_fresh", False))
        st.session_state._pending_send_text = ""
        st.session_state._pending_send_force_fresh = False
        try:
            do_send(send_text, force_fresh=send_force, live_target=msg_box)
        finally:
            st.session_state._is_generating = False
        st.session_state._ic += 1
        st.rerun()

    # ── Composer — fixed bottom, NO scroll ──
    # Wrap in a div class so CSS .cmp-bar targets it
    st.markdown('<div class="cmp-bar">',unsafe_allow_html=True)
    err_txt=_err()
    blocked=bool(err_txt) or bool(st.session_state.get("_is_generating"))
    ik=f"ci_{st.session_state._ic}"
    if st.session_state._pending_prompt_text:
        st.session_state[ik] = st.session_state._pending_prompt_text
        st.session_state._pending_prompt_text = ""

    # 3 columns: attach | textarea | [quiz+mic+send grouped]
    # Keep a slightly wider attach lane so uploader chrome never overlaps the textarea.
    c_att, c_inp, c_acts = st.columns([0.72, 5.4, 1.65], gap="small")

    with c_att:
        st.markdown('<div class="cmp-attach">',unsafe_allow_html=True)
        with st.popover("📎", use_container_width=True):
            up = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "txt", "md", "csv", "html"],
                key=f"up_{st.session_state._ic}",
                label_visibility="visible",
                help="Upload document",
            )
            if up:
                fn = up.name
                sz = up.getbuffer().nbytes / 1024 / 1024
                up.seek(0)
                st.toast(f"✓ {fn} ({sz:.1f}MB) uploading…", icon="📎")
                do_upload(up)
                st.session_state._ic += 1
                st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

    with c_inp:
        txt=st.text_area(
            "msg",
            key=ik,
            placeholder="Type your message…",
            label_visibility="collapsed",
            disabled=blocked,
            height=32,
        )
    pending=(txt or "").strip()

    # Action buttons — nested 3 equal columns so all buttons are identical size/height
    with c_acts:
        qok=bool(chat and chat.get("messages") and not chat.get("quiz_state") and not chat.get("show_quiz_setup"))
        ca, cb, cc = st.columns([1,1,1], gap="small")
        with ca:
            if st.button("🎯",key="b_quiz",disabled=not qok,help="Start quiz",use_container_width=True):
                qt=re.sub(r"^Document:\s*","",chat.get("topic",""),flags=re.IGNORECASE)
                qt=re.sub(r"\.(pdf|docx|txt|md|csv|html?)$","",qt,flags=re.IGNORECASE).strip()
                chat["show_quiz_setup"]=True; chat["quiz_topic"]=qt; st.rerun()
        with cb:
            mic_html = """
<style>
  body{margin:0;padding:0;overflow:hidden;}
  #mic{width:100%;height:36px;border:1px solid #c5d6d4;border-radius:9px;
       background:#fff;cursor:pointer;font-size:18px;font-weight:bold;
       font-family:inherit;display:flex;align-items:center;justify-content:center;}
  #mic.active{background:linear-gradient(135deg,#b91c1c,#dc2626);color:#fff;
              border-color:#991b1b;box-shadow:0 3px 10px rgba(185,28,28,.35);}
</style>
<button id="mic" onclick="toggle()">🎤</button>
<script>
  var recCtor=window.SpeechRecognition||window.webkitSpeechRecognition;
  var rec=null, active=false, finalText='';

  function toggle(){
    if(!recCtor){ alert('Live dictation needs Chrome or Edge.'); return; }
    if(active) stop(); else start();
  }

  function start(){
    if(!rec){
      rec=new recCtor();
      rec.continuous=true;
      rec.interimResults=true;
      rec.lang='en-US';
      rec.onresult=function(e){
        var interim='';
        for(var i=e.resultIndex;i<e.results.length;i++){
          var t=(e.results[i][0]&&e.results[i][0].transcript)||'';
          if(e.results[i].isFinal) finalText+=t+' ';
          else interim+=t;
        }
                var msg={type:'ax-dictation',text:(finalText+interim).trim()};
                try{ window.postMessage(msg,'*'); }catch(_e){}
                try{ window.parent.postMessage(msg,'*'); }catch(_e){}
                try{ window.top.postMessage(msg,'*'); }catch(_e){}
      };
      rec.onend=function(){ if(active){ try{rec.start();}catch(_){} } };
      rec.onerror=function(e){ console.log('mic error',e.error); active=false; setVisual(false); };
    }
    finalText='';
    // Signal parent to snapshot current textarea value as base
    var startMsg={type:'ax-dictation-start'};
    try{ window.postMessage(startMsg,'*'); }catch(_e){}
    try{ window.parent.postMessage(startMsg,'*'); }catch(_e){}
    try{ window.top.postMessage(startMsg,'*'); }catch(_e){}
    active=true; setVisual(true);
    try{ rec.start(); }catch(e){ console.log(e); }
  }

  function stop(){
    active=false; setVisual(false);
    try{ rec.stop(); }catch(_){}
  }

  function setVisual(on){
    var b=document.getElementById('mic');
    if(on) b.classList.add('active'); else b.classList.remove('active');
  }
</script>
"""
            mic_src = "data:text/html;charset=utf-8," + quote(mic_html)
            st.iframe(mic_src, height=40)
        with cc:
            sent=st.button("➤",key="b_send",disabled=blocked,help="Send",use_container_width=True)

    if sent and pending:
        _queue_send(pending)
        st.rerun()
    if err_txt:
        st.markdown(f'<p style="font-size:.74rem;color:#a62f2f;margin:3px 0 0;text-align:center">{_esc(err_txt)}</p>',unsafe_allow_html=True)
    elif st.session_state.get("_is_generating"):
        st.markdown('<p style="font-size:.74rem;color:#0f766e;margin:3px 0 0;text-align:center">Generating response...</p>',unsafe_allow_html=True)

    st.markdown('</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AgeXplain",page_icon="🧠",layout="wide",initial_sidebar_state="collapsed")
_inject_css()
_inject_js()

chat=_chat()
L,C,R=st.columns([0.82,3.0,1.10],gap="small")
with L:
    with st.container(border=True): render_left()
with C:
    with st.container(border=True): render_center(chat)
with R:
    with st.container(border=True): render_right()
