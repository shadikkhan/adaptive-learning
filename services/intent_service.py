from __future__ import annotations

import re
from typing import Dict, List, Optional

EXAMPLE_HINTS = {
    "another example", "give me an example", "show me an example",
    "example please", "can you give an example", "one more example"
}

SIMPLIFY_HINTS = {
    "simpler", "simple", "in simpler words", "simple words", "simple explanation",
    "break it down", "dumb it down", "easy version"
}

FOLLOWUP_HINTS = {
    "tell me more", "continue", "explain again", "what about", "difference", "next question", "more details"
}

QUESTION_REQUEST_HINTS = {
    "ask me", "give me a question", "ask me a question", "ask me another question",
    "quiz me", "test me", "what's the question"
}

QUESTION_STARTERS = (
    "what ", "why ", "how ", "when ", "where ", "who ", "which ",
    "can ", "could ", "is ", "are ", "does ", "do ", "did ", "explain ", "summarize "
)

ANSWER_LEADS = (
    "because", "i think", "it is", "it was", "they are", "they were", "yes", "no", "my answer"
)


def _last_assistant_question(messages: List[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "")
        if "Question:" in content:
            tail = content.rsplit("Question:", 1)[1]
            return tail.splitlines()[0].strip()
    return ""


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    if t.startswith(QUESTION_STARTERS):
        return True
    if re.match(r"^(please\s+)?(explain|summarize|compare|list|define)\b", t):
        return True
    return False


def _looks_like_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return False
    if t.startswith(ANSWER_LEADS):
        return True
    tokens = t.split()
    if len(tokens) <= 14 and not _looks_like_question(t):
        return True
    return False


def classify_intent(
    user_input: str,
    messages: Optional[List[dict]] = None,
    doc_id: Optional[str] = None,
    force_new_topic: bool = False,
    user_answer_hint: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    text = (user_input or "").strip()
    lowered = text.lower()
    messages = messages or []

    if force_new_topic:
        return {"intent": "new_question", "user_answer": None}

    has_pending_question = bool(_last_assistant_question(messages))
    looks_question = _looks_like_question(text)
    looks_answer = _looks_like_answer(text)
    
    has_example_hint = any(h in lowered for h in EXAMPLE_HINTS)
    has_simplify_hint = any(h in lowered for h in SIMPLIFY_HINTS)
    has_followup_hint = any(h in lowered for h in FOLLOWUP_HINTS)
    has_question_request = any(h in lowered for h in QUESTION_REQUEST_HINTS)

    # 1) User explicitly asks for a question ("ask me a question")
    if has_question_request and not looks_question:
        return {"intent": "followup", "user_answer": None}  # Will trigger question generation

    # 2) User explicitly asks for an example
    if has_example_hint:
        return {"intent": "followup_example", "user_answer": None}

    # 3) User asks for simplification
    if has_simplify_hint:
        if doc_id:
            return {"intent": "document_question", "user_answer": None}
        return {"intent": "followup", "user_answer": None}

    # 4) User answers a pending question
    if user_answer_hint and not looks_question:
        return {"intent": "answer", "user_answer": text}

    if has_pending_question and (looks_answer and not has_followup_hint):
        return {"intent": "answer", "user_answer": text}

    # 5) Other follow-up requests
    if has_followup_hint:
        if doc_id:
            return {"intent": "document_question", "user_answer": None}
        return {"intent": "followup", "user_answer": None}

    # 6) User asks a question
    if looks_question:
        if doc_id:
            return {"intent": "document_question", "user_answer": None}
        if has_pending_question:
            return {"intent": "followup", "user_answer": None}
        return {"intent": "new_question", "user_answer": None}

    # 7) User answers pending question (general case)
    if has_pending_question:
        return {"intent": "answer", "user_answer": text}

    return {"intent": "new_question", "user_answer": None}
