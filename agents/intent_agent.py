from configs.models import ExplainState
from services.intent_service import classify_intent


def infer_intent(state: ExplainState):
    user_input = (state.get("user_input") or "").strip()
    messages = state.get("messages", [])
    doc_id = state.get("doc_id")
    force_new_topic = bool(state.get("force_new_topic"))
    user_answer_hint = state.get("user_answer")

    result = classify_intent(
        user_input=user_input,
        messages=messages,
        doc_id=doc_id,
        force_new_topic=force_new_topic,
        user_answer_hint=user_answer_hint,
    )
    return result
    