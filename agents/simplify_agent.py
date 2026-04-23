# Turn raw question / follow-up → age-appropriate explanation
import re

from configs.models import ExplainState
from configs.config import llm
from services.json_logger import log_event

log_event("simplify_agent.loaded")


def _infer_doc_style(text: str) -> str:
    sample = (text or "").lower()
    research_hits = sum(k in sample for k in [
        "abstract", "method", "dataset", "experiment", "results", "accuracy", "benchmark", "glue", "squad"
    ])
    book_hits = sum(k in sample for k in [
        "chapter", "part ", "prologue", "epilogue", "author", "book"
    ])
    if research_hits >= 3 and research_hits >= book_hits + 1:
        return "research"
    if book_hits >= 2 and book_hits >= research_hits:
        return "book"
    return "generic"


def _style_rules(style: str) -> str:
    if style == "research":
        return (
            "- Prefer concrete factual details from the retrieved text (task names, metrics, comparisons) when available.\n"
            "- If metrics or benchmark names are present, mention at least two concrete ones in plain language when possible.\n"
            "- Do NOT use competition/game metaphors or generic storytelling.\n"
            "- Do NOT mention external models/tools unless they appear in the retrieved context.\n"
            "- Keep wording precise, concise, and evidence-based."
        )
    if style == "book":
        return (
            "- Keep focus on arguments/themes from the text, not generic AI explanations.\n"
            "- Mention one key claim and one implication when available.\n"
            "- Avoid fabricated chapter details not present in context."
        )
    return (
        "- Stay tightly grounded in retrieved context.\n"
        "- Prefer specific facts over generic statements."
    )


def _remove_unsupported_references(text: str, context: str) -> str:
    """Drop sentences that mention external model/tool names absent from retrieved context."""
    lowered_context = (context or "").lower()
    flagged_terms = [
        "openai", "gpt", "elmo", "xlnet", "roberta", "t5", "llama", "bert-large",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = []
    for s in sentences:
        ls = s.lower()
        has_unsupported = any(term in ls and term not in lowered_context for term in flagged_terms)
        if not has_unsupported:
            kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned or (text or "").strip()


def _strip_meta_openers(text: str) -> str:
    out = (text or "").strip()
    # Remove template-like openers that reduce enterprise quality.
    out = re.sub(r"^\s*Here'?s an explanation[^:]*:\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^\s*So, you (?:asked|want to know)[^\.]*\.\s*", "", out, flags=re.IGNORECASE)
    return out.strip()


def _extract_last_assistant_context(messages):
    """
    Pull most recent assistant Explanation/Example/Question text from parsed messages.
    """
    explanation = ""
    example = ""
    question = ""

    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue

        content = (msg.get("content") or "").strip()
        if not content:
            continue

        # Content may be a combined block with multiple lines
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("Explanation: ") and not explanation:
                explanation = line[len("Explanation: "):].strip()
            elif line.startswith("Example: ") and not example:
                example = line[len("Example: "):].strip()
            elif line.startswith("Question: ") and not question:
                question = line[len("Question: "):].strip()

        if explanation or example or question:
            break

    return explanation, example, question


def _first_user_topic(messages):
    """
    Get the first user message in the chat history as the core topic anchor.
    Example: 'Explain volcanoes for a 6-year-old'
    """
    for msg in messages or []:
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return ""

def simplify_agent(state: ExplainState):
    log_event(
        "simplify_agent.start",
        intent=state.get("intent"),
        user_input=(state.get("user_input") or "")[:200],
        has_messages=bool(state.get("messages")),
    )

    age = state["learner"]["age"]
    learner = state.get("learner", {})
    profession = (learner.get("profession") or "").strip()
    expertise_level = (learner.get("expertise_level") or "").strip()
    area_of_interest = (learner.get("area_of_interest") or "").strip()
    learner_profile = f"""
Learner profile:
- Profession: {profession or "Not provided"}
- Expertise level: {expertise_level or "Not provided"}
- Area of interest: {area_of_interest or "Not provided"}
""".strip()
    personalization_rule = (
        "- If Area of interest is provided, include one short relatable context sentence using that area "
        "without changing the topic facts."
        if area_of_interest
        else "- Keep examples neutral unless an area of interest is provided."
    )
    user_input = (state.get("user_input") or "").strip()
    messages = state.get("messages", [])
    intent = state.get("intent", "new_question")
    retrieved_context = state.get("retrieved_context")
    doc_style = _infer_doc_style(retrieved_context or "")

    normalized_input = user_input.lower()
    is_simplify_followup = (
        intent == "followup"
        and any(p in normalized_input for p in [
            "simpler",
            "simple words",
            "in simpler words",
            "explain that",
            "explain this again"
        ])
    )
    log_event(
        "simplify_agent.mode",
        is_simplify_followup=is_simplify_followup,
        has_retrieved_context=bool(retrieved_context),
        doc_style=doc_style,
    )

    prev_explanation, prev_example, prev_question = _extract_last_assistant_context(messages)
    topic_anchor = _first_user_topic(messages)
    log_event("simplify_agent.topic_anchor", topic_anchor=(topic_anchor or "")[:200])

    # 1a) Document question — strict RAG-grounded answer
    if intent == "document_question" and not retrieved_context:
        explanation = (
            "I could not find relevant context from your uploaded document right now. "
            "Please re-upload the file and ask the question again in the same chat."
        )
        return {"simplified_explanation": explanation}

    # 1b) Hybrid grounded-tutor mode:
    # new_question or followup BUT doc is loaded and RAG retrieved something.
    # Ground the explanation in the document while keeping full tutor style.
    if intent in ("new_question", "followup") and retrieved_context:
        context_section = f"""
Relevant excerpts from the learner's uploaded document:
{retrieved_context}
"""
        if intent == "followup" and (prev_explanation or prev_question):
            prev_section = f"""
Previous explanation (same topic):
{prev_explanation or "(none)"}

Previous question asked to learner:
{prev_question or "(none)"}
"""
        else:
            prev_section = ""

        prompt = f"""
You are AgeXplain, a tutor.

A learner aged {age} is exploring an uploaded document and has said:
"{user_input}"
{prev_section}
{context_section}
Task:
- Explain the topic/follow-up clearly for age {age}.
- Ground your explanation in the document excerpts above.
- If evidence is weak, say "This is not clearly stated in the document" once, then provide best-effort clarification.

Rules:
- Write for EXACTLY age {age}.
- If age <= 7: 1-2 sentences max, everyday words only, no technical terms, simple analogies.
- If 8 <= age <= 10: short paragraphs, define one hard word inline, simple analogies.
- If 11 <= age <= 14: school-level vocabulary, 1-2 technical terms OK if explained briefly.
- If 15 <= age <= 18: high-school depth, abstraction and cause-effect OK, domain terms welcome.
- If age >= 19: peer-level adult language, precise terminology, skip basics unless expertise is Beginner.
- If expertise is Beginner: always define the first key term, assume zero prior knowledge regardless of age.
- If expertise is Advanced and age >= 19: skip basic definitions, use domain-precise language directly.
- Do NOT repeat the excerpts verbatim. Use them as source material.
- Do NOT say "According to the document" repeatedly.
- Keep direct alignment with the latest user topic: "{user_input}".
- Use the learner profile to tune wording and analogy level.
- {personalization_rule}
- Document style detected: {doc_style}.
{_style_rules(doc_style)}
- Start with the direct answer to the selected topic, then add supporting detail.
- If style is research: use 3 short parts in order:
    1) direct answer sentence,
    2) evidence sentence with benchmark/metric from context,
    3) implication sentence.
- Return ONLY the explanation paragraph(s). No heading.
""".strip()

    elif intent == "document_question" and retrieved_context:
        prompt = f"""
You are AgeXplain, a tutor.

The learner is asking a question about an uploaded document.

Learner age: {age}
User question: {user_input}
{learner_profile}

Document context (from the upload):
{retrieved_context}

Task:
Answer the user's question based ONLY on the document context provided above.
Explain the answer in a way a {age}-year-old can understand.

Rules (must follow):
- Write for EXACTLY age {age}.
- If age <= 7: 1-2 sentences max, everyday words only, no technical terms, simple analogies.
- If 8 <= age <= 10: short paragraphs, define one hard word inline, simple analogies.
- If 11 <= age <= 14: school-level vocabulary, 1-2 technical terms OK if explained briefly.
- If 15 <= age <= 18: high-school depth, abstraction and cause-effect OK, domain terms welcome.
- If age >= 19: peer-level adult language, precise terminology, skip basics unless expertise is Beginner.
- If expertise is Beginner: always define the first key term, assume zero prior knowledge regardless of age.
- If expertise is Advanced and age >= 19: skip basic definitions, use domain-precise language directly.
- Answer ONLY based on the document context provided.
- Do NOT add outside knowledge.
- If the document doesn't contain the answer, say so clearly.
- Keep direct alignment with the latest user topic: "{user_input}".
- Use the learner profile to tune wording and analogy level.
- {personalization_rule}
- Document style detected: {doc_style}.
{_style_rules(doc_style)}
- Start with the direct answer to the selected topic, then add supporting detail.
- If style is research: use 3 short parts in order:
    1) direct answer sentence,
    2) evidence sentence with benchmark/metric from context,
    3) implication sentence.
- Do not ask a question yet.
- Return ONLY the explanation paragraph(s).
- Do not include a heading like "Explanation:".

Explanation:
""".strip()

    # 2) Special branch: simplify the PREVIOUS explanation (same topic, no meta talk)
    elif is_simplify_followup and prev_explanation:
        prompt = f"""
You are AgeXplain, a tutor.

The learner asked for the SAME explanation in SIMPLER words.

Learner age: {age}
User message: {user_input}
{learner_profile}

Core topic anchor (must stay on this topic):
{topic_anchor or "(none)"}

Rewrite this previous explanation in simpler words (same meaning, same topic):
{prev_explanation}

Rules (must follow):
- Keep the SAME topic.
- The core topic is the original user topic above.
- Examples/analogies are helpers only.
- Do NOT turn an example (like baking soda) into the new main topic.
- If you mention an analogy, connect it back to the core topic.
- Write for EXACTLY age {age}.
- For Beginner expertise, define one key term in plain language before details.
- Use shorter, easier words than before.
- Do NOT talk about the user's request itself.
- Do NOT say things like "You want me to explain..." or "Okay!".
- Do NOT introduce a new experiment/topic.
- Use the learner profile to tune wording and analogy level.
- {personalization_rule}
- If the previous explanation used an analogy, keep it short and make sure the main explanation is about the real topic.
- Return ONLY the rewritten explanation text.
- Do not include "Explanation:".

Rewritten explanation:
""".strip()

    # 3) General follow-up branch: more / why / another example / continue
    elif intent == "followup" and (prev_explanation or prev_question or prev_example):
        prompt = f"""
You are AgeXplain, a tutor.

The learner is asking a FOLLOW-UP about the SAME topic as before.

Learner age: {age}
Latest user message: {user_input}
{learner_profile}

Core topic anchor (must stay on this topic):
{topic_anchor or "(none)"}

Previous explanation (same topic):
{prev_explanation or "(none)"}

Previous example (same topic):
{prev_example or "(none)"}

Previous question asked to learner:
{prev_question or "(none)"}

Task:
Respond to the user's follow-up while staying on the SAME topic as the previous explanation.
Do NOT switch topics.

Rules (must follow):
- Write for EXACTLY age {age}.
- If age <= 7: 1-2 sentences max, everyday words only, no technical terms, simple analogies.
- If 8 <= age <= 10: short paragraphs, define one hard word inline, simple analogies.
- If 11 <= age <= 14: school-level vocabulary, 1-2 technical terms OK if explained briefly.
- If 15 <= age <= 18: high-school depth, abstraction and cause-effect OK, domain terms welcome.
- If age >= 19: peer-level adult language, precise terminology, skip basics unless expertise is Beginner.
- If expertise is Beginner: always define the first key term, assume zero prior knowledge regardless of age.
- If expertise is Advanced and age >= 19: skip basic definitions, use domain-precise language directly.
- The core topic is the original user topic above.
- Examples/analogies are helpers only.
- Do NOT turn an example (like baking soda) into the new main topic.
- If you mention an analogy, connect it back to the core topic.
- If user asks for "more" / "why", add one more clear explanation about the same topic.
- If user asks for "another example", keep the same topic and support a new example.
- Do NOT explain the user's phrase itself (e.g., don't start with "Tell me more..." or "You want another example...").
- Do NOT switch to a different topic.
- Use the learner profile to tune wording and analogy level.
- {personalization_rule}
- Start with the real topic first (the user's topic), then use an analogy only if helpful.
- Keep analogies short (1-2 sentences max).
- Prefer facts about the real topic over extending the analogy.
- Return ONLY the explanation paragraph(s).
- Do not include a heading like "Explanation:".

Explanation:
""".strip()

    # 4) New question branch
    else:
        prompt = f"""
You are AgeXplain, a tutor.

Task: Explain the user's question for a child aged {age}.
{learner_profile}

User question/topic:
{user_input}

Rules (must follow):
- Write for EXACTLY age {age}.
- If age <= 7: 1-2 sentences max, everyday words only, no technical terms, simple analogies.
- If 8 <= age <= 10: short paragraphs, define one hard word inline, simple analogies.
- If 11 <= age <= 14: school-level vocabulary, 1-2 technical terms OK if explained briefly.
- If 15 <= age <= 18: high-school depth, abstraction and cause-effect OK, domain terms welcome.
- If age >= 19: peer-level adult language, precise terminology, skip basics unless expertise is Beginner.
- If expertise is Beginner: always define the first key term, assume zero prior knowledge regardless of age.
- If expertise is Advanced and age >= 19: skip basic definitions, use domain-precise language directly.
- Stay on the user's question only (do not change the topic).
- Use the learner profile to tune wording and analogy level.
- {personalization_rule}
- Do not ask a question yet.
- Return ONLY the explanation paragraph(s).
- Do not include a heading like "Explanation:".

Explanation:
""".strip()

    try:
        explanation = llm.invoke(prompt).strip()
        if doc_style == "research" and retrieved_context:
            explanation = _remove_unsupported_references(explanation, retrieved_context)
        explanation = _strip_meta_openers(explanation)
    except Exception as e:
        log_event("simplify_agent.error", level="ERROR", error=str(e))
        explanation = "Sorry, I couldn't generate an explanation right now. Please try again."

    return {
        "simplified_explanation": explanation
    }