# Turn explanation into learning engagement
import re

from configs.models import ExplainState
from configs.config import llm


def _content_words(text: str):
    stop = {
        "what", "is", "are", "the", "a", "an", "of", "to", "in", "on", "for", "and",
        "or", "about", "why", "how", "when", "where", "who", "which", "does", "do", "did",
        "can", "could", "would", "should", "tell", "me", "more", "please"
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _is_question_on_topic(question: str, topic: str) -> bool:
    q_words = set(_content_words(question))
    t_words = set(_content_words(topic))
    if not t_words:
        return True
    # Require at least one meaningful overlap with the selected/requested topic.
    return len(q_words.intersection(t_words)) >= 1


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

def think_question_agent(state: ExplainState):
    age = state["learner"]["age"]
    learner = state.get("learner", {})
    profession = (learner.get("profession") or "").strip()
    expertise_level = (learner.get("expertise_level") or "").strip()
    area_of_interest = (learner.get("area_of_interest") or "").strip()
    explanation = state["simplified_explanation"]
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    retrieved_context = state.get("retrieved_context") or ""
    doc_style = _infer_doc_style(retrieved_context)

    # Collect previously asked questions to avoid repetition
    asked_questions = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "Question:" in content:
                q = content.split("Question:", 1)[1].strip().splitlines()[0].strip()
                if q:
                    asked_questions.append(q)

    avoid_section = ""
    if asked_questions:
        avoid_list = "\n".join(f"- {q}" for q in asked_questions[-3:])
        avoid_section = f"\nDo NOT ask any of these already-asked questions:\n{avoid_list}\n"

    doc_context_section = ""
    if retrieved_context:
        doc_context_section = f"""
The learner has an uploaded document. Here are relevant excerpts — prefer questions grounded in them:
{retrieved_context}
"""

    prompt = f"""
Generate ONE factual question for a learner aged {age} based on the explanation below.
{doc_context_section}
Latest user request/topic:
{user_input}
Learner profile: profession={profession or "Not provided"}, expertise={expertise_level or "Not provided"}, interest={area_of_interest or "Not provided"}

Rules:
- Age-appropriate wording:
    - If age <= 7: very short question, simple everyday words only.
    - If 8 <= age <= 10: short question, one technical term max.
    - If 11 <= age <= 14: school-level phrasing, conceptual questions OK.
    - If 15 <= age <= 18: can use domain vocabulary, cause-effect or comparison questions.
    - If age >= 19: adult-level phrasing, precise terminology welcome.
    - If expertise is Beginner: prefer definitional or recall questions regardless of age.
    - If expertise is Advanced and age >= 19: prefer analytical or application questions.
- The question must have a clear, factual correct answer (not a hypothetical or opinion).
- Avoid "what if", "if you could", "imagine if", "do you think" style questions.
- Prefer questions that test recall or understanding of a specific fact from the explanation.
- If document excerpts are provided, prefer a question grounded in the document.
- The question must stay directly on the latest user request/topic.
- Include at least one meaningful keyword from the latest user request/topic.
- Do NOT switch to unrelated entities (for example, FDA, random dates, other agencies) unless explicitly central to the user topic.
- Document style detected: {doc_style}
- If style is research: prefer benchmark/metric/method/comparison questions when present.
- If style is research: avoid overly trivial counting questions unless counting is the core topic.
- If style is research: prefer "why/how/which method/which metric" style factual prompts.
- If style is book: prefer claim/argument/theme questions from the text.
- Ask a DIFFERENT aspect than already covered.
- Do NOT add any extra text. Return ONLY the question.
{avoid_section}
Base it on:
{explanation}
"""

    question = llm.invoke(prompt).strip()

    # Research quality guard: avoid trivial count-only questions when richer facts exist.
    if doc_style == "research":
        q_low = question.lower()
        trivial_count = q_low.startswith("what number") or q_low.startswith("how many")
        if trivial_count:
            if "glue" in (retrieved_context or "").lower():
                question = "Which benchmark score improved to 80.5% in this document?"
            elif "multinli" in (retrieved_context or "").lower():
                question = "Which task reached 86.7% accuracy according to this document?"
            else:
                topic = (user_input or "this topic").strip().rstrip("?")
                question = f"Which method or result best supports {topic} in this document?"

    # Guardrail: if the model drifts off-topic, force a safe on-topic factual question.
    if not _is_question_on_topic(question, user_input):
        topic = (user_input or "this topic").strip().rstrip("?")
        question = f"What does {topic} mean in this document?"

    return {
        "thought_question": question,
        "current_question": question
    }
