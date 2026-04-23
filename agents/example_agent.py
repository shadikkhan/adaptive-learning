# Create Example
import re

from configs.models import ExplainState
from configs.config import llm


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


def _remove_unsupported_references(text: str, context: str) -> str:
    lowered_context = (context or "").lower()
    flagged_terms = ["openai", "gpt", "elmo", "xlnet", "roberta", "t5", "llama"]
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = []
    for s in sentences:
        ls = s.lower()
        if any(term in ls and term not in lowered_context for term in flagged_terms):
            continue
        kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned or (text or "").strip()



def example_agent(state: ExplainState):
    age = state["learner"]["age"]
    learner = state.get("learner", {})
    profession = (learner.get("profession") or "").strip()
    expertise_level = (learner.get("expertise_level") or "").strip()
    area_of_interest = (learner.get("area_of_interest") or "").strip()
    user_input = (state.get("user_input") or "").strip()
    explanation = (state.get("simplified_explanation") or "").strip()
    retrieved_context = (state.get("retrieved_context") or "").strip()
    doc_style = _infer_doc_style(retrieved_context)

    context_block = ""
    if retrieved_context:
        context_block = f"""
Document context (use for grounding):
{retrieved_context}
"""

    if doc_style == "research":
        prompt = f"""
Create ONE evidence-based example for a learner aged {age}.

Base explanation:
{explanation}
Latest user topic/request:
{user_input}
Learner profile: profession={profession or "Not provided"}, expertise={expertise_level or "Not provided"}, interest={area_of_interest or "Not provided"}
Document context:
{retrieved_context}

Rules (must follow):
- Match age {age} language level EXACTLY.
- Keep direct alignment with latest user topic: "{user_input}".
- Use learner profile tone: beginner explanations should avoid assumed prior knowledge.
- If area of interest is provided, connect the example to that context in one short sentence without changing factual content.
- Use a benchmark-style format: "Before -> After" or "Old score -> New score" when metrics exist.
- **MUST cite at least one specific metric from the Results section of the document context above.**
- Example: If Results show "GLUE improved by 7.7% to 80.5%", use that exact figure, not made-up numbers.
- Do NOT use cartoon stories (no "friend/game/competition" framing).
- Do NOT mention external models/tools unless they appear in the context.
- Return ONLY the example text.
- Keep it short (2-4 sentences).

Example:
""".strip()
    else:
        prompt = f"""
Create ONE real-world example that helps a learner aged {age} understand the explanation.

Base explanation:
{explanation}
Latest user topic/request:
{user_input}
Learner profile: profession={profession or "Not provided"}, expertise={expertise_level or "Not provided"}, interest={area_of_interest or "Not provided"}
Document style: {doc_style}
{context_block}

Rules (must follow):
- Match age {age} language level EXACTLY.
- Keep it on the same topic as the explanation.
- Use learner profile tone: beginner explanations should avoid assumed prior knowledge.
- If area of interest is provided, connect the example to that context in one short sentence without changing factual content.
- If age <= 7: everyday objects only, 1-2 sentences, zero technical terms (no geologist, artifacts, solidify).
- If 8 <= age <= 10: simple words, define one hard word inline if needed.
- If 11 <= age <= 14: school-level vocabulary, 1-2 technical terms OK if briefly explained.
- If 15 <= age <= 18: high-school depth, domain terms welcome, cause-effect framing OK.
- If age >= 19: peer-level adult language, precise terms, skip basics unless expertise is Beginner.
- If expertise is Beginner: use the simplest analogy available regardless of age.
- If expertise is Advanced and age >= 19: skip basic analogies, go straight to domain-level example.
- Keep direct alignment with latest user topic: "{user_input}".
- If document context exists, keep the example consistent with it (do not invent unsupported facts).
- For research-style docs: prefer benchmark/task/measurement style examples over cartoon analogies.
- For research-style docs: do NOT use "friend/game/competition" narrative framing.
- For research-style docs: if a metric/benchmark appears in context, include at least one concrete one.
- For book-style docs: prefer argument/theme examples connected to real decisions.
- No headings like "Example:".
- Return ONLY the example text.
- Keep it short (2-5 sentences).

Example:
""".strip()

    try:
        example = llm.invoke(prompt).strip()
        if doc_style == "research" and retrieved_context:
            example = _remove_unsupported_references(example, retrieved_context)
    except Exception:
        example = "Sorry, I couldn't generate an example right now. Please try again."

    return {
        "example": example
    }