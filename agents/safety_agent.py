from configs.models import ExplainState
from configs.config import llm
import os


def _is_strict_mode_enabled() -> bool:
    raw = (os.getenv("SAFETY_STRICT_MODE", "false") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_unsafe_reason(verdict: str) -> str:
    text = (verdict or "").strip()
    if ":" in text:
        reason = text.split(":", 1)[1].strip()
        if reason:
            return reason
    return "content may not be age-appropriate"


def _looks_like_complexity_flag(verdict: str) -> bool:
    text = (verdict or "").lower()
    markers = [
        "complex",
        "too advanced",
        "difficult",
        "hard to understand",
        "age",
        "terminology",
    ]
    return any(m in text for m in markers)


def safety_agent(state: ExplainState):
    age = state["learner"]["age"]
    strict_mode = _is_strict_mode_enabled()
    explanation = state.get("simplified_explanation") or ""
    example = state.get("example") or ""

    combined = f"""
                Explanation:
                {explanation}

                Example:
                {example}
            """

    prompt = f"""
            Review the content below for a learner aged {age}.

            Check for:
            - Age-inappropriate content
            - Unsafe instructions
            - Overly complex explanations

            If safe, respond with:
            SAFE

            If unsafe, respond with:
            UNSAFE: <brief reason>

            Content:
            {combined}
        """

    verdict = llm.invoke(prompt).strip()

    if verdict.startswith("UNSAFE"):
        if strict_mode:
            reason = _extract_unsafe_reason(verdict)
            strict_message = (
                f"This request is not age-appropriate for age {age}: {reason}. "
                "Please ask for a safer or simpler version."
            )
            return {
                "safety_checked_text": combined,
                "safe_text": strict_message,
                "example": "",
                "thought_question": "",
                "include_examples": False,
                "include_questions": False,
            }

        # Do not break the entire SSE stream: attempt to rewrite if it's complexity-only.
        if _looks_like_complexity_flag(verdict) and explanation:
            simplify_prompt = f"""
Rewrite the explanation below for a {age}-year-old.
- Keep it accurate and concise.
- Use simple words and short sentences.
- Remove advanced jargon.
- Keep it safe and educational.
Return only the rewritten explanation.

Explanation:
{explanation}
""".strip()
            try:
                rewritten = llm.invoke(simplify_prompt).strip()
                if rewritten:
                    return {
                        "safety_checked_text": combined,
                        "safe_text": rewritten,
                    }
            except Exception:
                pass

        safe_fallback = (
            f"I can explain this in a simpler and safer way for age {age}. "
            "Please ask me to break it down step by step, and I will keep it easy to follow."
        )
        return {
            "safety_checked_text": combined,
            "safe_text": safe_fallback,
        }

    return {
        "safety_checked_text": combined,
        "safe_text": explanation,
    }
