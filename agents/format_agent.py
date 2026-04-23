# Final output agent that formats the final response to the user, including the explanation and the question.
from configs.models import ExplainState


def format_agent(state: ExplainState):
    intent = state.get("intent", "new_question")
    rag_sources = state.get("rag_sources")
    include_examples = state.get("include_examples", True)
    include_questions = state.get("include_questions", True)
    
    if intent in ["answer", "quiz"]:
        output = {
            "intent": intent,
            "feedback": state.get("feedback", ""),
        }
        if rag_sources:
            output["sources"] = rag_sources
        return {"final_output": output}
    else:
        output = {
            "intent": intent,
            "explanation": state.get("safe_text") or state.get("simplified_explanation", ""),
            "example": state.get("example", "") if include_examples else "",
            "think_question": state.get("thought_question", "") if include_questions else "",
        }
        if rag_sources:
            output["sources"] = rag_sources
        return {"final_output": output}