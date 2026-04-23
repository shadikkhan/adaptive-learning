from langgraph.graph import StateGraph
from agents.example_agent import example_agent
from agents.format_agent import format_agent
from agents.intent_agent import infer_intent
from agents.retrieve_doc_agent import retrieve_document
from configs.models import ExplainState
from agents.quiz_agent import quiz_agent, score_agent, save_answer_agent
from agents.answer_feedback_agent import answer_feedback_agent
from agents.safety_agent import safety_agent
from agents.simplify_agent import simplify_agent
from agents.think_question_agent import think_question_agent

# -----------------------------
# Graph Initialization
# -----------------------------
graph = StateGraph(ExplainState)

# -----------------------------
# Node Registration
# -----------------------------

# Intent
graph.add_node("infer_intent", infer_intent)

# Document retrieval
graph.add_node("retrieve_doc", retrieve_document)

# Teaching agents
graph.add_node("simplify", simplify_agent)
graph.add_node("generate_example", example_agent)
graph.add_node("think", think_question_agent)

# Quiz agents
graph.add_node("quiz", quiz_agent)
graph.add_node("save_answer", save_answer_agent)
graph.add_node("evaluate_answer", score_agent)
graph.add_node("answer_feedback", answer_feedback_agent)

# Shared agents
graph.add_node("safety", safety_agent)
graph.add_node("format", format_agent)


# -----------------------------
# Entry Point
# -----------------------------
graph.set_entry_point("infer_intent")

# -----------------------------
# Intent-Based Routing
# -----------------------------
def _route_after_intent(state):
    """
    When a document is loaded, ALWAYS retrieve context first regardless of intent.
    This enables hybrid RAG-grounded tutor mode for every teaching interaction.
    Quiz and answer flows are unaffected.
    """
    intent = state.get("intent", "new_question")
    if intent in ["quiz", "answer"]:
        return intent
    if state.get("doc_id"):
        return "retrieve_first"   # always go through RAG when doc is loaded
    return intent

graph.add_conditional_edges(
    "infer_intent",
    _route_after_intent,
    {
        "quiz": "quiz",
        "answer": "save_answer",
        "retrieve_first": "retrieve_doc",     # doc loaded → always retrieve
        "document_question": "retrieve_doc",  # fallback (should not occur after above)
        "new_question": "simplify",
        "followup": "simplify",
    }
)

# -----------------------------
# Teaching Flow
# Document retrieval feeds into simplify.
graph.add_edge("retrieve_doc", "simplify")

def _route_after_simplify(state):
    intent = state.get("intent")
    if intent not in ["document_question", "new_question", "followup"]:
        return "skip"
    if state.get("include_examples", True):
        return "with_example"
    return "without_example"

# Document QA now follows tutor-style explanation + example before safety.
graph.add_conditional_edges(
    "simplify",
    _route_after_simplify,
    {
        "with_example": "generate_example",
        "without_example": "safety",
        "skip": "safety",
    }
)
graph.add_edge("generate_example", "safety")

# -----------------------------
# Quiz Generation Flow
# -----------------------------
graph.add_edge("quiz", "safety")

# -----------------------------
# Answer Evaluation Flow
# -----------------------------
graph.add_edge("save_answer", "evaluate_answer")
graph.add_edge("evaluate_answer", "answer_feedback")
graph.add_edge("answer_feedback", "safety")

# -----------------------------
# Safety to Next Step Routing
# -----------------------------
# After safety check, route based on intent
# - Teaching flows (new_question, followup) go through thought question
# - Document QA and Quiz/Answer flows go directly to format
graph.add_conditional_edges(
    "safety",
    lambda state: "think" if state.get("intent") in ["new_question", "followup", "document_question"] and state.get("include_questions", True) else "format",
    {
        "think": "think",
        "format": "format",
    }
)

# -----------------------------
# Final Formatting
# -----------------------------
graph.add_edge("think", "format")

# -----------------------------
# Finish Point
# -----------------------------
graph.set_finish_point("format")

# -----------------------------
# Compile Graph
# -----------------------------
learning_graph = graph.compile()
