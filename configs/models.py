from typing import Literal, TypedDict, List, Optional
    

class LearnerProfile(TypedDict):
    age: int
    difficulty: Literal["easy", "medium", "hard"]
    profession: Optional[str]
    expertise_level: Optional[str]
    area_of_interest: Optional[str]
    
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class ExplainState(TypedDict):
    # ---- Input & Conversation ----
    # topic: str
    user_input: str
    messages: List[Message]

    # ---- Learner Context ----
    learner: LearnerProfile

    # ---- User Output Preferences ----
    include_examples: Optional[bool]
    include_questions: Optional[bool]

    # ---- Request Metadata ----
    force_new_topic: Optional[bool]
    model_provider: Optional[str]
    model_name: Optional[str]

    # ---- Document Context (for RAG) ----
    doc_id: Optional[str]
    retrieved_context: Optional[str]
    rag_sources: Optional[List[str]]

    # ---- Intent ----
    intent: Literal["new_question", "answer", "followup", "quiz", "document_question"]

    # ---- Generated Pedagogical Artifacts ----
    simplified_explanation: Optional[str]
    example: Optional[str]
    safe_text: Optional[str]
    safety_checked_text: Optional[str]
    thought_question: Optional[str]
    
    # ---- Quiz Artifacts ----
    num_of_questions: Optional[int]
    quiz_question: Optional[str]
    quiz_options: Optional[List[str]]
    correct_answer: Optional[str]
    user_answer: Optional[str]
    score: Optional[int]

    # ---- Output ----
    final_output: Optional[dict]

    # ---- Feedback Loop (future) ----
    feedback: Optional[str]


