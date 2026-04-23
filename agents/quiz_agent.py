# Create quiz
import json
import re

from configs.models import ExplainState
from configs.config import DIFFICULTY_MAP
from configs.config import llm


def _parse_quiz_response(response: str):
    text = (response or "").strip()
    if not text:
        return None

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start:end + 1])
            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, dict):
                    return first
        except Exception:
            pass

    blocks = re.split(r"\n(?=Q\d+:)", text)
    for block in blocks:
        q_match = re.search(r"Q\d+:\s*(.+)", block)
        if not q_match:
            continue
        options = {}
        for letter in ["A", "B", "C", "D"]:
            opt_match = re.search(rf"{letter}\)\s*(.+)", block)
            if opt_match:
                options[letter] = opt_match.group(1).strip()
        correct_match = re.search(r"Correct:\s*([A-D])", block)
        explanation_match = re.search(r"Explanation:\s*(.+)", block, re.DOTALL)
        if len(options) == 4 and correct_match:
            return {
                "question": q_match.group(1).strip(),
                "options": options,
                "correct": correct_match.group(1).strip(),
                "explanation": (explanation_match.group(1).strip() if explanation_match else ""),
            }

    return None

def quiz_agent(state: ExplainState):
    learner = state.get("learner", {})
    age = learner["age"]
    difficulty = learner.get("difficulty", "medium")
    profession = (learner.get("profession") or "").strip()
    expertise_level = (learner.get("expertise_level") or "").strip()
    area_of_interest = (learner.get("area_of_interest") or "").strip()
    explanation = state.get("simplified_explanation") or ""
    # explanation = state.get("explanation") or state.get("simplified_explanation", "")
    num_of_questions = int(state.get("num_of_questions") or 5)
    
    prompt = f"""
                Generate {num_of_questions} multiple-choice quiz questions based on "{explanation}" for a learner aged {age} and Difficulty: {DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP['medium'])} .

                Learner profile:
                - Profession: {profession or 'Not provided'}
                - Expertise level: {expertise_level or 'Not provided'}
                - Area of interest: {area_of_interest or 'Not provided'}

                Difficulty: {DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP['medium'])}

                Age-appropriate language:
                - If age <= 7: very short questions, everyday words only, no technical terms.
                - If 8 <= age <= 10: simple words, one technical term max.
                - If 11 <= age <= 14: school-level phrasing, conceptual questions OK.
                - If 15 <= age <= 18: domain vocabulary welcome, cause-effect or comparison questions OK.
                - If age >= 19: adult-level language, precise terminology welcome.
                - If expertise is Beginner: prefer recall or definitional questions regardless of age.
                - If expertise is Advanced and age >= 19: prefer analytical or application questions.
                - If area of interest is provided, you may lightly frame wording with it but do not change facts.

                For each question, provide:
                1. The question text
                2. Four answer options (A, B, C, D)
                3. The correct answer (letter only)
                4. A brief explanation

                Format as JSON array:
                [
                {{
                    "question": "Question text here?",
                    "options": {{
                    "A": "Option A text",
                    "B": "Option B text",
                    "C": "Option C text",
                    "D": "Option D text"
                    }},
                    "correct": "A",
                    "explanation": "Brief explanation of the correct answer"
                }}
                ]

            """
    

    response = (llm.invoke(prompt) or "").strip()
    parsed = _parse_quiz_response(response)

    if parsed:
        question = parsed.get("question", "")
        options_dict = parsed.get("options") or {}
        options = [options_dict.get(k, "") for k in ["A", "B", "C", "D"]]
        correct = parsed.get("correct", "A")
        quiz_explanation = parsed.get("explanation") or explanation
    else:
        question = f"Which statement best matches the explanation about this topic for age {age}?"
        options = [
            "It is unrelated to the explanation.",
            "It is one key idea from the explanation.",
            "It says the opposite of the explanation.",
            "It cannot be answered from the explanation.",
        ]
        correct = "B"
        quiz_explanation = "The correct option reflects a core idea from the explanation."

    return {
        "quiz_question": question,
        "quiz_options": options,
        "correct_answer": correct,
        "quiz_explanation": quiz_explanation
    }



def save_answer_agent(state: ExplainState):
    user_answer = state["user_input"].strip().upper()

    return {
        "user_answer": user_answer
    }


def score_agent(state: ExplainState):
    user = (state.get("user_answer") or "").strip()
    correct = (state.get("correct_answer") or "").strip()

    # Quiz path: keep deterministic exact-match scoring.
    if correct:
        score = 1 if user.upper() == correct.upper() else 0
        return {"score": score}

    # Tutor path: no explicit correct option exists, so do semantic correctness check.
    question = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "Question:" in content:
            question = content.split("Question:", 1)[1].strip()
            break

    if not question or not user:
        return {"score": 0}

    learner = state.get("learner", {})
    age = learner["age"]
    expertise_level = (learner.get("expertise_level") or "").strip()
    prompt = f"""
You are grading a learner answer for age {age}.

Question: {question}
Learner answer: {user}
Learner expertise: {expertise_level or 'Not provided'}

Decide if the learner answer is scientifically correct for the question.
Return ONLY one token:
- CORRECT
- WRONG
""".strip()

    try:
        verdict = (llm.invoke(prompt) or "").strip().upper()
    except Exception:
        verdict = "WRONG"

    score = 1 if verdict.startswith("CORRECT") else 0
    return {"score": score}





    # prompt = f"""
    #             Create a short quiz question to test understanding for a learner aged {age} and Difficulty: {DIFFICULTY_MAP[difficulty]}.

    #             Rules:
    #             - One multiple-choice question
    #             - 4 options (A, B, C, D)
    #             - Clearly mark the correct answer

    #             Base it on:
    #             {explanation}

    #             Respond in this format:
    #             Question:
    #             Options:
    #             A)
    #             B)
    #             C)
    #             D)
    #             Correct:
    #         """
