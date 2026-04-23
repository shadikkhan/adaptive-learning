# # # # Provide feedback on quiz performance and guide next steps
# # # from configs.models import ExplainState
# # # from configs.config import llm

# # # def quiz_feedback_agent(state: ExplainState):
# # #     score = state.get("score", 0)
# # #     user_answer = state.get("user_answer", "")
# # #     correct_answer = state.get("correct_answer", "")
# # #     explanation = state.get("simplified_explanation", "")
# # #     thought_question = state.get("thought_question", "")

# # #     if score == 1:
# # #         feedback_text = "Correct! Great job! You understood the concept well."
# # #     else:
# # #         # Provide helpful feedback for incorrect answers
# # #         feedback_text = f"""Not quite right.

# # # Your answer: {user_answer}

# # # Let me help you understand better:
# # # {explanation}

# # # Think about the question again!"""

# # #     # Return feedback as a string
# # #     return {
# # #         "feedback": feedback_text
# # #     }


# # # from configs.models import ExplainState
# # # from configs.config import llm

# # # def extract_last_question(messages):
# # #     """
# # #     Extract the most recent explicit Question: ... from assistant messages.
# # #     """
# # #     for msg in reversed(messages):
# # #         if msg.get("role") != "assistant":
# # #             continue

# # #         content = msg.get("content", "")
# # #         lines = content.splitlines()

# # #         for line in reversed(lines):
# # #             line_clean = line.strip()
# # #             if line_clean.lower().startswith("question:"):
# # #                 return line_clean[len("question:"):].strip()

# # #     return ""

# # # def quiz_feedback_agent(state: ExplainState):
# # #     score = state.get("score", 0)
# # #     user_answer = state.get("user_answer", "")
# # #     explanation = state.get("simplified_explanation") or ""
# # #     topic = state.get("current_topic", "")
# # #     messages = state.get("messages", [])
# # #     question = state.get("thought_question") or extract_last_question(messages)

# # #     if score == 1:
# # #         feedback_text = "Correct! Great job! You understood the concept well."
# # #     else:
# # #         # Always generate explanation if missing
# # #         if not explanation:
# # #             prompt = f"""
# # # You are a tutor giving feedback on a student's answer.

# # # You MUST follow these rules:
# # # 1. Use the question EXACTLY as written.
# # # 2. You MUST talk about the Moon (not just Earth).
# # # 3. You MUST explain how gravity on the Moon is DIFFERENT from Earth.
# # # 4. You MUST clearly say whether objects fall or not on the Moon.

# # # Topic:
# # # {topic}

# # # Question:
# # # {question}

# # # Student answer:
# # # {user_answer}

# # # Explain in simple, child-friendly language why the answer is not correct and what the correct idea is.
# # # """
# # #             explanation = llm.invoke(prompt).strip()

# # #         feedback_text = f"""Not quite right.

# # # Your answer: {user_answer}

# # # Let me help you understand better:
# # # {explanation}

# # # Think about the question again!"""

# # #     return {
# # #         "feedback": feedback_text
# # #     }

# # # from configs.models import ExplainState
# # # from configs.config import llm

# # # def quiz_feedback_agent(state: ExplainState):
# # #     user_answer = (state.get("user_answer") or "").strip()
# # #     age = state["learner"]["age"]  # ✅ DYNAMIC from user's selection
    
# # #     # Extract question DIRECTLY from messages
# # #     question = ""
# # #     for msg in state.get("messages", []):
# # #         if msg.get("role") == "assistant" and "Question:" in msg.get("content", ""):
# # #             question = msg["content"].split("Question:", 1)[1].strip()
# # #             break
    
# # #     print(f"🔍 AGENT DEBUG: question='{question}' | user_answer='{user_answer}' | age={age}")
    
# # #     if not question:
# # #         return {"feedback": "Sorry — I lost the question context. Please try again."}
    
# # #     prompt = f"""
# # # You are a tutor for a {age}-year-old child.
# # # Question: {question}
# # # Child's answer: {user_answer}

# # # Check if the answer is CORRECT or WRONG. Then:

# # # - If CORRECT: "Great job! You got it right because..."
# # # - If WRONG: "Great try! Actually... [simple explanation]"

# # # Use 1-2 sentences. Easy words. Be encouraging always.

# # # Respond with ONLY the explanation paragraph.
# # # """
    
# # #     explanation = llm.invoke(prompt).strip()
    
# # #     if explanation.strip().lower().startswith(('great', 'correct', 'right', 'yes', 'perfect', 'good')):
# # #         feedback_prefix = "🎉 Great job!"
# # #         feedback_suffix = "You really understand this! What else would you like to learn? 😊"
# # #     else:
# # #         feedback_prefix = "Not quite right."
# # #         feedback_suffix = "Think about it and try again! 😊"

# # #     return {
# # #         "feedback": f"""{feedback_prefix}
# # # Your answer: {user_answer.lower()}

# # # Question: {question}

# # # {explanation}

# # # {feedback_suffix}"""
# # #     }

# # from configs.models import ExplainState
# # from configs.config import llm

# # def quiz_feedback_agent(state: ExplainState):
# #     user_answer = (state.get("user_answer") or "").strip()
# #     age = state["learner"]["age"]  # dynamic from user's selection

# #     # ✅ Extract the MOST RECENT question from messages (not the first one)
# #     question = ""
# #     for msg in reversed(state.get("messages", [])):
# #         if msg.get("role") == "assistant":
# #             content = msg.get("content", "")
# #             if "Question:" in content:
# #                 question = content.split("Question:", 1)[1].strip()
# #                 break

# #     print(f"🔍 AGENT DEBUG: question='{question}' | user_answer='{user_answer}' | age={age}")

# #     if not question:
# #         return {"feedback": "Sorry — I lost the question context. Please try again."}

# #     # ✅ Ask LLM for a strict label + short explanation
# # #     prompt = f"""
# # # You are a tutor for a {age}-year-old child.

# # # Question: {question}
# # # Child's answer: {user_answer}

# # # Decide if the child's answer is CORRECT or WRONG.

# # # Reply in EXACTLY this format (2 lines only):
# # # LABEL: CORRECT or WRONG
# # # EXPLANATION: (1-2 short sentences, easy words, encouraging)

# # # Do not add anything else.
# # # """
  
# # #     prompt = f"""
# # # You are a tutor for a {age}-year-old child.

# # # Question: {question}
# # # Child's answer: {user_answer}

# # # Decide if the child's answer is CORRECT or WRONG.

# # # If the answer is CORRECT but missing an important detail (for example: "the Moon's gravity is weaker so things fall slower"),
# # # still mark it as CORRECT and include that missing detail briefly in the explanation.

# # # Reply in EXACTLY this format (2 lines only):
# # # LABEL: CORRECT or WRONG
# # # EXPLANATION: (1-2 short sentences, easy words, encouraging)

# # # Do not add anything else.
# # # """
# #     prompt = f"""
# # You are a careful science tutor for a {age}-year-old child.

# # Question: {question}
# # Child's answer: {user_answer}

# # Decide if the child's answer is CORRECT or WRONG.

# # Rules:
# # - Be scientifically accurate. Do NOT add extra claims that are not guaranteed true.
# # - If CORRECT but missing a key detail, still mark CORRECT and add ONE missing key detail briefly.
# # - If WRONG, gently correct it and give ONE key fact.
# # - Avoid misleading phrases like "just like Earth" unless you also say how it is different.
# # - Use 1–2 short sentences. Easy words. Encouraging tone.

# # Output format (EXACTLY 2 lines):
# # LABEL: CORRECT or WRONG
# # EXPLANATION: <1–2 sentences>
# # """
# #     prompt = f"""
# # You are a careful science tutor for a {age}-year-old child.

# # Question: {question}
# # Child's answer: {user_answer}

# # Decide if the child's answer is CORRECT or WRONG.

# # Rules:
# # - Be scientifically accurate. Do NOT add claims that are not always true.
# # - If CORRECT but missing a key detail, still mark CORRECT and add ONE key detail.
# # - If WRONG, gently correct it and give ONE key fact.
# # - If the question is about gravity on the Moon vs Earth, ALWAYS mention: "the Moon's gravity is weaker, so things fall more slowly."
# # - NEVER say or imply: "just like Earth" (unless you also say "but weaker/slower"), "farther away from you", "nothing to fall onto", or "trampoline".
# # - Use 1–2 short sentences. Easy words. Encouraging tone.

# # Output format (EXACTLY 2 lines):
# # LABEL: CORRECT or WRONG
# # EXPLANATION: <1–2 sentences>
# # """
# #     raw = (llm.invoke(prompt) or "").strip()

# #     # ✅ Parse label safely
# #     label = "WRONG"
# #     explanation = ""

# #     lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
# #     for ln in lines:
# #         if ln.upper().startswith("LABEL:"):
# #             label_val = ln.split(":", 1)[1].strip().upper()
# #             if label_val in ("CORRECT", "WRONG"):
# #                 label = label_val
# #         elif ln.upper().startswith("EXPLANATION:"):
# #             explanation = ln.split(":", 1)[1].strip()

# #     # Fallback: if parsing failed, treat as WRONG but still show the text
# #     if not explanation:
# #         # Use whole response as explanation if it didn't follow format
# #         explanation = raw

# #     if label == "CORRECT":
# #         feedback_prefix = "🎉 Great job!"
# #         feedback_suffix = "You really understand this! What else would you like to learn? 😊"
# #     else:
# #         feedback_prefix = "Not quite right."
# #         feedback_suffix = "Think about it and try again! 😊"

# #     return {
# #         "feedback": f"""{feedback_prefix}
# # Your answer: {user_answer.lower()}

# # Question: {question}

# # {explanation}

# # {feedback_suffix}"""
# #     }

# from configs.models import ExplainState
# from configs.config import llm

# def quiz_feedback_agent(state: ExplainState):
#     user_answer = (state.get("user_answer") or "").strip()
#     age = state["learner"]["age"]  # dynamic from user's selection

#     # ✅ Extract the MOST RECENT question from messages (not the first one)
#     question = ""
#     for msg in reversed(state.get("messages", [])):
#         if msg.get("role") == "assistant":
#             content = msg.get("content", "")
#             if "Question:" in content:
#                 question = content.split("Question:", 1)[1].strip()
#                 break

#     print(f"🔍 AGENT DEBUG: question='{question}' | user_answer='{user_answer}' | age={age}")

#     if not question:
#         return {"feedback": "Sorry — I lost the question context. Please try again."}

#     # ✅ One general prompt that works for ALL topics (no hardcoding)
# #     prompt = f"""
# # You are a careful tutor for a {age}-year-old child.

# # Question: {question}
# # Child's answer: {user_answer}

# # Decide if the child's answer is CORRECT or WRONG.

# # Rules:
# # - Be scientifically accurate. Do NOT add claims that are not always true.
# # - If CORRECT but missing a key detail, still mark CORRECT and add ONE key detail briefly.
# # - If WRONG, gently correct it and give ONE key fact.
# # - If you use words like "same" or "just like", you MUST also say one difference (if relevant).
# # - NEVER invent extra effects that aren't guaranteed true.
# # - Use 1–2 short sentences. Easy words. Encouraging tone.
# # - Age-based style:
# #   - If age <= 7: use very simple words. No numbers, fractions, or formulas. Use phrases like "weaker" and "falls more slowly".
# #   - If 8 <= age <= 10: you may use ONE simple comparison number (example: "about 6 times weaker") but no fractions like "one-sixth".
# #   - If age >= 11: you may include ONE simple fact with a number or fraction (example: "about one-sixth as strong") if it helps.


# # Output format (EXACTLY 2 lines):
# # LABEL: CORRECT or WRONG
# # EXPLANATION: <1–2 sentences>
# # """.strip()
# #     prompt = f"""
# # You are a careful tutor for a {age}-year-old child.

# # Question: {question}
# # Child's answer: {user_answer}

# # Decide if the child's answer is CORRECT or WRONG.

# # Rules:
# # - Be scientifically accurate. Do NOT add claims that are not always true.
# # - If CORRECT but missing a key detail, still mark CORRECT and add ONE key detail briefly.
# # - If WRONG, gently correct it and give ONE key fact.
# # - If you use words like "same" or "just like", you MUST also say one difference (if relevant).
# # - NEVER invent extra effects that aren't guaranteed true.
# # - Use 1–2 short sentences. Easy words. Encouraging tone.
# # - Avoid confusing phrases like "not its name" or sarcasm.
# # - Age-based style (only use numbers if they help the explanation):
# #   - If age <= 7: very simple words. No numbers/fractions/formulas. Use "weaker" and "falls more slowly".
# #   - If 8 <= age <= 10: you may use ONE simple comparison number (e.g., "about 6 times weaker"), but no fractions like "one-sixth".
# #   - If age >= 11: you may include ONE simple number or fraction fact (e.g., "about one-sixth as strong") if helpful.

# # Output format (EXACTLY 2 lines):
# # LABEL: CORRECT or WRONG
# # EXPLANATION: <1–2 sentences>
# # """.strip()
#     prompt = f"""
# You are a careful tutor for a {age}-year-old child.

# Question: {question}
# Child's answer: {user_answer}

# Step 1: Decide if the child's answer is CORRECT or WRONG.
# Step 2: Write a short explanation.

# Rules:
# - Be accurate. Do NOT invent extra facts.
# - Focus on the key idea in the Question. Do not introduce unrelated concepts.
# - Prefer a direct explanation for this question; avoid generic textbook definitions unless needed.
# - If you mention a difference (weaker/stronger, hotter/colder, faster/slower), also state the effect if relevant.
# - EXPLANATION must be EXACTLY 1 or 2 short sentences.
# - EXPLANATION must match the LABEL:
#   - If LABEL is CORRECT: agree with the child and explain why.
#   - If LABEL is WRONG: gently correct the child and state the right idea.
# - If CORRECT but missing one key detail, still mark CORRECT and add ONE key detail briefly.
# - If WRONG, give ONE key fact only.
# - Avoid phrases like "just like" unless you also state one clear difference (if relevant).
# - Use simple, correct grammar and easy words. No sarcasm.

# Age style:
# - If age <= 7: very simple words; no numbers/fractions/formulas.
# - If 8 <= age <= 10: at most ONE simple number comparison (no fractions).
# - If age >= 11: at most ONE simple number OR fraction, only if it helps.

# You MUST start your reply with "LABEL:" on the first line and "EXPLANATION:" on the second line.

# Output format (EXACTLY 2 lines):
# LABEL: CORRECT or WRONG
# EXPLANATION: <1–2 sentences>
# """.strip()

#     raw = (llm.invoke(prompt) or "").strip()

#     # ✅ Parse label + explanation safely
#     label = "WRONG"
#     explanation = ""

#     lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
#     for ln in lines:
#         if ln.upper().startswith("LABEL:"):
#             label_val = ln.split(":", 1)[1].strip().upper()
#             if label_val in ("CORRECT", "WRONG"):
#                 label = label_val
#         elif ln.upper().startswith("EXPLANATION:"):
#             explanation = ln.split(":", 1)[1].strip()

#     # ✅ Smarter fallback if the model didn't follow the format
#     if not explanation:
#         nonempty = [ln.strip() for ln in raw.splitlines() if ln.strip()]
#         # If first line is LABEL, use the next line as explanation
#         if nonempty and nonempty[0].upper().startswith("LABEL:") and len(nonempty) >= 2:
#             explanation = nonempty[1]
#         else:
#             explanation = raw

#     # ✅ Ensure LABEL never appears in the user-facing explanation
#     explanation = "\n".join(
#         ln for ln in explanation.splitlines()
#         if not ln.strip().upper().startswith("LABEL:")
#     ).strip()

#     # ✅ Safety override: if the child clearly answered "no"/"won't", never mark CORRECT
#     ua = user_answer.lower()
#     if any(x in ua for x in ["no", "won't", "will not", "wont", "not fall", "doesn't", "does not"]):
#         label = "WRONG"

#     if label == "CORRECT":
#         feedback_prefix = "🎉 Great job!"
#         feedback_suffix = "You really understand this! What else would you like to learn? 😊"
#     else:
#         feedback_prefix = "Not quite right."
#         feedback_suffix = "Think about it and try again! 😊"

#     return {
#         "feedback": f"""{feedback_prefix}
# Your answer: {user_answer.lower()}

# Question: {question}

# {explanation}

# {feedback_suffix}"""
#     }


from configs.models import ExplainState
from configs.config import llm
from services.json_logger import log_event

def answer_feedback_agent(state: ExplainState):
    user_answer = (state.get("user_answer") or "").strip()
    learner = state.get("learner", {})
    age = learner["age"]
    profession = (learner.get("profession") or "").strip()
    expertise_level = (learner.get("expertise_level") or "").strip()
    area_of_interest = (learner.get("area_of_interest") or "").strip()
    score = state.get("score")

    # ✅ Extract MOST RECENT question from assistant messages
    question = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "Question:" in content:
                question = content.split("Question:", 1)[1].strip()
                break

    log_event(
        "answer_feedback_agent.context",
        age=age,
        profession=profession or "<none>",
        expertise_level=expertise_level or "<none>",
        area_of_interest=area_of_interest or "<none>",
        question=(question or "")[:240],
        user_answer=(user_answer or "")[:240],
    )

    if not question:
        return {"feedback": "Sorry — I lost the question context. Please try again."}

    # ✅ Detect "I don't know" type answers — reveal answer directly instead of marking wrong
    idk_phrases = ["i don't know", "i do not know", "no idea", "i have no idea", "dont know", "don't know", "idk", "not sure", "no clue"]
    if any(p in user_answer.lower() for p in idk_phrases):
        reveal_prompt = f"""
You are a friendly tutor for a learner aged {age}.

Learner profile: profession={profession or "Not provided"}, expertise={expertise_level or "Not provided"}, interest={area_of_interest or "Not provided"}

The learner said they don't know the answer to this question:
{question}

Give the correct answer in 1-2 simple, encouraging sentences.
Do NOT say "Not quite right". Start with "That's okay!"
""".strip()
        try:
            reveal = (llm.invoke(reveal_prompt) or "").strip()
        except Exception:
            reveal = "That's okay! Everyone is still learning."
        return {"feedback": f"That's okay! No worries.\n\n{reveal}"}

    # ✅ Dynamic prompt (works for any topic, age-aware)
    prompt = f"""
You are a careful tutor for a learner aged {age}.

Question: {question}
Learner answer: {user_answer}
Learner profile: profession={profession or "Not provided"}, expertise={expertise_level or "Not provided"}, interest={area_of_interest or "Not provided"}

Step 1: Decide if the learner answer is CORRECT or WRONG.
Step 2: Write a short explanation.

Rules:
- Be accurate. Do NOT invent extra facts.
- Focus on the key idea in the Question. Do not introduce unrelated concepts.
- Prefer a direct explanation for this question; avoid generic textbook definitions unless needed.
- If you mention a difference (weaker/stronger, hotter/colder, faster/slower), also state the effect if relevant.
- EXPLANATION must be EXACTLY 1 or 2 short sentences.
- EXPLANATION must match the LABEL:
  - If LABEL is CORRECT: agree with the child and explain why.
  - If LABEL is WRONG: gently correct the child and state the right idea.
- If CORRECT but missing one key detail, still mark CORRECT and add ONE key detail briefly.
- If WRONG, give ONE key fact only.
- Avoid phrases like "just like" unless you also state one clear difference (if relevant).
- Use simple, correct grammar and easy words. No sarcasm.

Age style:
- If age <= 7: very simple words, no numbers/fractions/formulas, use "weaker"/"slower" style comparisons.
- If 8 <= age <= 10: at most ONE simple number comparison, no fractions.
- If 11 <= age <= 14: school-level words, one technical term OK if explained in the same sentence.
- If 15 <= age <= 18: high-school level, domain terms OK, one number/fraction if it helps.
- If age >= 19: adult precise language, use correct terminology, numbers/fractions fine.
- If expertise is Beginner: keep explanation simple and define any term used, regardless of age.
- If expertise is Advanced and age >= 19: skip basic definitions, be concise and precise.

You MUST start your reply with "LABEL:" on the first line and "EXPLANATION:" on the second line.

Output format (EXACTLY 2 lines):
LABEL: CORRECT or WRONG
EXPLANATION: <1–2 sentences>
""".strip()

    raw = (llm.invoke(prompt) or "").strip()

    # ✅ Parse label + explanation
    label = "WRONG"
    explanation = ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        if ln.upper().startswith("LABEL:"):
            label_val = ln.split(":", 1)[1].strip().upper()
            if label_val in ("CORRECT", "WRONG"):
                label = label_val
        elif ln.upper().startswith("EXPLANATION:"):
            explanation = ln.split(":", 1)[1].strip()

    # ✅ If model didn't follow format, infer label from beginning (rare fallback)
    if not any(l.upper().startswith("LABEL:") for l in lines):
        first = (raw.strip().lower()[:60])
        if first.startswith(("yes", "correct", "that's right", "great job", "good job")):
            label = "CORRECT"

    # ✅ Fallback if EXPLANATION missing
    if not explanation:
        nonempty = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if nonempty and nonempty[0].upper().startswith("LABEL:") and len(nonempty) >= 2:
            second = nonempty[1]
            if second.upper().startswith("EXPLANATION:"):
                explanation = second.split(":", 1)[1].strip()
            else:
                explanation = second
        else:
            explanation = raw

    # ✅ Never show LABEL/EXPLANATION tags to the user
    explanation = "\n".join(
        ln for ln in explanation.splitlines()
        if not ln.strip().upper().startswith(("LABEL:", "EXPLANATION:"))
    ).strip()

    # Keep feedback consistent with scorer output when available.
    if score in (0, 1):
        label = "CORRECT" if score == 1 else "WRONG"

    if label == "CORRECT":
        feedback_prefix = "🎉 Great job!"
        feedback_suffix = "You really understand this! What else would you like to learn? 😊"
    else:
        feedback_prefix = "Not quite right."
        feedback_suffix = "Think about it and try again! 😊"

    return {
        "feedback": f"""{feedback_prefix}
Your answer: {user_answer.lower()}

Question: {question}

{explanation}

{feedback_suffix}"""
    }