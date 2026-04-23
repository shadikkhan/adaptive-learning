# Lets the learner respond and close the loop later
from configs.models import ExplainState
from configs.config import llm

def feedback_agent(state: ExplainState):
    feedback = state.get("feedback")

    if not feedback:
        return {}

    prompt = f"""
The learner gave the following feedback:
{feedback}

Briefly assess:
- Did they understand?
- Should the difficulty be adjusted next time?

Respond in 2–3 sentences.
"""

    analysis = llm.invoke(prompt).strip()

    return {
        "feedback": analysis
    }
