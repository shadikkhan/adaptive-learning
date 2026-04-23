## write code to run intent_agent.py and call the function to test it with different inputs. You can create a simple test function that simulates different conversation states and prints the inferred intent.
from agents import intent_agent
from configs.models import ExplainState, LearnerProfile, Message

def test_infer_intent():
    learner = LearnerProfile(age=25)
    messages = [
        Message(role="user", content="What is AI?"),
        Message(role="assistant", content="AI stands for Artificial Intelligence. Do you want an example?")
    ]
    state = ExplainState(
        topic="AI",
        user_input="Yes, give me an example.",
        messages=messages,
        learner=learner,
        intent="new_question",
        simplified_explanation=None,
        example=None,
        safety_checked_text=None,
        thought_question=None,
        final_output=None,
        feedback=None
    )

    intent = intent_agent.infer_intent(state)
    print(f"Inferred intent: {intent}")

if __name__ == "__main__":
    test_infer_intent()
