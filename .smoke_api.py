import io
import json
import traceback
from fastapi.testclient import TestClient
import main
import api.routes as routes


class FakeLLM:
    def invoke(self, prompt: str):
        p = (prompt or "").lower()
        if "reply with exactly ok" in p or "validate" in p:
            return "OK"
        if "quiz" in p or "questions" in p or "json" in p:
            payload = {
                "questions": [
                    {"question": "Q1?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "A", "explanation": "e1"},
                    {"question": "Q2?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "B", "explanation": "e2"},
                    {"question": "Q3?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "C", "explanation": "e3"},
                    {"question": "Q4?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "D", "explanation": "e4"},
                    {"question": "Q5?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "A", "explanation": "e5"},
                ]
            }
            return json.dumps(payload)
        return "Short generated response."


class FakeIndex:
    def __init__(self, text=""):
        self.chunks = [text[:300] or "chunk"]

    def retrieve(self, query, top_k=3):
        return [(c, 1.0) for c in self.chunks[:top_k]]


class FakeRag:
    def __init__(self):
        self.indices = {}

    def index_document(self, doc_id, text):
        self.indices[doc_id] = FakeIndex(text)


class FakeGraph:
    async def ainvoke(self, initial_state):
        return {
            "final_output": {
                "intent": initial_state.get("intent") or "document_question",
                "explanation": "Fake explanation",
                "example": "Fake example",
                "think_question": "Fake question",
            },
            "rag_sources": ["fake source"],
        }

    async def astream(self, initial_state):
        yield {"infer_intent": {"intent": "new_question"}}
        yield {"simplify": {"intent": "new_question", "simplified_explanation": "Fake explanation"}}
        yield {"generate_example": {"intent": "new_question", "example": "Fake example"}}
        yield {"think": {"intent": "new_question", "thought_question": "Fake question"}}


routes._resolve_runtime_llm = lambda cfg: FakeLLM()
routes.get_rag_service = lambda: FakeRag()
routes.learning_graph = FakeGraph()
routes.synthesize_tts_mp3 = lambda text: None

client = TestClient(main.app)
passes = 0
fails = 0
state = {}


def check(name, fn):
    global passes, fails
    try:
        fn()
        print(f"PASS {name}")
        passes += 1
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        traceback.print_exc()
        fails += 1


check("GET /topics", lambda: (lambda r: (r.raise_for_status(), isinstance(r.json(), dict)))(client.get("/topics")))

check(
    "POST /models/validate",
    lambda: (lambda r: (r.raise_for_status(), isinstance(r.json(), dict)))(
        client.post("/models/validate", json={"llm_config": {"provider": "local"}, "prompt": "Reply with exactly OK"})
    ),
)

check(
    "POST /quiz/generate",
    lambda: (lambda r: (r.raise_for_status(), len(r.json().get("questions") or []) > 0))(
        client.post("/quiz/generate", json={"topic": "photosynthesis", "age": 10, "num_questions": 5, "difficulty": "medium", "llm_config": {"provider": "local"}})
    ),
)


def do_upload():
    sample_text = (
        b"Plants convert light energy into chemical energy through photosynthesis. "
        b"In this process, chlorophyll in leaves absorbs sunlight and helps transform "
        b"water and carbon dioxide into glucose and oxygen."
    )
    files = {"file": ("sample.txt", io.BytesIO(sample_text), "text/plain")}
    data = {
        "age": "10",
        "profession": "Student",
        "expertise_level": "Beginner",
        "area_of_interest": "General",
        "character": "Friendly Teacher",
        "llm_provider": "local",
    }
    r = client.post("/documents/upload", files=files, data=data)
    r.raise_for_status()
    j = r.json()
    assert j.get("doc_id")
    state["doc_id"] = j["doc_id"]


check("POST /documents/upload", do_upload)

check(
    "POST /documents/ask",
    lambda: (lambda r: (r.raise_for_status(), "answer" in r.json()))(
        client.post(
            "/documents/ask",
            json={
                "doc_id": state.get("doc_id"),
                "question": "What is this about?",
                "age": 10,
                "profession": "Student",
                "expertise_level": "Beginner",
                "area_of_interest": "General",
                "character": "Friendly Teacher",
                "include_examples": True,
                "include_questions": True,
                "llm_config": {"provider": "local"},
            },
        )
    ),
)


def do_stream():
    r = client.post(
        "/explain/stream",
        json={
            "topic": "What is gravity?",
            "age": 10,
            "context": "",
            "profession": "Student",
            "expertise_level": "Beginner",
            "area_of_interest": "General",
            "character": "Friendly Teacher",
            "include_examples": True,
            "include_questions": True,
            "force_new_topic": False,
            "llm_config": {"provider": "local"},
        },
    )
    r.raise_for_status()
    txt = r.text
    assert '"type": "done"' in txt or '"type":"done"' in txt


check("POST /explain/stream", do_stream)

print(f"SUMMARY PASS={passes} FAIL={fails}")
if fails:
    raise SystemExit(1)
