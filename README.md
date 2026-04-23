# AgeXplain — Streamlit

A self-contained, Streamlit-based version of **AgeXplain**, an age-adaptive multi-agent educational assistant.  
Explains any topic at the learner's level, supports document Q&A with RAG, and includes interactive quizzes — no separate backend server required.

---

## Features

- **Age-adaptive explanations** — vocabulary and depth auto-adjust to the learner's age (5–35)
- **Multi-section responses** — Explanation, Example, Thought Question, and Answer Feedback sections
- **Document Q&A (RAG)** — upload PDF / DOCX / TXT / MD / CSV / HTML; hybrid FAISS + BM25 retrieval
- **Topic-aware quizzes** — free-topic or document-grounded quizzes with per-question feedback
- **Learner profile** — Profession, Expertise Level, and Example Context sliders tune every response
- **Multi-provider LLM support** — Local (Ollama), OpenAI, Gemini, Claude, or GitHub Copilot
- **Topic packs** — curated topic collections for quick exploration
- **Three-panel layout** — chat history sidebar | chat area | settings panel
- **Safety agent** — validates generated content for age-appropriateness

---

## Project Structure

```
streamlit/
├── app.py                  # Streamlit UI entry point
├── graph.py                # LangGraph multi-agent graph definition
├── agents/                 # Individual LangGraph agent nodes
│   ├── intent_agent.py
│   ├── safety_agent.py
│   ├── simplify_agent.py
│   ├── answer_feedback_agent.py
│   ├── format_agent.py
│   ├── quiz_agent.py
│   ├── retrieve_doc_agent.py
│   └── think_question_agent.py
├── configs/
│   ├── config.py           # LLM provider config & context manager
│   ├── model_registry.py   # Model fallback chains
│   └── models.py           # Pydantic state models
├── services/
│   ├── document_service.py # Text extraction & map-reduce summarization
│   ├── rag_service.py      # FAISS + BM25 hybrid RAG
│   ├── model_provider.py   # Multi-provider LLM factory
│   ├── intent_service.py
│   ├── tts_service.py
│   └── json_logger.py
├── db/
│   └── database.py         # SQLite document persistence
├── requirements.txt        # All dependencies (includes streamlit>=1.35.0)
└── .env                    # Environment variables (API keys, settings)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| pip | Latest recommended |
| Ollama *(for local LLM)* | [ollama.ai](https://ollama.ai) with `llama3.1:8b` pulled |

---

## Setup

### 1. Clone and enter the directory

```bash
git clone <your-repo-url>
cd streamlit
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads `sentence-transformers` model weights (~90 MB). This is normal.

### 4. Configure environment variables

Copy `.env.example` to `.env` (or edit `.env` directly):

```bash
cp .env.example .env   # if an example file exists
```

Key variables:

```env
# LLM defaults (overridden at runtime from the UI)
LLM_PROVIDER=local          # local | openai | gemini | claude | copilot
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
GITHUB_TOKEN=               # for Copilot provider

# Safety
SAFETY_STRICT_MODE=false    # true = block unsafe content; false = attempt rewrite

# Database
DB_PATH=../data/agexplain.db
```

### 5. Pull the local model *(optional — only needed for local provider)*

```bash
ollama pull llama3.1:8b
```

---

## Running

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## Usage

### Chat
1. Type a topic or question in the input box at the bottom of the chat area.
2. The assistant returns an **Explanation**, **Example**, and **Question to Think About**.
3. Answer the question to get personalised **Feedback**.

### Document Q&A
1. Upload a file using the **Upload Document** section in the left sidebar.
2. An age-adapted summary is generated automatically, with clickable topic pills.
3. Ask any question about the document — RAG retrieval grounds every answer.

### Quiz
1. After at least one message in a chat, click **🎯 Start Quiz**.
2. Configure topic, difficulty, and number of questions, then click **Generate Quiz**.
3. Answer each question; get immediate correct/incorrect feedback and explanations.
4. Review your full score and per-question breakdown at the end.

### Model Settings (right panel)
- **Provider** — switch between Local, OpenAI, Gemini, Claude, or Copilot at any time.
- **API Key** — enter the key for cloud providers (stored in session only, never persisted).
- **Test Connection** — validates the selected provider before sending messages.

### Profile Settings (right panel)
| Setting | Effect |
|---|---|
| **Explain Age** (slider) | Adjusts vocabulary complexity across all responses |
| **Profession** | Tailors analogies and framing |
| **Expertise Level** | Controls assumed background knowledge |
| **Example Context** | Sets the domain used in examples |

---

## Supported LLM Providers

| Provider | Config Key | Notes |
|---|---|---|
| Local (Ollama) | `local` | No API key; fully offline; requires `ollama` running |
| OpenAI | `openai` | GPT-4o with automatic model fallback |
| Google Gemini | `gemini` | Native Gemini API with fallback chain |
| Anthropic Claude | `claude` | Claude 3.x series with fallback chain |
| GitHub Copilot | `copilot` | Azure-hosted endpoint; requires `GITHUB_TOKEN` |

---

## Dependency Notes

- `click==8.1.8` — pinned to satisfy both `gTTS<8.2` and `typer==0.9.4`
- `typer==0.9.4` — pinned for `click<9.0.0` compatibility
- `torch` and `sentence-transformers` are required for FAISS dense retrieval

---

## Team

Group 12 — Drexel University  
- Deepak Saxena  
- Somasekhar Obulareddy  
- Shadik Khan
