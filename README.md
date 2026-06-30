# 🛟 SupportAI — AI Customer Support Bot

A production-style RAG (Retrieval-Augmented Generation) chatbot that answers customer questions from uploaded company documents. Built with **FastAPI + Streamlit + LangChain + ChromaDB + Groq (Llama 3.1)**.

> **100% accuracy** on a 25-question benchmark (LLM-as-judge methodology), including **100% correct refusal rate** on out-of-scope queries.

---

## ✨ Features

- 📄 **Multi-format upload** — PDF, TXT, CSV
- 🔍 **Multi-query retrieval** — LLM rewrites the question into 3 alternative phrasings for better recall
- 📚 **Source citations** — answers include which document and chunk each fact came from (tracked internally)
- 🧠 **Conversation memory** — follow-up questions work naturally ("what about for the Pro plan?")
- 🛡️ **Anti-hallucination** — refuses to answer if the docs don't contain the info
- 📊 **Confidence scoring** — every answer tagged high / medium / low based on retrieval similarity
- 🐳 **Dockerized** — `docker-compose up` and you're live
- ✅ **Evaluation harness** — 25 test questions, keyword + LLM-judge + refusal accuracy

---

## 🏗️ Architecture



---

## 📊 Evaluation Results

Run on a 25-question benchmark using the included TechFlow Solutions knowledge base:

| Metric | Score |
|---|---|
| Answer questions correct (LLM-as-judge) | **22 / 22 (100.0%)** |
| Refusal questions correct | **3 / 3 (100.0%)** |
| Keyword match average | **84.1%** |
| **Overall accuracy** | **25 / 25 (100.0%)** |

Reproduce: `python evaluation/eval.py`


## 📖 Usage

1. Open `http://localhost:8501`
2. Upload PDF/TXT/CSV files via the "Knowledge Base" panel (try `data/sample_docs/techflow_knowledge_base.txt`)
3. Ask questions in the chat. Click suggested prompts on the welcome screen for examples.
4. Click "+ New Chat" to clear memory and start over.

---

## 🧪 Run the Evaluation

```bash
# Backend must be running and docs must be uploaded
cd evaluation
python eval.py
```

Output:

Answer questions: 22

Keyword score (avg): 84.1%

LLM-judge accuracy:  100.0%

Refusal questions: 3

Correctly refused:   3/3 (100.0%)
Overall accuracy: 100.0%  (25/25)



Detailed per-question results are saved to `evaluation/eval_results.json`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Status check, chunk count, active model |
| POST   | `/upload` | Multipart upload of PDF/TXT/CSV files |
| POST   | `/ask` | `{question, session_id}` → answer + sources + confidence |
| POST   | `/clear` | `{session_id}` → wipe that session's memory |
| GET    | `/memory/{session_id}` | Inspect conversation history (debug) |

Interactive Swagger UI at `http://localhost:8000/docs`.

---

## ⚙️ How the RAG Pipeline Works

For each question:

1. **Multi-query expansion** — Llama generates 3 alternative phrasings (plus the original = 4 search queries).
2. **Vector retrieval** — Each query hits ChromaDB; top-K chunks per query are collected.
3. **Dedup + rerank** — Duplicate chunks removed; remaining sorted by L2 distance.
4. **Confidence check** — Best distance maps to high / medium / low confidence.
5. **Prompted generation** — Llama 3.1 receives a strict system prompt: *answer only from context, cite sources, refuse if unsure*.
6. **Citation extraction** — `[Source N]` markers in the response are matched back to retrieved chunks.
7. **Memory update** — The exchange is saved to the session's history (last 6 turns sent on follow-ups).

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.1 8B (via Groq for fast inference) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally on CPU) |
| Vector DB | ChromaDB (persistent local storage) |
| Orchestration | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit (custom premium theme) |
| Containerization | Docker + docker-compose |

---
## 🔧 Future Improvements

- Swap to cloud vector DB (Pinecone / Qdrant) for horizontal scale
- Add a cross-encoder reranker between retrieval and generation
- Persist memory in Redis for multi-replica deploys
- Stream LLM responses for faster perceived latency
- Replace Streamlit frontend with Next.js + Tailwind for full custom UI (Phase 2)

---
