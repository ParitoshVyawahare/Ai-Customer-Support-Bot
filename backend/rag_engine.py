"""RAG engine: embeddings, vector store, multi-query retrieval, memory, citations."""
from typing import List, Dict, Tuple
import re

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from config import settings


class RAGEngine:
    """Encapsulates the entire RAG pipeline.

    Pipeline:
      question -> multi_query (LLM rewrites) -> vector search (each query)
              -> dedup + rerank by score -> build context with citation markers
              -> LLM answer with strict 'only-from-context' instruction
              -> confidence score based on best similarity
    """

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment."
            )
        if not settings.COHERE_API_KEY:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Add it to your .env file or environment."
            )

        self.embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.EMBED_MODEL,
        )
        self.vectorstore = Chroma(
            collection_name="support_docs",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PATH,
        )
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=1024,
        )
        # session_id -> list of {role, content}
        self.memory: Dict[str, List[Dict[str, str]]] = {}

    # ---------- storage ----------

    def add_documents(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        self.vectorstore.add_documents(docs)
        return len(docs)

    def get_doc_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    # ---------- multi-query ----------

    def _generate_multi_queries(self, question: str) -> List[str]:
        """Use the LLM to generate N alternative phrasings of the question."""
        n = settings.MULTI_QUERY_N
        prompt = (
            f"You are a search query rewriter. Given a user question, write {n} "
            "different ways to phrase it for a vector search engine. Use synonyms, "
            "rephrase, and vary specificity. Output ONLY the queries, one per line, "
            "no numbering, no commentary.\n\n"
            f"User question: {question}"
        )
        try:
            resp = self.llm.invoke(prompt).content
        except Exception:
            return [question]

        lines = [re.sub(r"^[\d\.\-\)\s]+", "", ln).strip() for ln in resp.split("\n")]
        queries = [ln for ln in lines if ln and len(ln) > 3][:n]
        # Always include the original — it's the ground truth phrasing
        if question not in queries:
            queries.append(question)
        return queries

    # ---------- retrieval ----------

    def _retrieve(self, question: str) -> List[Tuple[Document, float]]:
        """Multi-query retrieval with dedup. Returns (doc, score) pairs sorted best-first."""
        queries = self._generate_multi_queries(question)
        seen = set()
        results: List[Tuple[Document, float]] = []
        for q in queries:
            try:
                hits = self.vectorstore.similarity_search_with_score(q, k=settings.TOP_K)
            except Exception:
                continue
            for doc, score in hits:
                key = (doc.metadata.get("source"), doc.metadata.get("chunk_id"))
                if key in seen:
                    continue
                seen.add(key)
                results.append((doc, float(score)))
        # Chroma default = L2 distance, lower is better
        results.sort(key=lambda x: x[1])
        return results[: settings.TOP_K + 2]  # keep a few extras for context

    # ---------- main query ----------

    def query(self, question: str, session_id: str = "default") -> Dict:
        retrieved = self._retrieve(question)

        if not retrieved:
            return {
                "answer": (
                    "I don't have enough information to answer that based on the "
                    "available documents. Please upload relevant documents first."
                ),
                "sources": [],
                "confidence": "low",
            }

        best_score = retrieved[0][1]
        if best_score < settings.HIGH_CONF_THRESHOLD:
            confidence = "high"
        elif best_score < settings.MEDIUM_CONF_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"

        # Build numbered context for citations
        context_blocks = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            src = doc.metadata.get("source", "unknown")
            cid = doc.metadata.get("chunk_id", "?")
            context_blocks.append(
                f"[Source {i} | file: {src} | chunk: {cid}]\n{doc.page_content}"
            )
        context = "\n\n".join(context_blocks)

        # Build conversation history (last 6 turns max)
        history = self.memory.get(session_id, [])[-6:]
        history_str = ""
        if history:
            history_str = "\n".join(
                f"{m['role'].title()}: {m['content']}" for m in history
            )

        system_prompt = (
            "You are a helpful customer support assistant. You MUST follow these rules:\n"
            "1. Answer ONLY using information found in the provided Context.\n"
            "2. Cite sources inline using the format [Source N] matching the numbered "
            "blocks in Context. Multiple sources allowed: [Source 1][Source 3].\n"
            "3. If the Context does NOT contain enough information to answer, respond "
            "EXACTLY with: \"I don't have enough information to answer that based on "
            "the available documents.\" Do not guess. Do not use outside knowledge.\n"
            "4. Use the conversation history to understand follow-up questions, but "
            "still only answer from the Context.\n"
            "5. Be concise. Avoid filler. No preambles like 'Based on the context...'."
        )

        user_prompt = f"Context:\n{context}\n\n"
        if history_str:
            user_prompt += f"Conversation history:\n{history_str}\n\n"
        user_prompt += f"Question: {question}\n\nAnswer:"

        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        ).content.strip()

        # Update memory
        self.memory.setdefault(session_id, []).append(
            {"role": "user", "content": question}
        )
        self.memory[session_id].append({"role": "assistant", "content": response})

        # Identify which sources the LLM actually cited (for cleaner UI)
        cited_ids = set(int(m) for m in re.findall(r"\[Source (\d+)\]", response))

        sources = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            sources.append(
                {
                    "id": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "score": round(score, 4),
                    "cited": i in cited_ids,
                    "snippet": (
                        doc.page_content[:240]
                        + ("..." if len(doc.page_content) > 240 else "")
                    ),
                }
            )

        # If the model refused, force confidence low
        if "don't have enough information" in response.lower():
            confidence = "low"

        return {
            "answer": response,
            "sources": sources,
            "confidence": confidence,
        }

    # ---------- memory ----------

    def clear_memory(self, session_id: str = "default") -> None:
        self.memory.pop(session_id, None)

    def get_memory(self, session_id: str = "default") -> List[Dict[str, str]]:
        return self.memory.get(session_id, [])