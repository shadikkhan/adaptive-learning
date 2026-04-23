"""
RAG Service: Retrieval-Augmented Generation for document Q&A.

Provides:
- Chunking of document text
- Embedding generation via sentence-transformers
- FAISS vector similarity search
- Citation generation with source references
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
from services.json_logger import log_event

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    raise RuntimeError(
        "RAG requires: pip install sentence-transformers faiss-cpu numpy"
    )

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log_event(
        "rag_service.bm25_unavailable",
        level="WARNING",
        message="rank_bm25 not installed. Using dense-only retrieval.",
    )


class RAGIndex:
    """In-memory RAG index for a single document (FAISS dense + BM25 sparse hybrid)."""

    def __init__(self, doc_id: str, embedder: SentenceTransformer):
        self.doc_id = doc_id
        self.embedder = embedder
        self.chunks: List[str] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.bm25 = None  # BM25Okapi instance if available

    def add_chunks(self, chunks: List[str]) -> None:
        """Add text chunks and build FAISS dense index and BM25 sparse index."""
        if not chunks:
            return

        self.chunks = chunks

        # --- Dense index (FAISS) ---
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

        # --- Sparse index (BM25) ---
        if _BM25_AVAILABLE:
            tokenized = [c.lower().split() for c in chunks]
            self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Hybrid retrieval: combine FAISS dense scores and BM25 sparse scores."""
        if not self.index or not self.chunks:
            return []

        n = len(self.chunks)
        actual_k = min(top_k, n)

        # --- Dense scores (FAISS) ---
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype("float32"), actual_k)

        dense_scores = np.zeros(n)
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < n:
                dense_scores[idx] = 1.0 / (1.0 + float(dist))

        # Normalize dense scores to [0, 1]
        dense_max = float(dense_scores.max()) if dense_scores.max() > 0 else 1.0
        dense_norm = dense_scores / dense_max

        # --- Sparse scores (BM25) ---
        if self.bm25 is not None:
            bm25_raw = np.array(self.bm25.get_scores(query.lower().split()), dtype=float)
            bm25_max = float(bm25_raw.max()) if bm25_raw.max() > 0 else 1.0
            bm25_norm = bm25_raw / bm25_max
        else:
            bm25_norm = np.zeros(n)

        # --- Hybrid combination (60% dense, 40% sparse) ---
        alpha = 0.6
        combined = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        top_indices = np.argsort(combined)[::-1][:actual_k]
        return [(self.chunks[int(i)], float(combined[i])) for i in top_indices if combined[i] > 0]


class RAGService:
    """RAG service manager for multiple documents."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.indices: Dict[str, RAGIndex] = {}

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = _split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def index_document(self, doc_id: str, text: str) -> int:
        """Index a document for RAG retrieval."""
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        rag_index = RAGIndex(doc_id, self.embedder)
        rag_index.add_chunks(chunks)
        self.indices[doc_id] = rag_index

        return len(chunks)

    def retrieve(self, doc_id: str, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks for a query."""
        if doc_id not in self.indices:
            return []

        return self.indices[doc_id].retrieve(query, top_k)

    def answer_with_rag(
        self,
        doc_id: str,
        query: str,
        llm,
        top_k: int = 3,
        age: int = 10,
    ) -> Tuple[str, List[str]]:
        """Generate an answer grounded in retrieved document chunks.

        Returns (answer, sources).
        """
        retrieved = self.retrieve(doc_id, query, top_k)
        if not retrieved:
            return "No relevant information found in the document.", []

        context = "\n\n".join([f"[{i + 1}] {chunk}" for i, (chunk, _) in enumerate(retrieved)])

        prompt = f"""
You are AgeXplain, an age-adaptive educational tutor.

Answer this question based ONLY on the provided document chunks.
Write for age {age}.

User question: {query}

Document chunks:
{context}

Answer rules:
- Use only information from the chunks above.
- If the answer is not in the chunks, say "Not found in the document."
- Keep the answer age-appropriate and concise.
- Cite which chunk(s) you used by number [1], [2], etc.
""".strip()

        answer = llm.invoke(prompt).strip()
        sources = [chunk for chunk, _ in retrieved]

        return answer, sources


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations."""
    text = (text or "").strip()
    if not text:
        return []

    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "|||", text)
    text = re.sub(r"(?<=[.!?])\s+", "|||", text)

    sentences = text.split("|||")
    clean = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

    return clean


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create global RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def reload_indices_from_db() -> int:
    """Re-index all persisted documents from SQLite into the RAG service.

    Called at server startup so documents survive a restart.
    Returns the number of documents re-indexed.
    """
    try:
        from db.database import get_all_documents
        docs = get_all_documents()
        service = get_rag_service()
        count = 0
        for doc_id, doc in docs.items():
            text = doc.get("text", "")
            if text:
                service.index_document(doc_id, text)
                count += 1
        log_event("rag_service.reload_indices", count=count)
        return count
    except Exception as e:
        log_event("rag_service.reload_indices_error", level="WARNING", error=str(e))
        return 0
