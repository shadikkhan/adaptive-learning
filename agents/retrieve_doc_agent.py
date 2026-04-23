"""
Document Retrieval Agent using RAG.

Retrieves relevant chunks from uploaded documents for grounded Q&A.
"""

from configs.models import ExplainState
from services.rag_service import get_rag_service
from services.json_logger import log_event

log_event("retrieve_doc_agent.loaded")


def retrieve_document(state: ExplainState):
    """Retrieve relevant document chunks for answering user question."""
    doc_id = state.get("doc_id")
    user_input = (state.get("user_input") or "").strip()
    
    if not doc_id or not user_input:
        return {
            "retrieved_context": None,
            "rag_sources": None,
        }

    try:
        rag_service = get_rag_service()
        results = rag_service.retrieve(doc_id, user_input, top_k=3)
        
        if not results:
            log_event("retrieve_doc_agent.empty", doc_id=doc_id, query=user_input[:200])
            return {
                "retrieved_context": None,
                "rag_sources": None,
            }
        
        # Build context from retrieved chunks
        retrieved_chunks = [chunk for chunk, _ in results]
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        log_event("retrieve_doc_agent.success", doc_id=doc_id, chunks=len(retrieved_chunks), query=user_input[:200])
        
        return {
            "retrieved_context": context,
            "rag_sources": retrieved_chunks,
        }
    except Exception as e:
        log_event("retrieve_doc_agent.error", level="ERROR", doc_id=doc_id, error=str(e))
        return {
            "retrieved_context": None,
            "rag_sources": None,
        }
