from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a specialized Reserve Bank of India (RBI) Regulatory QA Assistant.

Your instructions are absolute:
1. Answer strictly and only based on the retrieved context below.
2. If the context does not supply enough factual evidence to answer, state: "I couldn't find sufficient evidence in the provided circulars."
3. Every single factual statement must be cited directly in-text referencing the source (e.g., [Doc: RBI_2024_KYC_01, Page: 12]).
4. Do NOT extrapolate, hallucinate, or bring in external legal/regulatory knowledge."""

def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += (
            f"\n--- Context Document [{idx}] ---\n"
            f"Document ID: {chunk['document_id']}\n"
            f"Page: {chunk['page']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Text: {chunk['content']}\n"
        )

    return f"Context Evidence:\n{context_str}\n\nQuestion: {query}\n\nPrecise Grounded Answer:"