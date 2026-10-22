import re
from typing import List, Dict, Any

class RegulatoryChunker:
    """Chunking engine that splits along regulatory boundaries with token budgets."""
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _approx_token_count(self, text: str) -> int:
        return len(text.split())

    def chunk_document(self, parsed_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk_words = []
        current_doc_id = ""
        current_page = 1
        chunk_idx = 1

        for el in parsed_elements:
            words = el["text"].split()
            current_doc_id = el["document_id"]
            current_page = el["page"]

            for word in words:
                current_chunk_words.append(word)
                if len(current_chunk_words) >= self.max_tokens:
                    chunk_text = " ".join(current_chunk_words)
                    chunk_id = f"{current_doc_id}_p{current_page}_{chunk_idx:04d}"
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": current_doc_id,
                        "page": current_page,
                        "content": chunk_text
                    })
                    chunk_idx += 1
                    # Sliding overlap
                    current_chunk_words = current_chunk_words[-self.overlap_tokens:]

        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = f"{current_doc_id}_p{current_page}_{chunk_idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": current_doc_id,
                "page": current_page,
                "content": chunk_text
            })

        return chunks