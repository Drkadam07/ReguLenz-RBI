import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any

class RegulatoryPDFParser:
    """Parses RBI circulars while stripping headers/footers and tracking sections."""
    def __init__(self, header_margin: float = 45.0, footer_margin: float = 45.0):
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        # Regex capturing RBI sections, paragraph numbers, and clauses
        self.section_pattern = re.compile(
            r"^(Section\s+[IVXLCDM]+|\d+[\.\)]|\([a-z]\)|Part\s+[A-Z])", 
            re.IGNORECASE
        )

    def parse(self, pdf_path: Path) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        extracted_elements = []
        doc_name = pdf_path.stem

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            blocks = page.get_text("blocks")

            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                
                # Filter headers, footers, and non-text blocks
                if block_type != 0:
                    continue
                if y0 < self.header_margin or y1 > (page_height - self.footer_margin):
                    continue

                cleaned_text = re.sub(r"\s+", " ", text).strip()
                if not cleaned_text:
                    continue

                extracted_elements.append({
                    "document_id": doc_name,
                    "page": page_num + 1,
                    "text": cleaned_text
                })
                
        doc.close()
        return extracted_elements