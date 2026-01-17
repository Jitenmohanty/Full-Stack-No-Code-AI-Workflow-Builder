import fitz  # PyMuPDF
from typing import List, Dict
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": chunk,
                "start": start,
                "end": min(end, text_length)
            })
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_bytes: bytes, filename: str) -> Dict:
        """Process a document and return chunks"""
        text = self.extract_text_from_pdf(file_bytes)
        chunks = self.chunk_text(text)
        
        return {
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }