"""
PDF Analyzer Agent - Process and analyze PDF documents
"""
from typing import Dict, Any, List, Optional
from .llm import LLMClient
from pathlib import Path
import fitz  # PyMuPDF

class PDFAgent:
    """Analyze PDF documents"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="doc")
    
    def extract_text(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """Extract text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            pages_processed = 0
            
            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
                pages_processed += 1
            
            doc.close()
            
            return {
                "success": True,
                "text": text,
                "pages": pages_processed,
                "total_pages": len(doc),
                "path": pdf_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def summarize(self, pdf_path: str) -> Dict[str, Any]:
        """Summarize PDF content"""
        extraction = self.extract_text(pdf_path, max_pages=20)
        if not extraction["success"]:
            return extraction
        
        text = extraction["text"][:15000]  # Limit for LLM
        
        prompt = f"""Summarize this PDF document:

{text}

Provide:
1. Main topic (1 sentence)
2. Key points (5 bullets)
3. Important data/numbers
4. Conclusion
"""
        
        summary = self.llm.chat(prompt)
        
        return {
            "success": True,
            "summary": summary,
            "pages_analyzed": extraction["pages"],
            "path": pdf_path
        }
    
    def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables from PDF"""
        try:
            doc = fitz.open(pdf_path)
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Simple table detection via text blocks
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        # Heuristic for tables
                        text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]])
                        if "|" in text or "\t" in text:
                            tables.append({
                                "page": page_num + 1,
                                "content": text
                            })
            
            doc.close()
            return {"success": True, "tables": tables, "count": len(tables)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def qa(self, pdf_path: str, question: str) -> Dict[str, Any]:
        """Ask question about PDF"""
        extraction = self.extract_text(pdf_path, max_pages=30)
        if not extraction["success"]:
            return extraction
        
        context = extraction["text"][:12000]
        
        prompt = f"""Based on this PDF content, answer the question.

PDF Content:
{context}

Question: {question}

Answer concisely with page references if possible."""
        
        answer = self.llm.chat(prompt)
        return {"success": True, "answer": answer, "question": question}
