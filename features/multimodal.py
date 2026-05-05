import io
import re
import base64
from typing import Dict, Any
import pandas as pd
from PIL import Image, ImageOps


class MultimodalProcessor:
    """Handle images, audio, documents - ENHANCED FOR BUSINESS"""
    
    @staticmethod
    def process_image(image_bytes: bytes, max_size: int = 1568) -> Dict[str, Any]:
        """Process and resize image for vision models"""
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        if max(img.size) > max_size:
            img = ImageOps.contain(img, (max_size, max_size))
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "bytes": image_bytes,
            "base64": encoded,
            "size": img.size,
            "format": "jpeg"
        }
    
    @staticmethod
    def extract_document_text(file_bytes: bytes, filename: str) -> str:
        """Extract text from various document formats - BUSINESS ENHANCED"""
        ext = filename.lower().split('.')[-1]
        
        if ext == "pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except:
                return "[PDF extraction failed]"
                
        elif ext in ("docx", "doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                return "\n".join(p.text for p in doc.paragraphs)
            except:
                return "[DOCX extraction failed]"
                
        elif ext in ("xlsx", "xls"):
            try:
                # BUSINESS ENHANCEMENT: Read all sheets with analysis
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
                output = []
                output.append(f"EXCEL WORKBOOK: {filename}")
                output.append(f"Sheets: {', '.join(xls.sheet_names)}\n")
                
                for sheet in xls.sheet_names[:3]:  # First 3 sheets
                    df = pd.read_excel(xls, sheet_name=sheet)
                    output.append(f"=== SHEET: {sheet} ===")
                    output.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    output.append(f"Columns: {', '.join(map(str, df.columns))}")
                    
                    # Add data types
                    dtypes = df.dtypes.astype(str).to_dict()
                    output.append(f"Types: {dtypes}")
                    
                    # Sample data (first 20 rows)
                    output.append("\nSample Data:")
                    output.append(df.head(20).to_markdown(index=False))
                    
                    # Basic stats for numeric columns
                    numeric = df.select_dtypes(include=[np.number])
                    if not numeric.empty:
                        output.append("\nSummary Statistics:")
                        output.append(numeric.describe().to_markdown())
                    
                    output.append("\n")
                
                return "\n".join(output)
            except Exception as e:
                return f"[Excel extraction failed: {e}]"
                
        elif ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
                output = []
                output.append(f"CSV FILE: {filename}")
                output.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                output.append(f"Columns: {', '.join(map(str, df.columns))}")
                output.append("\nSample Data:")
                output.append(df.head(30).to_markdown(index=False))
                
                numeric = df.select_dtypes(include=[np.number])
                if not numeric.empty:
                    output.append("\nSummary:")
                    output.append(numeric.describe().to_markdown())
                
                return "\n".join(output)
            except Exception as e:
                return f"[CSV extraction failed: {e}]"
        
        elif ext in ("txt", "md", "py", "js", "ts", "json"):
            return file_bytes.decode('utf-8', errors='ignore')
        
        else:
            return f"[Unsupported format: {ext}]"
    
    @staticmethod
    def analyze_excel_business(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Deep business analysis of Excel files"""
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            analysis = {
                "filename": filename,
                "sheets": {},
                "summary": {}
            }
            
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                sheet_info = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample": df.head(5).to_dict('records'),
                    "numeric_summary": {}
                }
                
                # Business metrics
                numeric = df.select_dtypes(include=[np.number])
                for col in numeric.columns:
                    sheet_info["numeric_summary"][col] = {
                        "sum": float(numeric[col].sum()),
                        "mean": float(numeric[col].mean()),
                        "min": float(numeric[col].min()),
                        "max": float(numeric[col].max())
                    }
                
                analysis["sheets"][sheet] = sheet_info
            
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def edit_excel_with_instructions(file_bytes: bytes, filename: str, instruction: str) -> bytes:
        """Edit Excel based on natural language instruction"""
        try:
            df_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            
            # Simple instruction parser for business tasks
            instruction_lower = instruction.lower()
            
            for sheet_name, df in df_dict.items():
                # Example: "increase prices by 10%"
                if "increase" in instruction_lower and ("price" in instruction_lower or "cost" in instruction_lower):
                    pct = 10
                    match = re.search(r'(\d+)%', instruction)
                    if match:
                        pct = int(match.group(1))
                    
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if any(k in col.lower() for k in ['price','cost','amount','revenue','sales']):
                            df[col] = df[col] * (1 + pct/100)
                
                # Example: "filter top 10"
                elif "top" in instruction_lower:
                    match = re.search(r'top\s+(\d+)', instruction_lower)
                    if match:
                        n = int(match.group(1))
                        # Sort by first numeric column descending
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) > 0:
                            df = df.nlargest(n, num_cols[0])
                
                df_dict[sheet_name] = df
            
            # Save back to bytes
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in df_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output.getvalue()
        except Exception as e:
            raise Exception(f"Edit failed: {e}")


# =============================================================================
# RAG SYSTEM (Enhanced with chunking strategies)
# =============================================================================


