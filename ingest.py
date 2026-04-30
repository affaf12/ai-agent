import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from pypdf import PdfReader
from rag_system_pro import RAGSystem, Chunker

rag = RAGSystem(
    llm_model="tinyllama:latest",
    embed_model="nomic-embed-text",
    index_path="data/faiss.index",
)

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

docs = []
for p in Path("docs").rglob("*"):
    if not p.is_file():
        continue
    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            text = read_pdf(p)
        elif suffix in {".txt", ".md", ".py", ".js", ".ts", ".java",
                        ".cpp", ".c", ".go", ".rs", ".rb", ".html",
                        ".css", ".json", ".yaml", ".yml"}:
            text = read_text_file(p)
        else:
            continue
        if text.strip():
            docs.append({
                "id": str(p),
                "text": text,
                "metadata": {"source": p.name, "type": suffix.lstrip(".")},
            })
            print(f"Loaded {p.name}")
    except Exception as e:
        print(f"Skipped {p}: {e}")

print(f"\nIngesting {len(docs)} files...")
n = rag.ingest(docs)
print(f"Indexed {n} chunks. Saved to data/faiss.index")