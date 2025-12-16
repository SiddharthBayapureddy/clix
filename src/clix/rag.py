import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Database Setup
DB_PATH = Path.home() / ".clix_rag_db"

try:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="codebase_context",
        embedding_function=default_ef
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChromaDB at {DB_PATH}: {e}")
    client = None
    collection = None

IGNORE_DIRS = {".git", "__pycache__", "node_modules", "venv", ".idea", ".vscode", "dist", "build", ".clix_rag_db"}
IGNORE_EXTS = {".pyc", ".exe", ".dll", ".so", ".dylib", ".png", ".jpg", ".jpeg", ".gif", ".ico"}

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    
    return chunks

def get_files(root_path: Path):
    """Recursively yields file paths to index."""
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in IGNORE_EXTS:
                yield file_path

def index_codebase(root_path_str: str, progress_callback=None):
    """Indexes the codebase at the given path."""
    if collection is None:
        raise RuntimeError("RAG database is not initialized.")

    root_path = Path(root_path_str).resolve()
    project_id = str(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Path not found: {root_path_str}")

    # Reset index for this project
    try:
        collection.delete(where={"project_id": project_id})
    except Exception:
        pass

    all_files = list(get_files(root_path))
    total_files = len(all_files)
    
    ids = []
    documents = []
    metadatas = []

    print(f"Found {total_files} files to index...")

    for i, file_path in enumerate(all_files):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            rel_path = str(file_path.relative_to(root_path))
            file_chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(file_chunks):
                # Unique ID: project_id:rel_path:chunk_index
                chunk_id = f"{project_id}:{rel_path}:{chunk_idx}"
                
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "path": str(file_path), 
                    "filename": file_path.name,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(file_chunks),
                    "project_id": project_id
                })
            
            if progress_callback:
                progress_callback(i + 1, total_files, file_path.name)
                
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    # Batch Upsert
    if ids:
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

    return total_files

def retrieve_context(query_text: str, project_id: str, n_results=5):
    """Retrieves top-k relevant code snippets for a query within a specific project."""
    if collection is None or not project_id:
        return ""

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"project_id": project_id}
    )
    
    context_parts = []
    if results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        for doc, meta in zip(docs, metas):
            path = meta.get("path", "Unknown")
            context_parts.append(f"File: {path}\n```\n{doc}\n```")
            
    return "\n\n".join(context_parts)
