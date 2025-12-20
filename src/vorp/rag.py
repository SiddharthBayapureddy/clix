import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Suppress ChromaDB Info/Warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Load Configuration
CONSTANTS_PATH = Path(__file__).parent / "constants.json"
try:
    with open(CONSTANTS_PATH, "r") as f:
        _CONSTANTS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    _CONSTANTS = {}

# Constants
DB_PATH = Path.home() / ".vorp_rag_db"
IGNORE_DIRS = set(_CONSTANTS.get("RAG_IGNORE_DIRS", [".git", "__pycache__", "node_modules", "venv", ".idea", ".vscode", "dist", "build", ".vorp_rag_db"]))
IGNORE_EXTS = set(_CONSTANTS.get("RAG_IGNORE_EXTS", [".pyc", ".exe", ".dll", ".so", ".dylib", ".png", ".jpg", ".jpeg", ".gif", ".ico"]))

# Initialization
try:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="codebase_context",
        embedding_function=default_ef
    )
except Exception as e:
    # Fail silently/gracefully if RAG isn't critical, but warn user in stderr
    import sys
    print(f"[Warning] RAG Initialization failed: {e}", file=sys.stderr)
    client = None
    collection = None

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits text into overlapping chunks to preserve context at boundaries.
    """
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
    """
    Recursively scans the directory for valid files to index.
    """
    for root, dirs, files in os.walk(root_path):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in IGNORE_EXTS:
                yield file_path

def index_codebase(root_path_str: str, progress_callback=None):
    """
    Indexes the target codebase into ChromaDB.
    """
    if collection is None:
        raise RuntimeError("RAG database could not be initialized.")

    root_path = Path(root_path_str).resolve()
    project_id = str(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path_str}")

    # Clear previous index for this project to avoid stale data
    try:
        collection.delete(where={"project_id": project_id})
    except Exception:
        pass # Collection might be empty or new

    all_files = list(get_files(root_path))
    total_files = len(all_files)
    
    ids = []
    documents = []
    metadatas = []
    errors = []

    for i, file_path in enumerate(all_files):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Use relative path for cleaner context
            try:
                rel_path = str(file_path.relative_to(root_path))
            except ValueError:
                rel_path = str(file_path)

            file_chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(file_chunks):
                # ID format: project_id:relative_path:chunk_index
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
            errors.append(f"{file_path}: {e}")

    # Batch upsert to improve performance
    if ids:
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

    return total_files, len(ids), errors

def retrieve_context(query_text: str, project_id: str, n_results=5):
    """
    Searches the vector database for code snippets relevant to the user's query.
    """
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
            # Wrap in markdown code block for clarity
            context_parts.append(f"File: {path}\n```\n{doc}\n```")
            
    return "\n\n".join(context_parts)
