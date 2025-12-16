# 🧠 clix

<img src="./logo.jpeg" alt="clix logo" width="200">

**clix** is a terminal-based AI pair programmer. It indexes your codebase, allowing you to ask context-aware questions and retrieve relevant code snippets without leaving your command line environment.

> **Note:** This project is under active development.

## 🚀 Key Features

*   **RAG (Chat with Codebase):** Index any project folder to enable context-aware queries.
    *   **Project Isolation:** Uses a global vector database with metadata filtering. Context from Project A will never leak into Project B.
    *   **Local Storage:** All embeddings are stored locally in `~/.clix_rag_db`.
*   **Multi-Model Support:** Integrates with Groq and Google Gemini to provide access to models like Llama 3.3, DeepSeek R1, and Gemini 2.5 Pro.
*   **Session Persistence:** Chat history is saved locally, allowing you to resume sessions later.
*   **Context Management:** Manually inject specific files into the context window for targeted assistance.
*   **Cross-Platform:** Designed to work consistently on Windows, macOS, and Linux.

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SiddharthBayapureddy/clix.git
    cd clix
    ```

2.  **Create a virtual environment:**
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the root directory and add your keys:
    ```env
    GROQ_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here
    ```

## 🎮 Usage

Start the application:
```bash
clix
```

### Interactive Commands

| Command | Description |
| :--- | :--- |
| `/index <path>` | Scans and indexes the specified directory. This creates a searchable vector index for RAG. |
| `/rag` | Toggles RAG mode on or off. When enabled, the assistant retrieves context from the indexed project. |
| `/add <file>` | Loads the content of a specific file into the active chat context. |
| `/context` | Displays a list of currently loaded files and the active RAG project path. |
| `/clear` | Clears the terminal screen. |
| `/exit-v` | Exits the application and **saves** the current chat history. |
| `/exit` | Exits the application and **deletes** the current chat history. |

### CLI Arguments

You can configure `clix` at startup using these flags:

| Flag | Description |
| :--- | :--- |
| `--model <id>` | Starts the session with a specific model (e.g., `groq/llama-3.3-70b-versatile`). |
| `--list` | Lists all supported models and their IDs, then exits. |
| `--help` | Displays the help message. |

*Example:*
```bash
clix --model "gemini/gemini-2.5-pro"
```

## 🏗️ Architecture

The Retrieval-Augmented Generation (RAG) system in `clix` is built for speed and privacy. Here is how it works under the hood:

1.  **Ingestion & Chunking:**
    *   When you run `/index`, the system walks through your project directory.
    *   Files are read and split into smaller segments using a **Sliding Window** approach (1000 characters with 200 character overlap). This ensures that context at the boundaries of chunks is preserved.

2.  **Embedding Generation:**
    *   Each chunk is passed through the `all-MiniLM-L6-v2` model. This is a lightweight, high-performance model that runs locally on your CPU.
    *   The model converts the text code into a 384-dimensional vector (a list of numbers representing the semantic meaning).

3.  **Vector Storage (ChromaDB):**
    *   These vectors are stored in **ChromaDB**, a persistent local vector database located at `~/.clix_rag_db`.
    *   **Isolation Layer:** Every vector is tagged with a `project_id` metadata field (the absolute path of the project). This acts as a strict filter, ensuring that queries only search within the active project's scope.

4.  **Retrieval (Cosine Similarity):**
    *   When you ask a question in RAG mode, your query is embedded using the same model.
    *   The database performs a similarity search (using Cosine Similarity) to find the top 5 chunks that are mathematically closest to your query.
    *   This retrieval is strictly filtered by the active `project_id`.

5.  **Context Injection:**
    *   The retrieved code snippets are formatted and injected into the LLM's system prompt.
    *   The LLM then generates an answer using this retrieved knowledge, allowing it to "see" your code.

## 🔮 Roadmap

*   **File Editing:** Capabilities for the agent to autonomously modify files.
*   **Command Execution:** Safe execution of shell commands for testing and linting.
*   **Diff View:** Enhanced visualization of code changes.

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements.
