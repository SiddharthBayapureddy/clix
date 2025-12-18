# 🧠 Vorp

## 🎬 Demo

![vorp demo](assets/demo.gif)

**Vorp** is a terminal-based AI pair programmer and companion. It seamlessly indexes your codebase, allowing you to ask context-aware questions, retrieve relevant code snippets, and even modify files or execute commands without leaving your command line environment.

> **Note:** This project is under active development.

## 🚀 Key Features

*   **Flexible Deployment:** Run `vorp` in two modes:
    *   **Local Mode:** Use your own API keys for direct access to LLM providers (Groq, Google Gemini).
    *   **Cloud Mode:** Route requests through a secure proxy backend (either hosted by you or a public instance) for a frictionless experience without personal API keys.
*   **RAG (Chat with Codebase):** Index any project folder to enable context-aware queries using local embeddings.
*   **Autonomous Tools:** The agent can read, write, and manage files, list directories, and execute shell commands.
*   **Session Persistence:** Chat history is saved locally, allowing you to resume sessions later.
*   **Context Management:** Manually inject specific files into the context window for targeted assistance.
*   **Safety First:** Critical operations like file deletion or shell execution require explicit user confirmation.
*   **Cross-Platform:** Designed to work consistently on Windows (PowerShell/CMD), macOS, and Linux.

## 🛠️ Installation

### Via PyPI (Recommended)
You can install Vorp directly from PyPI:
```bash
pip install vorp
```

### From Source
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SiddharthBayapureddy/vorp.git
    cd vorp
    ```

2.  **Create a virtual environment:**
    *   **Windows:** `python -m venv venv` and `.\venv\Scripts\activate`
    *   **macOS/Linux:** `python3 -m venv venv` and `source venv/bin/activate`

3.  **Install in editable mode:**
    ```bash
    pip install -e .
    ```

4.  **Configure API Keys (Local Mode):**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

## 💻 Usage

Start the application:
```bash
vorp
```

### Interactive Commands

| Command | Description |
| :--- | :--- |
| `/index <path>` | Scans and indexes the specified directory for RAG. |
| `/rag` | Toggles RAG mode on or off. |
| `/add <file>` | Loads the content of a specific file into the active chat context. |
| `/context` | Displays currently loaded files and the active RAG project. |
| `/key` | Interactive setup for your API keys. |
| `/clear` | Clears the terminal screen. |
| `/exit-v` | Exits the application and **saves** the current chat history. |
| `/exit` | Exits the application and **deletes** the current chat history. |

### CLI Arguments

| Flag | Description |
| :--- | :--- |
| `--model <id>` | Starts the session with a specific model ID. |
| `--list` | Lists all supported models and their IDs. |

## 🏗️ Architecture & Capabilities

### Core System
Vorp is built using **Typer** for the CLI structure and **Rich** for beautiful terminal rendering (Markdown, Spinners, Tables). It uses **LiteLLM** to provide a unified interface to multiple LLM providers.

### Autonomous Capabilities (Tools)
In **Local Mode**, Vorp provides the LLM with a set of tools to interact with your system:
*   **`read_file`**: Allows the AI to examine your code.
*   **`write_file`**: Enables the AI to create or update files (overwrites entire content).
*   **`delete_file`**: Permanently removes files (requires confirmation).
*   **`list_files`**: Lets the AI explore your directory structure.
*   **`run_shell_command`**: Allows the AI to run tests, install packages, or use git (requires confirmation).

### RAG (Retrieval-Augmented Generation)
The RAG system ensures the AI has a deep understanding of your specific codebase:
1.  **Ingestion:** Files are split using a **Sliding Window** (1000 chars, 200 overlap).
2.  **Embeddings:** Uses `all-MiniLM-L6-v2` locally via **Sentence-Transformers**.
3.  **Storage:** Vectors are stored in **ChromaDB** at `~/.vorp_rag_db`.
4.  **Retrieval:** Uses Cosine Similarity filtered by `project_id` to fetch the top 5 most relevant code snippets.

### Cloud Proxy Mode
For a zero-config experience, Vorp can route requests through a **FastAPI-based proxy**. This proxy securely handles API keys and streams responses back to the CLI using Server-Sent Events (SSE).

## ⚙️ Configuration

Advanced configuration is managed via `src/vorp/constants.json`. You can customize:
*   **Models:** Add or remove supported model IDs.
*   **Ignore Patterns:** Define which files/folders RAG should skip.
*   **System Prompt:** Modify the core instructions given to the AI.

## 🔮 Roadmap
*   **Multi-file Editing:** Improving the logic for refactoring across multiple files.
*   **Better Diff Support:** Visualizing changes before they are applied.
*   **Plugin System:** Allow users to define custom tools.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.