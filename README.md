# 🧠 vorp

## 🎬 Demo

![vorp demo](demo.gif)



**vorp** is a terminal-based AI pair programmer. It indexes your codebase, allowing you to ask context-aware questions and retrieve relevant code snippets without leaving your command line environment.

> **Note:** This project is under active development.

## 🚀 Key Features

*   **Flexible Deployment:** Run `vorp` in two modes:
    *   **Local Mode:** Use your own API keys for direct access to LLM providers.
    *   **Cloud Mode:** Route requests through a secure proxy backend (either hosted by you or a public instance) for a frictionless experience without personal API keys.
*   **RAG (Chat with Codebase):** Index any project folder to enable context-aware queries.
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
    git clone https://github.com/SiddharthBayapureddy/vorp.git
    cd vorp
    ```
    *(Note: If the repository is renamed to `vorp`, clone that instead.)*

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

4.  **Configure API Keys (Local Mode):**
    For **Local Mode**, you need to provide your own API keys. Create a `.env` file in the root directory and add your keys:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    ```
    If these keys are not found, `vorp` will automatically switch to **Cloud Mode**.

5.  **Cloud Mode Configuration (Optional, for Custom Backends):**
    If you are running your own backend server or wish to use a specific Cloud Mode instance, you can configure `vorp` to point to it.
    Add the following to your `.env` file (or set as environment variables):
    ```env
    VORP_BACKEND_URL="https://your-custom-backend-url.com/chat"
    VORP_ACCESS_TOKEN="your_access_token" # Only needed if your backend requires a custom token
    ```
    *Note: The CLI has a hardcoded public access token (`sk-vorp-public-beta`) that is used if `VORP_ACCESS_TOKEN` is not explicitly provided, and local API keys are missing. This token must also be configured on your backend server.*

### Cloud Mode Backend

For users who prefer not to manage their own API keys, `vorp` can operate in **Cloud Mode**. In this mode, the CLI routes chat requests through a proxy backend server you host. This server securely holds your LLM API keys and handles the communication with providers like Groq and Google Gemini.

**Architecture:**
*   The CLI sends chat requests to your hosted backend (e.g., `https://your-backend.vercel.app/chat`).
*   The backend validates an `Authorization` header with an access token (which is either a public default or one you supply).
*   The backend securely uses its own environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`) to call the LLM providers.
*   LLM responses are streamed back through the backend to the CLI.

**Benefits:**
*   **Frictionless User Experience:** Users don't need to provide their own API keys.
*   **Centralized Control:** You control API key management, rate limits, and monitoring on your backend.
*   **Security:** Your private API keys are never exposed to client-side applications.

**Deployment (Example using Vercel):**

1.  **Project Setup:**
    *   Ensure your `server/` directory contains `app.py` and `requirements.txt`.
    *   Place a `vercel.json` file in your project root with the following (adjust `runtime` if needed):
        ```json
        {
          "version": 2,
          "builds": [
            {
              "src": "server/app.py",
              "use": "@vercel/python",
              "config": {
                "maxLambdaSize": "15mb",
                "runtime": "python3.10"
              }
            }
          ],
          "routes": [
            {
              "src": "/(.*)",
              "dest": "server/app.py"
            }
          ]
        }
        ```
2.  **Host on Vercel:**
    *   Commit and push your entire project (including `server/` and `vercel.json`) to a GitHub repository.
    *   Go to [vercel.com](https://vercel.com/) and create a new project from your repository.
    *   **Configure Build:** In Vercel Project Settings, set the **Root Directory** to `server` (this tells Vercel to only build the backend part of your repo).
    *   **Environment Variables:** Add the following to your Vercel project's Environment Variables:
        *   `GROQ_API_KEY`: Your actual Groq API key.
        *   `GEMINI_API_KEY`: Your actual Google Gemini API key.
        *   `VORP_ACCESS_TOKEN`: Set this to `sk-vorp-public-beta` (to match the CLI's default hardcoded token).
    *   Deploy the project.
3.  **Update CLI:** Once deployed, your CLI will automatically use this backend if local API keys are not found, or you can explicitly set `VORP_BACKEND_URL` in your `.env` file.

Start the application:
```bash
vorp
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

You can configure `vorp` at startup using these flags:

| Flag | Description |
| :--- | :--- |
| `--model <id>` | Starts the session with a specific model (e.g., `groq/llama-3.3-70b-versatile`). |
| `--list` | Lists all supported models and their IDs, then exits. |
| `--help` | Displays the help message. |

*Example:*
```bash
vorp --model "gemini/gemini-2.5-pro"
```

## 🏗️ Architecture

The Retrieval-Augmented Generation (RAG) system in `vorp` is built for speed and privacy. Here is how it works under the hood:

1.  **Ingestion & Chunking:**
    *   When you run `/index`, the system walks through your project directory.
    *   Files are read and split into smaller segments using a **Sliding Window** approach (1000 characters with 200 character overlap). This ensures that context at the boundaries of chunks is preserved.

2.  **Embedding Generation:**
    *   Each chunk is passed through the `all-MiniLM-L6-v2` model. This is a lightweight, high-performance model that runs locally on your CPU.
    *   The model converts the text code into a 384-dimensional vector (a list of numbers representing the semantic meaning).

3.  **Vector Storage (ChromaDB):**
    *   These vectors are stored in **ChromaDB**, a persistent local vector database located at `~/.vorp_rag_db`.
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