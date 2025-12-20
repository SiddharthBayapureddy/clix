# 🧠 Vorp: The Intelligent Terminal Companion

![vorp demo](assets/demo.gif)

**Vorp** is not just another CLI tool; it is a fully autonomous AI agent that lives in your terminal. It bridges the gap between natural language and system operations, allowing you to code, debug, refactor, and manage your system using plain English. 

Built on top of powerful Large Language Models (LLMs) and a robust tool-use framework, Vorp acts as a pair programmer that doesn't just suggest code—it **writes** it, **runs** it, and **fixes** it.

---

## 📚 Table of Contents

1.  [🚀 Quick Start](#-quick-start)
2.  [✨ Core Capabilities](#-core-capabilities)
3.  [🧠 Model Selection Guide](#-model-selection-guide)
4.  [🛠️ Advanced Usage & Workflows](#-advanced-usage--workflows)
5.  [🔒 Security & Privacy](#-security--privacy)
6.  [🏗️ Under the Hood (Architecture)](#-under-the-hood-architecture)
7.  [❓ Troubleshooting & FAQ](#-troubleshooting--faq)
8.  [🤝 Contributing](#-contributing)

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- A terminal (PowerShell, CMD, Bash, Zsh)
- **API Keys** (Optional but Recommended): [Groq](https://console.groq.com/keys) or [Google Gemini](https://aistudio.google.com/app/apikey).

### Installation

#### Option A: Install via PyPI (Stable)
The easiest way to get started is to install the latest stable release.
```bash
pip install vorp
```

#### Option B: Install from Source (Bleeding Edge)
For developers who want the latest features:
```bash
git clone https://github.com/SiddharthBayapureddy/vorp.git
cd vorp
# Create a virtual environment (Recommended)
python -m venv venv
# Activate it:
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
# Install
pip install -e .
```

### Initial Setup
Run Vorp for the first time:
```bash
vorp
```
Vorp will start in **Cloud Mode** (Trial) by default. To unlock full power, set up your keys:
1.  Run `/key` inside the Vorp chat.
2.  Paste your Groq or Gemini API key.
3.  Vorp will switch to **Local Mode**, giving you direct access to the models and tool capabilities.

---

## ✨ Core Capabilities

Vorp is designed to be an **agent**, meaning it takes actions to achieve goals.

### 1. Autonomous File Management
Vorp can Create, Read, Update, and Delete (CRUD) files.
*   **"Create a basic HTML5 boilerplate in index.html"** -> Vorp writes the file.
*   **"Read config.json and fix the syntax error"** -> Vorp reads, thinks, and applies the fix.
*   **"Refactor src/main.py to use async functions"** -> Vorp rewrites the code.

### 2. Chat with Your Codebase (RAG)
Retrieval-Augmented Generation (RAG) allows Vorp to "learn" your project.
*   **Index your project:** `/index ./my-project`
*   **Ask questions:** "How is authentication handled in this app?"
*   **Benefit:** Vorp retrieves the exact files and functions relevant to your query, reducing hallucinations and enabling large-project understanding.

### 3. System Operations
Vorp can execute shell commands (with your permission).
*   **"Install the dependencies in requirements.txt"** -> Runs `pip install -r requirements.txt`.
*   **"Run the unit tests and tell me which ones failed"** -> Runs `pytest` and analyzes the output.
*   **"Git commit my changes with a message about the UI update"** -> Stages and commits files.

### 4. Context Management
*   **`/add <file>`**: Manually add a file to the conversation context. Great for debugging a specific file.
*   **`/context`**: See what files and RAG indices are currently active.
*   **`/clear`**: Wipe the screen to focus on the current task.

---

## 🧠 Model Selection Guide

Vorp supports a variety of models via Groq and Gemini. Choosing the right model is crucial for performance and cost.

| Model | Provider | Best For | Speed | Context |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 8B (Instant)** | Groq | **Default.** Quick chats, simple edits, shell commands. | ⚡️⚡️⚡️ | 8k |
| **Llama 3.3 70B** | Groq | **Complex Coding.** Refactoring, architecting, difficult logic. | ⚡️⚡️ | 128k |
| **Gemini 2.5 Flash** | Google | **Large Context.** Analyzing massive files or logs. | ⚡️⚡️ | 1M+ |
| **DeepSeek R1 Distill** | Groq | **Reasoning.** Math, logic puzzles, intricate algorithms. | ⚡️ | 128k |

**💡 Pro Tip:**
*   Start with **Llama 3.1 8B** for speed.
*   Switch to **Llama 3.3 70B** (`vorp --model groq/llama-3.3-70b-versatile`) if the agent gets stuck or needs to generate high-quality code.
*   Use **Gemini** if you need to paste a 5,000-line log file.

To see all available models:
```bash
vorp --list
```

---

## 🛠️ Advanced Usage & Workflows

### The "Agentic" Workflow
Don't just use Vorp as a chatbot. Use it as a **coworker**.

**Bad Prompt:** "Write code for a snake game."
**Good Prompt (Agentic):** "Create a file named `snake.py`. Implement a classic Snake game using the `pygame` library. Then, create a `requirements.txt` file with the necessary dependencies."

**Why?** The second prompt triggers Vorp's **tool use**. It will:
1.  Create `snake.py`.
2.  Write the game code.
3.  Create `requirements.txt`.
4.  Write `pygame` into it.
5.  Summarize its actions.

### Debugging Workflow
1.  **Add the file:** `/add src/buggy_script.py`
2.  **Explain the error:** "I'm getting a RecursionError when I run this."
3.  **Let Vorp fix it:** "Analyze the logic, find the base case missing, and apply the fix."
4.  **Verify:** "Run the script to verify it works." (Vorp will run `python src/buggy_script.py` and check the output).

### RAG Workflow for New Codebases
1.  **Clone the repo:** `git clone ...`
2.  **Index it:** `vorp` -> `/index .`
3.  **Onboarding:** "Explain the high-level architecture of this project. Where is the entry point?"
4.  **Feature Add:** "I need to add a new API route for user login. Based on existing patterns in `routes/`, how should I implement this?"

---

## 🔒 Security & Privacy

Vorp runs **locally** on your machine.
*   **Local Mode:** Your code *never* leaves your machine except to be sent to the LLM provider (Groq/Google) for inference. Vorp does *not* store your code on any external servers.
*   **API Keys:** Your API keys are stored in `~/.vorp_config.json` on your local machine. They are never transmitted to Vorp developers.
*   **Human-in-the-Loop:** For dangerous operations (`delete_file`, `run_shell_command`), Vorp **halts and asks for permission**. You always have the final say.
*   **Sandboxing (Recommended):** While Vorp is safe, it's always good practice to run agentic AI tools inside a virtual environment or a container (Docker) to limit accidental system changes.

---

## 🏗️ Under the Hood (Architecture)

Vorp is a modular Python application designed for extensibility.

### 1. The Brain (`src/vorp/main.py`)
This is the command center. It initializes the `typer` app, handles the REPL (Read-Eval-Print Loop), and manages the conversation state. It acts as the "Tool Controller," intercepting the LLM's requests to use tools and executing the corresponding Python functions.

### 2. The Toolbelt (`src/vorp/tools.py`)
Defines the `read_file`, `write_file`, `replace_string`, etc. functions. These are pure Python functions that interface with the OS. The `get_tool_definitions()` function generates the JSON schema that tells the LLM how to use these tools.

### 3. The Memory (`src/vorp/rag.py`)
Handles the RAG pipeline.
*   **Ingestion:** Scans directories, respecting `.gitignore` and `RAG_IGNORE_DIRS`.
*   **Chunking:** Splits code into semantic chunks (overlapping windows) to preserve context.
*   **Vector Database:** Uses `ChromaDB` (local) to store embeddings generated by `sentence-transformers`.

### 4. The Connector (`litellm`)
We use `litellm` as an abstraction layer. This allows Vorp to swap between Groq, Gemini, OpenAI, Anthropic, or even local Ollama models with a single line of configuration change.

---

## ❓ Troubleshooting & FAQ

### "I encountered a hiccup."
*   **Cause:** Usually an API timeout or the model generated invalid JSON for a tool call.
*   **Fix:** Just type "Retry" or rephrase your request. If it persists, check your internet connection or switch models.

### "Rate Limit Reached"
*   **Cause:** You are using the free tier of Groq or Gemini and have sent too many messages quickly.
*   **Fix:** Wait a minute or switch to a different model (e.g., from Llama 70B to 8B).

### "The model keeps listing files without me asking!"
*   **Fix:** We've updated the system prompt to prevent this, but LLMs can be unpredictable. Tell it explicitly: "Stop listing files. Just answer the question."

### "Can Vorp delete my whole hard drive?"
*   **No.** Vorp requires explicit `y/n` confirmation for every single deletion and shell command. Unless you blindly type `y` to "delete /", you are safe.

---

## 🤝 Contributing

We love contributions! Whether it's a new feature, a bug fix, or a documentation improvement.

1.  **Fork** the repo.
2.  **Create** a branch: `git checkout -b feature/amazing-feature`.
3.  **Commit** your changes.
4.  **Push** to the branch.
5.  **Open** a Pull Request.

**Development Tips:**
*   Run `pip install -e .` to reflect changes immediately.
*   Check `src/vorp/constants.json` to tweak system prompts during testing.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SiddharthBayapureddy">Siddharth Bayapureddy</a>
</p>
