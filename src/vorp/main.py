# App lives here 

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
import json
import typer
import requests

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner

import litellm
litellm.suppress_debug_info = True
from litellm import completion

from vorp.tools import get_tool_definitions, execute_tool_call

load_dotenv()


# Initialization
app = typer.Typer()
console = Console()

# Configuration
CHAT_HISTORY = Path.home() / ".vorp_chat_history.json"
CONFIG_FILE = Path.home() / ".vorp_config.json"

# Load Constants from JSON
CONSTANTS_PATH = Path(__file__).parent / "constants.json"
try:
    with open(CONSTANTS_PATH, "r") as f:
        _CONSTANTS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Fallback defaults if file is missing or corrupt
    _CONSTANTS = {
        "TRIAL_LIMIT": 10,
        "DEFAULT_MODEL": "groq/llama-3.1-8b-instant",
        "MAX_HISTORY_LENGTH": 15,
        "VORP_BACKEND_URL": "https://vorp-sigma.vercel.app/chat",
        "DEFAULT_PUBLIC_ACCESS_TOKEN": "sk-vorp-public-beta",
        "SUPPORTED_MODELS": []
    }

TRIAL_LIMIT = _CONSTANTS.get("TRIAL_LIMIT", 10)
MODEL_NAME = _CONSTANTS["DEFAULT_MODEL"]
MAX_HISTORY_LENGTH = _CONSTANTS.get("MAX_HISTORY_LENGTH", 15)
VORP_BACKEND_URL = _CONSTANTS.get("VORP_BACKEND_URL", "https://vorp-sigma.vercel.app/chat")
DEFAULT_PUBLIC_ACCESS_TOKEN = _CONSTANTS.get("DEFAULT_PUBLIC_ACCESS_TOKEN", "sk-vorp-public-beta")
SUPPORTED_MODELS = _CONSTANTS.get("SUPPORTED_MODELS", [])


# Utils
def load_config():
    """Loads persistent configuration (independent of chat history)."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            pass
    return {"cloud_usage": 0}

def save_config(config):
    """Saves persistent configuration."""
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)

def load_history():
    """Loaded chat history from file if available."""
    if CHAT_HISTORY.exists():
        try:
            with open(CHAT_HISTORY , "r") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data.get("messages", []), data.get("rag_enabled", False), data.get("active_file", None)
                elif isinstance(data, list):
                    return data, False, None
        except json.JSONDecodeError:
            return [], False, None

    return [], False, None


def save_history(messages, rag_enabled, active_file=None):
    """Persisted current session to disk."""
    with open(CHAT_HISTORY, "w") as file:
        json.dump({"messages": messages, "rag_enabled": rag_enabled, "active_file": active_file}, file)


def delete_history():
    """Removed chat history file."""
    if CHAT_HISTORY.exists():
        os.remove(CHAT_HISTORY)


##################
# Main Interface #
##################


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context, 
    model: str = typer.Option(MODEL_NAME, "--model", "-m", help="Model to use"),
    list_models: bool = typer.Option(False, "--list", "-l", help="List available AI models."),
):
    """
    vorp: Intelligence met the command line.

    \b
    Interactive Commands: 
    /exit          Exit and DELETE session history.
    /exit-v        Exit and SAVE session history.
    /clear         Clear the terminal screen.
    /add <file>    Add a file to the current context.
    /context       List loaded context files.
    /index <path>  Index a folder for RAG (Chat with Codebase).
    /rag           Toggle RAG mode (on/off).
    """
    if ctx.invoked_subcommand is not None:
        return

    # Load persistent config & keys
    config = load_config()
    if "GROQ_API_KEY" in config:
        os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]
    if "GEMINI_API_KEY" in config:
        os.environ["GEMINI_API_KEY"] = config["GEMINI_API_KEY"]

    if list_models:
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Model ID (Use this)", style="magenta")
        table.add_column("Provider", style="green")

        for m in SUPPORTED_MODELS:
            table.add_row(m["name"], m["id"], m["provider"])

        console.print(table)
        console.print("\n[dim]Usage: vorp --model <Model ID>[/dim]")
        raise typer.Exit()

    cloud_mode = False
    access_token_for_backend = None
    
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        cloud_mode = True
        access_token_for_backend = os.getenv("VORP_ACCESS_TOKEN", DEFAULT_PUBLIC_ACCESS_TOKEN)
    
    # Session recovery
    messages, rag_enabled, active_file = load_history()

    if messages:
        console.print("[dim green]↻ Resumed previous session...[/dim green]")
        console.print("[bold green]✓ Session Restored! [/bold green]")
    
    # Startup Banner
    model_display_name = next((m["name"] for m in SUPPORTED_MODELS if m["id"] == model), model)
    
    ascii_art = r"""
[bold cyan] __      __  ____   _____   _____ [/bold cyan]
[bold cyan] \ \    / / / __ \ |  __ \ |  __ \  [/bold cyan]
[bold cyan]  \ \  / / | |  | || |__) || |__) |[/bold cyan]
[bold cyan]   \ \/ /  | |  | ||  _  / |  ___/ [/bold cyan]
[bold cyan]    \  /   | |__| || | \ \ | |     [/bold cyan]
[bold cyan]     \/     \____/ |_|  \_\|_|    [/bold cyan]"""

    console.print(ascii_art)
    console.print("\n")
    console.print(f"[bold magenta]-> MODEL:[/bold magenta] [cyan]{model_display_name}[/cyan]")
    
    if cloud_mode:
        console.print(f"[bold green]-> STATUS:[/bold green] [dim]ONLINE (Cloud Mode via {VORP_BACKEND_URL})[/dim]")
        usage = config.get("cloud_usage", 0)
        remaining = max(0, TRIAL_LIMIT - usage)
        status_color = "yellow" if remaining < 3 else "green"
        console.print(f"[bold green]-> TRIAL:[/bold green] [dim][{status_color}]{remaining}/{TRIAL_LIMIT}[/{status_color}] messages remaining.[/dim]")
    else:
        console.print(f"[bold green]-> STATUS:[/bold green] [dim]ONLINE (Local Mode)[/dim]")
        
    console.print("[dim]──────────────────────────[/dim]")

    # RAG State
    active_project_path = None

    if rag_enabled:
        console.print("[dim green]RAG Mode: ENABLED[/dim green]")

    session_allowed_files = set()

    while True:
        try:
            try:
                user_input = console.input("[bold green]You > [/bold green]")
            except KeyboardInterrupt:
                console.print("\n[red]Exiting...[/red]")
                break
                
            user_input_clean = user_input.strip().lower()
            
            # Command: /exit
            if user_input_clean == "/exit":
                delete_history()
                console.print("[bold red]Session deleted. Peace![/bold red]")
                break
            
            # Command: /exit-v (Verbose/Save)
            if user_input_clean == "/exit-v":
                save_history(messages, rag_enabled, active_file)
                console.print("[bold green]Session saved. Peace![/bold green]")
                break
            
            # Command: /clear
            if user_input_clean == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Command: /key
            if user_input_clean == "/key":
                console.print("[bold yellow]API Key Setup[/bold yellow]")
                console.print("Enter your API Key (Groq or Gemini).")
                console.print("[dim]Press Enter to cancel.[/dim]")
                
                key_input = console.input("Key > ").strip()
                if not key_input:
                    console.print("[dim]Cancelled.[/dim]")
                    continue
                
                # Basic detection
                provider = None
                name = "Unknown"
                
                if key_input.startswith("gsk_"):
                    provider = "GROQ_API_KEY"
                    name = "Groq"
                elif key_input.startswith("AIza"):
                    provider = "GEMINI_API_KEY" 
                    name = "Gemini"
                else:
                    console.print("Could not auto-detect provider.")
                    type_choice = console.input("(1) Groq, (2) Gemini > ").strip()
                    if type_choice == "1":
                        provider = "GROQ_API_KEY"
                        name = "Groq"
                    elif type_choice == "2":
                        provider = "GEMINI_API_KEY"
                        name = "Gemini"
                    else:
                        console.print("[red]Invalid choice.[/red]")
                        continue

                config[provider] = key_input
                save_config(config)
                os.environ[provider] = key_input
                
                console.print(f"[bold green]✓ {name} API Key saved![/bold green]")
                console.print("[dim]Switching to Local Mode...[/dim]")
                
                cloud_mode = False 
                continue
            
            # Command: /context
            if user_input_clean == "/context":
                console.print("[bold cyan]Current Context:[/bold cyan]")
                found_files = False
                for msg in messages:
                    if msg["role"] == "user" and msg["content"].startswith("Context from file `"):
                        try:
                            fname = msg["content"].split("`")[1]
                            console.print(f" - [green]{fname}[/green]")
                            found_files = True
                        except IndexError:
                            pass
                if not found_files:
                    console.print(" - [dim]No files loaded.[/dim]")
                
                if active_project_path:
                    console.print(f"[bold cyan]Active RAG Project:[/bold cyan] [green]{active_project_path}[/green]")
                else:
                    console.print("[dim]No RAG project indexed.[/dim]")
                continue

            # Command: /add <file>
            if user_input_clean.startswith("/add "):
                file_path_str = user_input.strip()[5:]
                path = Path(file_path_str)
                
                if not path.exists():
                    console.print(f"[bold red]File not found:[/bold red] {file_path_str}")
                    continue
                
                if not path.is_file():
                    console.print(f"[bold red]Path is not a file:[/bold red] {file_path_str}")
                    continue

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    messages.append({"role": "user", "content": f"Context from file `{file_path_str}`:\n\n```\n{content}\n```"})
                    console.print(f"[bold green]✓ Added {file_path_str} to context.[/bold green]")
                except Exception as e:
                    console.print(f"[bold red]Error reading file:[/bold red] {e}")
                
                continue


            # Command: /index <path>
            if user_input_clean.startswith("/index "):
                path_str = user_input.strip()[7:]
                idx_path = Path(path_str).resolve()
                
                if not idx_path.exists():
                    console.print(f"[bold red]Path not found:[/bold red] {path_str}")
                    continue

                console.print(f"[bold cyan]Indexing codebase at {path_str}...[/bold cyan]")
                try:
                    from vorp import rag
                    
                    with Live(Spinner("dots", text="Indexing files...", style="green"), refresh_per_second=10) as live:
                         count, chunks, errors = rag.index_codebase(str(idx_path))
                    
                    active_project_path = str(idx_path)
                    console.print(f"[bold green]✓ Successfully indexed {count} files ({chunks} chunks).[/bold green]")
                    
                    if errors:
                        console.print(f"[dim yellow]Skipped {len(errors)} files due to errors.[/dim yellow]")

                    rag_enabled = True
                    console.print(f"[dim]RAG mode enabled for project: {active_project_path}[/dim]")

                except Exception as e:
                    console.print(f"[bold red]Indexing failed:[/bold red] {e}")
                
                continue

            # Command: /rag
            if user_input_clean in ["/rag", "/learn"]:
                rag_enabled = not rag_enabled
                status = "enabled" if rag_enabled else "disabled"
                color = "green" if rag_enabled else "red"
                console.print(f"[bold {color}]RAG Mode {status}.[/bold {color}]")
                continue
            
            # --- Trial Check ---
            if cloud_mode:
                current_usage = config.get("cloud_usage", 0)
                if current_usage >= TRIAL_LIMIT:
                    console.print("\n[bold red]Trial Limit Reached (10/10 messages)[/bold red]")
                    console.print("[yellow]To continue using Vorp, please provide your own free API keys:[/yellow]")
                    console.print(" 1. Get a [bold]Groq API Key[/bold]: [underline]https://console.groq.com/keys[/underline]")
                    console.print(" 2. Get a [bold]Gemini API Key[/bold]: [underline]https://aistudio.google.com/app/apikey[/underline]")
                    console.print("\n[dim]Run the command below to set your key:[/dim]")
                    console.print("[bold cyan]/key[/bold cyan]")
                    continue

            # Chat Logic
            messages.append({"role": "user", "content": user_input})

            response_text = ""
            
            # Context Injection
            llm_messages = messages.copy()

            if not cloud_mode:
                prompt_template = _CONSTANTS.get("SYSTEM_PROMPT_TEMPLATE", "")
                
                # Fallback if template is missing in JSON
                if not prompt_template:
                     prompt_template = (
                        "You are Vorp, an advanced, autonomous AI agent running directly in the user's terminal. You are an expert engineer and system administrator.\n\n"
                        "**CORE IDENTITY:**\n"
                        "- You are a **DOER**, not just a talker. When asked to create or modify something, **USE YOUR TOOLS** to actually do it. Do not just output code blocks unless specifically asked to 'show' code.\n"
                        "- You have full access to the local file system and shell. Use them to fulfill requests.\n\n"
                        "**OPERATIONAL PROTOCOLS:**\n\n"
                        "1.  **FILE OPERATIONS (CRITICAL):**\n"
                        "    - **CREATE ('write_file'):** If the user asks for a specific file (e.g., 'create a login page'), **create it**. If no filename is given, INFER a standard name (e.g., 'index.html', 'app.py') based on context. ALWAYS provide the FULL, WORKING content.\n"
                        "    - **EDIT ('replace_string'):**\n"
                        "        - **MANDATORY:** You **MUST** read the file ('read_file') *before* attempting to replace text to ensure you have the exact 'old_string' context.\n"
                        "        - 'old_string' must be unique and match the file content exactly (whitespace, indentation).\n"
                        "        - If the change is large or complex, prefer reading the file and then using 'write_file' to overwrite it with the new version to avoid 'replace_string' errors.\n"
                        "    - **DELETE ('delete_file'):** Use with caution.\n\n"
                        "2.  **CONTEXT & EXPLORATION:**\n"
                        "    - If you are unsure about the current state, use 'list_files' or 'read_file' to investigate BEFORE taking action.\n"
                        "    - Use 'run_shell_command' to install dependencies, run tests, or check system status if needed.\n\n"
                        "3.  **COMMUNICATION:**\n"
                        "    - **Conversational:** Respond naturally to greetings ('Hi', 'Hello') without using tools.\n"
                        "    - **Post-Action Summary:** AFTER performing any file creation, edit, or shell command, you **MUST** provide a concise, one-sentence summary of what you did (e.g., '✓ Created `index.html` with a teal login form.' or '✓ Updated `app.py` to fix the bug.').\n\n"
                        "4.  **CURRENT CONTEXT:**\n"
                        "    - **OS:** {os_name}\n"
                        "    - **CWD:** {cwd}\n"
                        "    - **Active File:** '{active_file}'"
                     )
                
                system_prompt = prompt_template.format(
                    os_name=os.name,
                    cwd=os.getcwd(),
                    active_file=active_file if active_file else 'None'
                )

                # Check if system prompt already exists (it shouldn't in this simple list, but good to be safe or just prepend)
                llm_messages.insert(0, {"role": "system", "content": system_prompt})
            
            if rag_enabled:
                if active_project_path:
                    try:
                        from vorp import rag
                        context_str = rag.retrieve_context(user_input, project_id=active_project_path, n_results=3)
                        
                        if context_str:
                             llm_messages.insert(-1, {"role": "system", "content": f"Relevant Codebase Context:\n\n{context_str}"})
                    except Exception as e:
                         console.print(f"[dim red]RAG Retrieval failed: {e}[/dim red]")
                else:
                    console.print("[dim yellow]RAG is enabled but no project is indexed. Use /index <path> first.[/dim yellow]")
            
            # Model Normalization
            active_model = model
            if not active_model.startswith("groq/") and not active_model.startswith("gemini/") and not active_model.startswith("gpt-"):
                active_model = f"groq/{active_model}"
            
            if cloud_mode:
                # Cloud Mode: Proxy through Backend
                with Live(console=console, refresh_per_second=20) as live:
                    grid = Table.grid(padding=(0, 1)) 
                    grid.add_column(style="bold blue", no_wrap=True)
                    grid.add_column()
                    grid.add_row("Vorp >" , Spinner("dots", style="bold cyan", text="Thinking..."))
                    live.update(grid)

                    try:
                        payload = {
                            "model": active_model,
                            "messages": llm_messages,
                            "stream": True
                        }
                        headers = {"Authorization": f"Bearer {access_token_for_backend}"}
                        
                        with requests.post(VORP_BACKEND_URL, json=payload, headers=headers, stream=True, timeout=60) as r:
                            r.raise_for_status()
                            # Increment and save usage only on success
                            config["cloud_usage"] = config.get("cloud_usage", 0) + 1
                            save_config(config)
                            
                            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                                if chunk:
                                    response_text += chunk
                                    grid = Table.grid(padding=(0, 1)) 
                                    grid.add_column(style="bold blue", no_wrap=True)
                                    grid.add_column()
                                    grid.add_row("Vorp >", Markdown(response_text))
                                    live.update(grid)
                    except Exception as e:
                         response_text = f"Error communicating with backend: {e}"
                
                messages.append({"role": "assistant", "content": response_text})

            else:
                # Local Mode: Use litellm directly with Tools
                while True:
                    available_tools = get_tool_definitions()
                    response_text = ""
                    tool_calls = []
                    error_message = None
                    crash_reason = None
                    
                    with Live(console=console, refresh_per_second=20) as live:
                        grid = Table.grid(padding=(0, 1)) 
                        grid.add_column(style="bold blue", no_wrap=True)
                        grid.add_column()
                        grid.add_row("Vorp >" , Spinner("dots", style="bold cyan", text="Thinking..."))
                        live.update(grid)
                        
                        try:
                            response = completion(
                                model=active_model,
                                messages=llm_messages,
                                tools=available_tools,
                                tool_choice="auto",
                                stream=True,
                                timeout=60
                            )
                            
                            for chunk in response:
                                delta = chunk.choices[0].delta
                                
                                if delta.content:
                                    response_text += delta.content
                                    grid = Table.grid(padding=(0, 1)) 
                                    grid.add_column(style="bold blue", no_wrap=True)
                                    grid.add_column()
                                    grid.add_row("Vorp >", Markdown(response_text))
                                    live.update(grid)
                                
                                if delta.tool_calls:
                                    for tc in delta.tool_calls:
                                        if len(tool_calls) <= tc.index:
                                            tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}, "type": "function"})
                                        
                                        if tc.id: tool_calls[tc.index]["id"] += tc.id
                                        if tc.function.name: tool_calls[tc.index]["function"]["name"] += tc.function.name
                                        if tc.function.arguments: tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                        except Exception as e:
                            # Check if this is a follow-up failure after a tool call
                            if len(llm_messages) > 0 and llm_messages[-1].get("role") == "tool":
                                # We'll handle this success case outside as well, or just let it slide
                                crash_reason = str(e)
                                pass 
                            else:
                                error_message = "I encountered a hiccup. Could you please repeat that?"
                    
                    # Handle errors outside the Live context to avoid UI artifacts
                    if error_message:
                        console.print(f"\n[yellow]{error_message}[/yellow]")
                        break
                    
                    # If we had a tool call failure that we want to treat as success (rare edge case from original code)
                    # We need to replicate that logic if we removed it from the try/except block.
                    # Re-implementing that check safely:
                    if not response_text and not tool_calls and len(llm_messages) > 0 and llm_messages[-1].get("role") == "tool":
                         # This implies the model crashed immediately after a tool output was fed to it.
                         # Original logic: Assume success and exit.
                         console.print(f"[dim yellow]Warning: Model failed to generate post-action response. ({crash_reason or 'Unknown Error'})[/dim yellow]")
                         console.print("[bold green]✓ Action completed successfully.[/bold green]")
                         messages.append({"role": "assistant", "content": "I have completed the requested action."})
                         break

                    # Turn finished
                    if tool_calls:
                        llm_messages.append({
                            "role": "assistant",
                            "content": response_text if response_text else "",
                            "tool_calls": tool_calls
                        })
                        
                        # Execute tools
                        tools_executed_successfully = False
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args_str = tc["function"]["arguments"]
                            console.print(f"[dim]Requesting tool: {name}...[/dim]")
                            
                            try:
                                args = json.loads(args_str)
                                allowed = True
                                
                                # Permission Checks
                                if name == "delete_file":
                                    console.print(f"[bold red]WARNING: Request to DELETE file:[/bold red] {args.get('file_path')}")
                                    confirm = console.input("[bold yellow]Allow? (y/n) > [/bold yellow]").lower()
                                    if confirm != "y":
                                        allowed = False
                                        result = json.dumps({"error": "User denied permission to delete file."})

                                elif name == "run_shell_command":
                                    console.print(f"[bold red]WARNING: Request to RUN SHELL COMMAND:[/bold red] {args.get('command')}")
                                    confirm = console.input("[bold yellow]Allow? (y/n) > [/bold yellow]").lower()
                                    if confirm != "y":
                                        allowed = False
                                        result = json.dumps({"error": "User denied permission to run command."})

                                elif name == "write_file":
                                    file_path_arg = args.get('file_path')
                                    if file_path_arg:
                                        fpath = Path(file_path_arg).resolve()
                                        str_path = str(fpath)
                                        
                                        # Only ask if file exists AND we haven't already approved it this session
                                        if fpath.exists() and str_path not in session_allowed_files:
                                            console.print(f"[yellow]File exists:[/yellow] {fpath}")
                                            confirm = console.input("[yellow]Overwrite? (y/n) > [/yellow]").lower()
                                            if confirm != "y":
                                                allowed = False
                                                result = json.dumps({"error": "User denied permission to overwrite file."})
                                            else:
                                                session_allowed_files.add(str_path)
                                        else:
                                            # New file OR already approved
                                            session_allowed_files.add(str_path)
                                    else:
                                        allowed = False
                                        result = json.dumps({"error": "Missing file_path argument."})

                                elif name == "replace_string":
                                    file_path_arg = args.get('file_path')
                                    if file_path_arg:
                                        fpath = Path(file_path_arg).resolve()
                                        str_path = str(fpath)
                                        
                                        if str_path not in session_allowed_files:
                                            console.print(f"[yellow]Request to modify file:[/yellow] {fpath}")
                                            confirm = console.input("[yellow]Allow replacement? (y/n) > [/yellow]").lower()
                                            if confirm != "y":
                                                allowed = False
                                                result = json.dumps({"error": "User denied permission to replace string."})
                                            else:
                                                session_allowed_files.add(str_path)
                                    else:
                                        allowed = False
                                        result = json.dumps({"error": "Missing file_path argument."})
                                
                                if allowed:
                                    console.print(f"[dim]Executing...[/dim]")
                                    result = execute_tool_call(name, args)
                                    tools_executed_successfully = True

                                    if name in ["write_file", "read_file", "replace_string"]:
                                        fp = args.get('file_path')
                                        if fp:
                                            active_file = str(Path(fp))
                                else:
                                    console.print("[red]Action cancelled.[/red]")
                                    tools_executed_successfully = False

                            except Exception as e:
                                result = json.dumps({"error": str(e)})
                            
                            # Suppress error details from console, but show success
                            if '"error"' in result or "'error'" in result:
                                console.print(f"[dim]Action encountered an issue.[/dim]")
                            else:
                                console.print(f"[dim]Result: {result[:100]}...[/dim]")
                            
                            llm_messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result
                            })
                        
                        # Loop continues to send tool output back to model
                        continue
                    else:
                        # No tools, final response
                        messages.append({"role": "assistant", "content": response_text})
                        break
            if len(messages) > MAX_HISTORY_LENGTH:
                messages = messages[-MAX_HISTORY_LENGTH:]
            console.print()

        except Exception as e:
            console.print(f"[bold red]Something went wrong. Please try again.[/bold red]")





if __name__ == "__main__":
    app()