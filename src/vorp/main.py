# App lives here 

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
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

load_dotenv()


# Initialization
app = typer.Typer()
console = Console()

# Configuration
CHAT_HISTORY = Path.home() / ".vorp_chat_history.json"
MODEL_NAME = "groq/llama-3.1-8b-instant" 
MAX_HISTORY_LENGTH = 30
VORP_BACKEND_URL = "https://vorp-sigma.vercel.app/chat"
DEFAULT_PUBLIC_ACCESS_TOKEN = "sk-vorp-public-beta" # Hardcoded public token

SUPPORTED_MODELS = [
    # Groq / Llama
    {"name": "Llama 3.3 70B (Versatile)", "id": "groq/llama-3.3-70b-versatile", "provider": "Groq"},
    {"name": "Llama 3.1 8B (Instant)", "id": "groq/llama-3.1-8b-instant", "provider": "Groq"},
    {"name": "Llama 4 Maverick 17B", "id": "groq/meta-llama/llama-4-maverick-17b-128e-instruct", "provider": "Groq"},
    {"name": "Llama 4 Scout 17B", "id": "groq/meta-llama/llama-4-scout-17b-16e-instruct", "provider": "Groq"},
    {"name": "Llama Guard 4 12B", "id": "groq/meta-llama/llama-guard-4-12b", "provider": "Groq"},
    
    # Groq / Other
    {"name": "DeepSeek R1 Distill 70B", "id": "groq/deepseek-r1-distill-llama-70b", "provider": "Groq"},
    {"name": "Gemma 2 9B", "id": "groq/gemma2-9b-it", "provider": "Groq"},
    {"name": "Mixtral 8x7B", "id": "groq/mixtral-8x7b-32768", "provider": "Groq"},
    {"name": "Qwen 3 32B", "id": "groq/qwen/qwen3-32b", "provider": "Groq"},
    {"name": "Allam 2 7B", "id": "groq/allam-2-7b", "provider": "Groq"},
    
    # Groq / Experimental
    {"name": "Groq Compound", "id": "groq/groq/compound", "provider": "Groq"},
    {"name": "Groq Compound Mini", "id": "groq/groq/compound-mini", "provider": "Groq"},
    
    # Moonshot
    {"name": "Kimi k2 Instruct", "id": "groq/moonshotai/kimi-k2-instruct", "provider": "Groq"},
    
    # OpenAI (via Groq/Litellm mapping if applicable, otherwise assume standard routing)
    {"name": "GPT-OSS 120B", "id": "groq/openai/gpt-oss-120b", "provider": "Groq"},
    {"name": "GPT-OSS 20B", "id": "groq/openai/gpt-oss-20b", "provider": "Groq"},

    # Google
    {"name": "Gemini 2.5 Flash", "id": "gemini/gemini-2.5-flash", "provider": "Google"},
    {"name": "Gemini 2.5 Pro", "id": "gemini/gemini-2.5-pro", "provider": "Google"},
]


# Utils
def load_history():
    """Loaded chat history from file if available."""
    if CHAT_HISTORY.exists():
        try:
            with open(CHAT_HISTORY , "r") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data.get("messages", []), data.get("rag_enabled", False)
                elif isinstance(data, list):
                    return data, False
        except json.JSONDecodeError:
            return [], False

    return [], False


def save_history(messages, rag_enabled):
    """Persisted current session to disk."""
    with open(CHAT_HISTORY, "w") as file:
        json.dump({"messages": messages, "rag_enabled": rag_enabled}, file)


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
    
    from litellm import completion

    # Session recovery
    messages, rag_enabled = load_history()
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
    else:
        console.print(f"[bold green]-> STATUS:[/bold green] [dim]ONLINE (Local Mode)[/dim]")
        
    console.print("[dim]──────────────────────────[/dim]")

    # RAG State
    active_project_path = None

    if rag_enabled:
        console.print("[dim green]RAG Mode: ENABLED[/dim green]")

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
                save_history(messages, rag_enabled)
                console.print("[bold green]Session saved. Peace![/bold green]")
                break
            
            # Command: /clear
            if user_input_clean == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
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
                         count = rag.index_codebase(str(idx_path))
                    
                    active_project_path = str(idx_path)
                    console.print(f"[bold green]✓ Successfully indexed {count} files.[/bold green]")
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

            # Chat Logic
            messages.append({"role": "user", "content": user_input})
            if len(messages) > MAX_HISTORY_LENGTH:
                messages = messages[-MAX_HISTORY_LENGTH:]


            response_text = ""
            
            # Context Injection
            llm_messages = messages.copy()
            
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
            
            with Live(console=console, refresh_per_second=20) as live:
                grid = Table.grid(padding=(0, 1)) 
                grid.add_column(style="bold blue", no_wrap=True)
                grid.add_column()
                
                grid.add_row(
                    "Vorp >" , Spinner("dots", style="bold cyan", text="Thinking...")
                )
                live.update(grid)

                if cloud_mode:
                    # Cloud Mode: Proxy through Backend
                    try:
                        payload = {
                            "model": active_model,
                            "messages": llm_messages,
                            "stream": True
                        }
                        headers = {"Authorization": f"Bearer {access_token_for_backend}"}
                        
                        with requests.post(VORP_BACKEND_URL, json=payload, headers=headers, stream=True, timeout=60) as r:
                            r.raise_for_status()
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
                else:
                    # Local Mode: Use litellm directly
                    response = completion(
                        model=active_model,
                        messages=llm_messages,
                        stream=True
                    )
                    
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        response_text += content

                        grid = Table.grid(padding=(0, 1)) 
                        grid.add_column(style="bold blue", no_wrap=True)
                        grid.add_column()
                        
                        grid.add_row("Vorp >", Markdown(response_text))
                        live.update(grid)
            

            messages.append({"role": "assistant", "content": response_text})
            if len(messages) > MAX_HISTORY_LENGTH:
                messages = messages[-MAX_HISTORY_LENGTH:]
            console.print()

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")





if __name__ == "__main__":
    app()