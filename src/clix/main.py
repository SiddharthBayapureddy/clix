# App lives here 

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path
from dotenv import load_dotenv
import json # Need this to persist chat history in .json files is required by user
import typer # For the CLI


# Rich - Markdown language and colors
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner # Spinnig loading animation

# LiteLLM - One lib to access any llm 
from litellm import completion

load_dotenv()


# Initliazining app
app = typer.Typer()
console = Console()

# MACROS
CHAT_HISTORY = Path.home() / ".clix_chat_history.json"

# Default model
MODEL_NAME = "groq/llama-3.1-8b-instant"  # Using this for now, later implement customizability

SUPPORTED_MODELS = [
    {"name": "Llama 3.1 8B (Fast)", "id": "groq/llama-3.1-8b-instant", "provider": "Groq"},
    {"name": "Llama 3.3 70B (Smart)", "id": "groq/llama-3.3-70b-versatile", "provider": "Groq"},
    {"name": "Gemini 2.5 Flash", "id": "gemini/gemini-2.5-flash", "provider": "Google"},
    {"name": "Gemini 2.5 Pro", "id": "gemini/gemini-2.5-pro", "provider": "Google"},
]


# Helper functions
def load_history():
    """ Loads chat history from the file if exists """  # Called Docstrings, easy to find function function
    if CHAT_HISTORY.exists():
        try:
            with open(CHAT_HISTORY , "r") as file:
                return json.load(file) # Returns the chat history
        except json.JSONDecodeError:
            return []

    return []


def save_history(messages):
    """Saves the current session to disk for persistence"""
    with open(CHAT_HISTORY, "w") as file:
        json.dump(messages , file)


def delete_history():
    """Deletes the chat history"""
    if CHAT_HISTORY.exists():
        os.remove(CHAT_HISTORY)


######################
# THE MAIN CHAT LOOP #
#######################


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context, 
    # To change model
    model: str = typer.Option(MODEL_NAME, "--model", "-m", help="Model to use"),
):
    # If the user ran "clix --help", don't start the chat
    if ctx.invoked_subcommand is not None:
        return

    # Check for API Key
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No API Key found in .env file.")
        raise typer.Exit()

    # Load history
    messages = load_history()
    if messages:
        console.print("[dim green]↻ Resuming previous session...[/dim green]")
        console.print("[bold green] Session Restored! [/bold green]")
    
    console.print("[bold cyan]Clix Online.[/bold cyan]")

    
    
    while True:
        try:

            user_input = console.input("[bold green]You > [/bold green]")
            
            # If "/exit" , session history isn't saved -> Chat gets deleted
            if user_input.strip() == "/exit":
                delete_history()
                console.print("[bold red]Session deleted. Peace![/bold red]")
                break
            
            # if "/exit-v" , session history is saved -> Chat is stored
            if user_input.strip() == "/exit-v":
                save_history(messages)
                console.print("[bold green]Session saved. Peace![/bold green]")
                break
            
            # if "/clear" , clears the terminal
            if user_input.strip() == "/clear":
                console.clear()
                continue


            # Saving each chat
            messages.append({"role": "user", "content": user_input})


            response_text = ""
            
            # --- AUTO-FIX: Ensure Groq models have the prefix ---
            active_model = model
            if not active_model.startswith("groq/") and not active_model.startswith("gemini/") and not active_model.startswith("gpt-"):
                active_model = f"groq/{active_model}"
            # ----------------------------------------------------
            
            with Live(console=console, refresh_per_second=20) as live:

                # Table grid for better printing
                grid = Table.grid(padding=(0, 1)) 
                grid.add_column(style="bold blue", no_wrap=True)  # Col1 : Clix > (Prevent wrapping)
                grid.add_column()  # Col2 : Streaming markdown
                
                # Add the row with both pieces of content
                grid.add_row(
                    "Clix >" , Spinner("dots", style="bold cyan", text="Thinking...")
                )
                live.update(grid)


                response = completion(
                    model=active_model, # Using the prefixed variable
                    messages=messages,
                    stream=True
                )
                
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    response_text += content

                    # Re-create grid with new text
                    grid = Table.grid(padding=(0, 1)) 
                    grid.add_column(style="bold blue", no_wrap=True)
                    grid.add_column()
                    
                    grid.add_row("Clix >", Markdown(response_text))
                    live.update(grid)
            

            # Append AI response to history
            messages.append({"role": "assistant", "content": response_text})
            console.print() # Add a newline for spacing

        except KeyboardInterrupt:
            # Ctrl+C handled
            console.print("\n[yellow]Exiting without saving...[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")



# Listing available models
@app.command(name="models")
def list_models():
    """
    Show available AI models.
    """
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Model ID (Use this)", style="magenta")
    table.add_column("Provider", style="green")

    for model in SUPPORTED_MODELS:
        table.add_row(model["name"], model["id"], model["provider"])

    console.print(table)
    console.print("\n[dim]Usage: clix --model <Model ID>[/dim]")

if __name__ == "__main__":
    app()