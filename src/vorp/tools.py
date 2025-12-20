import os
import json
import subprocess
from pathlib import Path

# Tool Definitions
def get_tool_definitions():
    """
    Returns the JSON schema definitions for the tools available to the LLM.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads the content of a specific file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute or relative path to the file."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Creates or overwrites a file with the provided content. WARNING: This overwrites the ENTIRE file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path where the file should be written."
                        },
                        "content": {
                            "type": "string",
                            "description": "The complete content to write into the file."
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "replace_string",
                "description": "Replaces a specific string in a file with a new string. Use this for targeted edits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to modify."
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact string to be replaced."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The new string to replace the old string with."
                        }
                    },
                    "required": ["file_path", "old_string", "new_string"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Permanently deletes a file from the filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to be deleted."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Lists all files within a specified directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "The directory to list. Defaults to the current working directory."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Executes a shell command on the host system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute (e.g., 'pip install', 'ls -la')."
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]

# Tool Implementations

def read_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {e}"})

def write_file(file_path: str, content: str):
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"success": True, "message": f"Successfully wrote to {file_path}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to write file: {e}"})

def replace_string(file_path: str, old_string: str, new_string: str):
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        count = content.count(old_string)
        if count == 0:
            return json.dumps({"error": "old_string not found in file."})
        elif count > 1:
            return json.dumps({"error": f"Ambiguous: old_string found {count} times. Please provide more context in old_string to make it unique."})
        
        new_content = content.replace(old_string, new_string)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return json.dumps({"success": True, "message": f"Successfully replaced string in {file_path}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to replace string: {e}"})

def delete_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})
    try:
        os.remove(path)
        return json.dumps({"success": True, "message": f"Successfully deleted {file_path}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to delete file: {e}"})

def list_files(directory: str = "."):
    try:
        p = Path(directory)
        if not p.exists():
             return json.dumps({"error": f"Directory not found: {directory}"})
             
        files = [str(f.name) for f in p.glob("*")] # Just names usually better for LLM token count, or relative paths
        return json.dumps(files)
    except Exception as e:
        return json.dumps({"error": f"Failed to list files: {e}"})

def run_shell_command(command: str):
    try:
        # Use shell=True for flexibility, but capture output safely
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=120 # Prevent hanging commands
        )
        return json.dumps({
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        })
    except subprocess.TimeoutExpired:
         return json.dumps({"error": "Command timed out."})
    except Exception as e:
        return json.dumps({"error": f"Command failed: {e}"})

# Dispatcher
def execute_tool_call(tool_name: str, arguments: dict):
    """
    Dispatches the tool execution to the appropriate function.
    """
    if tool_name == "read_file":
        return read_file(arguments.get("file_path"))
    elif tool_name == "write_file":
        return write_file(arguments.get("file_path"), arguments.get("content"))
    elif tool_name == "replace_string":
        return replace_string(arguments.get("file_path"), arguments.get("old_string"), arguments.get("new_string"))
    elif tool_name == "delete_file":
        return delete_file(arguments.get("file_path"))
    elif tool_name == "list_files":
        return list_files(arguments.get("directory", "."))
    elif tool_name == "run_shell_command":
        return run_shell_command(arguments.get("command"))
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})