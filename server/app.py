import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from litellm import completion, acompletion
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI(title="Vorp Backend")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    # Security: Check Access Token
    expected_token = os.getenv("VORP_ACCESS_TOKEN")
    if expected_token:
        auth_header = req.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Invalid Vorp Access Token")
    
    # 1. Validation
    # Ensure the model is one we support/allow
    allowed_prefixes = ["groq/", "gemini/"]
    if not any(request.model.startswith(prefix) for prefix in allowed_prefixes):
         # Allow bare model names if they map to our providers, otherwise basic check
         if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
             raise HTTPException(status_code=500, detail="Server misconfiguration: No API keys available.")

    try:
        # 2. Forward to LLM Provider
        # litellm will automatically use GROQ_API_KEY or GEMINI_API_KEY from os.environ
        response = await acompletion(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            stream=True 
        )

        # 3. Stream Response
        async def stream_generator():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "active", "service": "vorp-backend"}
