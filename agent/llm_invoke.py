import httpx
from enum import Enum
from google import genai
from google.genai import types
from .utils import remove_newline_characters
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")

class LLMS(str, Enum):
    ollama = "OLLAMA"
    gemini = "GEMINI"

async def query_llm(prompt: str, llm_type: LLMS = LLMS.gemini) -> str:
    """Query an llm with prompt, and return the response."""
    llm_functions = {
        LLMS.ollama: query_ollama,
        LLMS.gemini: query_gemini,
    }

    if llm_type not in llm_functions:
        raise ValueError(f"Invalid LLM type: {llm_type}")

    response = await llm_functions[llm_type](prompt)
    return response


async def query_gemini(prompt: str) -> str:
    client = genai.Client(
        api_key=GEMINI_KEY,
    )

    model = "gemini-2.0-flash"
    # model = "gemini-2.0-flash-lite"
    # model = "gemini-2.0-pro-exp-02-05"d

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        response += chunk.text

    cleaned_response = remove_newline_characters(text=response).strip()
    return cleaned_response


async def query_ollama(prompt: str) -> str:
    """Query Ollama's locally running model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",  # Ollama's API
                json={"model": "qwen2.5", "prompt": prompt, "stream": False},
                timeout=60,
            )
            response_data = response.json()

            if "response" in response_data:
                response = response_data["response"]
                cleaned_response = remove_newline_characters(text=response).strip()
                return cleaned_response
            else:
                return f"Error: Unexpected response format {response_data}"
    except Exception as e:
        return f"Error in LLM query: {str(e)}"
