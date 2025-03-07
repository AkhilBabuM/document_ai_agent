
import httpx

async def query_llm(prompt: str) -> str:
    """Query an llm with prompt, and return the response."""
    response = await query_ollama(prompt=prompt)
    return response

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
                return response_data["response"].strip()
            else:
                return f"Error: Unexpected response format {response_data}"
    except Exception as e:
        return f"Error in LLM query: {str(e)}"