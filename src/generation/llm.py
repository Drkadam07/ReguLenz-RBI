import httpx
from src.config import settings
from src.generation.prompt import SYSTEM_PROMPT

class OllamaLLMClient:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.LLM_MODEL):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self.model = model

    async def generate_answer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1
            }
        }
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            return f"Error communicating with local LLM engine: {str(e)}"