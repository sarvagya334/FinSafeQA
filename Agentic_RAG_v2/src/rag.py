import os
import requests
from typing import Callable
from src.config import NVIDIA_BASE_URL, NVIDIA_MODEL, DEFAULT_TEMPERATURE

def init_llm(api_key: str, model: str = NVIDIA_MODEL) -> Callable[[str], str]:
    """
    Initializes a callable interface to the NVIDIA LLM API.
    This callable will be passed directly into the LangGraph Agent nodes.
    """
    base_url = os.environ.get("NVIDIA_BASE_URL", NVIDIA_BASE_URL).strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def llm(prompt: str, system_prompt: str = "You are a specialized Cross-Border Financial AI.") -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": 1500,
        }
        
        try:
            r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM API Error: {str(e)}"

    return llm