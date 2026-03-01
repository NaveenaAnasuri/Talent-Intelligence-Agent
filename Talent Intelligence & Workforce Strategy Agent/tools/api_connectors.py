from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    ENABLE_NEO4J: bool = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


SETTINGS = Settings()


@dataclass
class LLMResponse:
    text: str
    used_llm: bool
    meta: Dict[str, Any]


class LLMClient:
    """
    Ollama chat client (free, local).
    Uses /api/chat for better instruction following.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, timeout_s: int = 120):
        self.model = model or SETTINGS.OLLAMA_MODEL
        self.base_url = (base_url or SETTINGS.OLLAMA_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def is_reachable(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, system: str, user: str) -> LLMResponse:
        if not self.is_reachable():
            return LLMResponse(
                text="Ollama is not reachable. Start it with: `ollama serve` and ensure model exists: `ollama pull llama3`.",
                used_llm=False,
                meta={"provider": "ollama", "model": self.model, "reachable": False},
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            return LLMResponse(
                text=text,
                used_llm=True,
                meta={"provider": "ollama", "model": self.model, "reachable": True},
            )
        except Exception as e:
            return LLMResponse(
                text=f"LLM call failed: {e}",
                used_llm=False,
                meta={"provider": "ollama", "model": self.model, "reachable": True},
            )