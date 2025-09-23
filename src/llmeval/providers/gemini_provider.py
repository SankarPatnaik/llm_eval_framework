import json
import os
from typing import Any, Dict, List, Optional

import requests


class GeminiProvider:
    """Provider wrapper around Google's Gemini API.

    The provider mirrors the :class:`OpenAIProvider` interface, supporting
    judge, embed and (noop) moderation helpers so it can drop in wherever the
    OpenAI adapter is currently used.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.generation_config = generation_config or {}
        self.safety_settings = safety_settings or []
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Helpers
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    def _params(self) -> Optional[Dict[str, str]]:
        # The API accepts both header and query params for the key. To avoid
        # leaking keys in URLs we prefer a header, but keep params for
        # compatibility with self-hosted proxies that expect it there.
        if not self.api_key:
            return None
        return {"key": self.api_key}

    def _model_path(self, name: str) -> str:
        return name if name.startswith("models/") else f"models/{name}"

    # ------------------------------------------------------------------
    # Public interface
    def judge(self, prompt: str, rubric_json: Dict[str, Any]) -> Dict[str, Any]:
        """Call Gemini to evaluate a prompt using the supplied rubric."""

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        if rubric_json.get("system"):
            payload["systemInstruction"] = {
                "parts": [{"text": rubric_json["system"]}],
            }
        if self.generation_config:
            payload["generationConfig"].update(self.generation_config)
        if self.safety_settings:
            payload["safetySettings"] = self.safety_settings

        url = f"{self.base_url}/models/{self.model}:generateContent"
        response = requests.post(
            url,
            headers=self._headers(),
            params=self._params(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from exc
        return json.loads(text)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/models/{self.embedding_model}:batchEmbedContents"
        payload = {
            "model": self._model_path(self.embedding_model),
            "requests": [
                {"content": {"parts": [{"text": text}]}} for text in texts
            ],
        }
        response = requests.post(
            url,
            headers=self._headers(),
            params=self._params(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        return [emb.get("values", []) for emb in embeddings]

    def moderate(self, text: str) -> Dict[str, Any]:
        # Gemini safety checks are handled via ``safetySettings`` passed during
        # ``judge`` calls, so we no-op here to mirror the GenericProvider.
        return {}
