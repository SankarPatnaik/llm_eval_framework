import os
import json
import requests


class GorqProvider:
    """Provider wrapper for the Gorq (Groq-compatible) HTTP API.

    The API follows the OpenAI-compatible schema exposed by Groq, so we reuse the
    same payload shapes as the OpenAI provider. Only the authentication header
    and default base URL differ.
    """

    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        embedding_model: str | None = "text-embedding-3-large",
        moderation: bool = False,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.moderation = moderation
        self.api_key = os.getenv("GORQ_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "GORQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def judge(self, prompt: str, rubric_json: dict):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rubric_json.get("system", "")},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    def embed(self, texts):
        if not self.embedding_model:
            raise RuntimeError("embedding_model not configured for GorqProvider")
        payload = {"model": self.embedding_model, "input": texts}
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        return [item["embedding"] for item in body.get("data", [])]

    def moderate(self, text: str):
        if not self.moderation:
            return {}
        payload = {"model": "omni-moderation-latest", "input": text}
        response = requests.post(
            f"{self.base_url}/moderations",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
