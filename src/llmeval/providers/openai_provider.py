import os, requests, json

class OpenAIProvider:
    def __init__(self, model='gpt-4o-mini', embedding_model='text-embedding-3-large', moderation=True):
        self.model = model
        self.embedding_model = embedding_model
        self.moderation = moderation
        self.api_key = os.getenv('OPENAI_API_KEY','')
        self.base_url = os.getenv('OPENAI_BASE','https://api.openai.com/v1')

    def judge(self, prompt: str, rubric_json: dict):
        # simple JSON-instruction call
        payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content":rubric_json.get("system","")},
                {"role":"user","content":prompt}
            ],
            "response_format": {"type":"json_object"}
        }
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type":"application/json"},
                          data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        js = r.json()
        txt = js["choices"][0]["message"]["content"]
        return json.loads(txt)

    def embed(self, texts):
        payload = {"model": self.embedding_model, "input": texts}
        r = requests.post(f"{self.base_url}/embeddings",
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type":"application/json"},
                          data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        js = r.json()
        return [item["embedding"] for item in js["data"]]

    def moderate(self, text: str):
        if not self.moderation:
            return {}
        payload = {"model":"omni-moderation-latest","input":text}
        r = requests.post(f"{self.base_url}/moderations",
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type":"application/json"},
                          data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        return r.json()
