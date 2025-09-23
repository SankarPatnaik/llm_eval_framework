import requests, json

class GenericHTTPProvider:
    def __init__(self, judge_url='', embed_url='', headers=None):
        self.judge_url = judge_url
        self.embed_url = embed_url
        self.headers = headers or {}

    def judge(self, prompt: str, rubric_json: dict):
        payload = {"prompt": prompt, "rubric": rubric_json}
        r = requests.post(self.judge_url, headers=self.headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        return r.json()

    def embed(self, texts):
        if not self.embed_url:
            raise RuntimeError("embed_url not set for GenericHTTPProvider")
        payload = {"texts": texts}
        r = requests.post(self.embed_url, headers=self.headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        return r.json().get("embeddings", [])

    def moderate(self, text: str):
        return {}
