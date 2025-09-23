import importlib

class LocalProvider:
    def __init__(self, judge_callable='', embed_callable=''):
        self.judge_fn = self._resolve(judge_callable) if judge_callable else None
        self.embed_fn = self._resolve(embed_callable) if embed_callable else None

    def _resolve(self, dotted):
        mod, fn = dotted.rsplit('.',1)
        return getattr(importlib.import_module(mod), fn)

    def judge(self, prompt: str, rubric_json: dict):
        if not self.judge_fn:
            raise RuntimeError('No local judge callable configured')
        return self.judge_fn(prompt, rubric_json)

    def embed(self, texts):
        if not self.embed_fn:
            raise RuntimeError('No local embed callable configured')
        return self.embed_fn(texts)

    def moderate(self, text: str):
        return {}
