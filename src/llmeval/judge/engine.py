import json, random
from .prompts import build_pointwise_prompt, build_pairwise_prompt

class JudgeEngine:
    def __init__(self, provider, rubric):
        self.provider = provider
        self.rubric = rubric

    def score_pointwise(self, prompt, output):
        jp = build_pointwise_prompt(prompt, output, self.rubric)
        return self.provider.judge(jp, self.rubric)

    def score_pairwise(self, prompt, a, b):
        jp = build_pairwise_prompt(prompt, a, b, self.rubric)
        return self.provider.judge(jp, self.rubric)

    def calibrate(self, anchors):
        # simple check to catch inverted judges
        ok, total = 0, 0
        for a in anchors:
            res = self.score_pairwise(a['prompt'], a['good'], a['bad'])
            win = res.get('winner','tie').lower()
            ok += 1 if win=='a' else 0
            total += 1
        return {"anchor_accuracy": ok/total if total else None}
