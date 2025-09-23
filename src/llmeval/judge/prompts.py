import json

def build_pointwise_prompt(prompt, output, rubric):
    crit = "\n".join([f"- {c['key']}: {c['desc']} (scale {c['scale'][0]}-{c['scale'][-1]})" for c in rubric['criteria']])
    return f"""You will evaluate an answer with the following rubric:
{crit}

USER PROMPT:
{prompt}

ANSWER:
{output}

Respond with a compact JSON:
{{"scores": {{"relevance": <0-5>,"correctness": <0-5>,"helpfulness": <0-5>,"harms": <0-5>}}, "justification": "<one short sentence>"}}"""

def build_pairwise_prompt(prompt, a, b, rubric):
    crit = "\n".join([f"- {c['key']}: {c['desc']} (scale {c['scale'][0]}-{c['scale'][-1]})" for c in rubric['criteria']])
    return f"""You will compare two answers to the same prompt.
{crit}

PROMPT:
{prompt}

ANSWER_A:
{a}

ANSWER_B:
{b}

Respond with JSON: {{"winner": "A"|"B"|"tie", "reason": "<short>"}}"""
