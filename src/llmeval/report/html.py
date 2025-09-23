from jinja2 import Template
import pandas as pd, json, os

TPL = """<!doctype html>
<html><head><meta charset="utf-8"><title>LLM Eval Report</title>
<style>body{font-family:system-ui,Arial;margin:24px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px 10px}</style>
</head><body>
<h1>LLM Evaluation Report</h1>
<p>Model(s): {{ models }}</p>
<h2>Aggregate Metrics</h2>
<table>
<tr><th>metric</th><th>value</th></tr>
{% for k,v in aggregates.items() %}
<tr><td>{{k}}</td><td>{{"%.4f"%v if v is not none else ""}}</td></tr>
{% endfor %}
</table>

<h2>Per-Item Scores</h2>
<table>
<tr><th>ID</th><th>relevance</th><th>semantic</th><th>lex_f1</th><th>tox_hits</th><th>judge_relevance</th><th>judge_correctness</th></tr>
{% for r in rows %}
<tr>
<td>{{r["id"]}}</td>
<td>{{"%.3f"%r.get("relevance",0) if r.get("relevance") is not none else ""}}</td>
<td>{{"%.3f"%r.get("semantic",0) if r.get("semantic") is not none else ""}}</td>
<td>{{"%.3f"%r.get("lexical_f1",0) if r.get("lexical_f1") is not none else ""}}</td>
<td>{{r.get("toxic_hits","")}}</td>
<td>{{r.get("judge_scores",{}).get("relevance","")}}</td>
<td>{{r.get("judge_scores",{}).get("correctness","")}}</td>
</tr>
{% endfor %}
</table>
</body></html>"""

def render_report(rows, aggregates, models, out_path):
    html = Template(TPL).render(rows=rows, aggregates=aggregates, models=", ".join(sorted(models)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
