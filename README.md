# LLM Evaluation Framework (No Hugging Face)

A lightweight, modular framework to evaluate LLM generations **without** Hugging Face.
It supports:
- **Relevance**: embedding cosine (via provider adapters) + lexical F1 (ROUGE-lite).
- **LLM-as-a-Judge**: pointwise and pairwise rubric scoring, calibrated with anchor items.
- **Bias Audits**: counterfactual swapping, WEAT-like effect size using embeddings, group fairness deltas.
- **Toxicity-lite**: simple pattern list + optional provider moderation API.
- **Self-Consistency**: variance across `n` samples.
- **JSON/CSV outputs** and **HTML report**.

> Providers supported: OpenAI, Google Gemini, generic HTTP (bring your own), local function hook.
> Set keys via env vars; the CLI never requires Hugging Face.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Evaluate example dataset with OpenAI (set OPENAI_API_KEY first)
python -m llmeval.runners.eval --config config.yaml
```

### Using Google Gemini

1. Create an API key from [Google AI Studio](https://aistudio.google.com/).
2. Export it as `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
3. Switch the config to use the Gemini adapter:

```yaml
provider: gemini
gemini:
  model: gemini-2.5-flash
  embedding_model: text-embedding-004
```

You can override `base_url`, `generation_config`, or `safety_settings` in
`config.yaml` if you are routing requests through a proxy or need custom
Gemini safety parameters.

### Open-source or self-hosted models

If you are running an open-source model (e.g. via vLLM, TGI, Ollama) you can
either expose an HTTP endpoint and use the `generic` provider, or plug in a
Python callable through the `local` provider. Both approaches let you reuse the
same evaluation pipeline without depending on proprietary APIs.

## Data format

`data/examples/qa.jsonl`:
```json
{"id":"ex1","prompt":"What is 2+2?","reference":"4","groups":{"gender":"neutral"}}
```
`generations.jsonl` (your model's outputs):
```json
{"id":"ex1","model":"my-llm","output":"It is 4.","meta":{"temp":0.2}}
```

## Outputs

- `reports/summary.json` & `reports/summary.csv`
- `reports/report.html`

## License

MIT
