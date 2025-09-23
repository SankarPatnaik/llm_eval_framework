# LLM Evaluation Framework (No Hugging Face)

This project helps you check how well a Large Language Model (LLM) is doing
without needing any Hugging Face tools. The framework can score answers for
quality, fairness, safety, and consistency, and it works with popular hosted
LLM providers or with models you run yourself.

If you are new to Python or APIs, do not worry—this guide walks you through the
basics step by step.

---

## 1. What you can measure

The framework can calculate several types of checks:

| Check | What it means |
| --- | --- |
| **Relevance** | Compares the model answer to the reference answer using embeddings and keyword overlap. |
| **LLM-as-a-Judge** | Sends the answer to another model that scores it with a rubric (either one answer at a time or comparing two answers). |
| **Bias Audits** | Looks for unfair differences between demographic groups. |
| **Toxicity-lite** | Flags simple unsafe phrases or uses a provider safety API if you enable it. |
| **Self-Consistency** | Runs the same prompt multiple times and measures how different the answers are. |
| **Reports** | Saves the results as JSON, CSV, and a simple HTML dashboard. |

---

## 2. Before you start

You only need a few things in place:

1. **Python 3.10 or newer** installed on your machine.
2. **An API key** for the model provider you want to use (OpenAI, Google Gemini,
   or another HTTP endpoint). If you are running a local model, you will point
the framework to that instead.
3. **A small dataset** that contains the prompts you want to evaluate and, if you
   have them, the reference answers.

If you have never created a virtual environment before, the instructions below
include the exact commands to copy and paste.

---

## 3. Set up the project (one time)

Open a terminal or command prompt in the project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Tip:** Each time you come back to the project, run `source .venv/bin/activate`
> (or the Windows command above) so that Python uses the right dependencies.

---

## 4. Add your API keys

Different providers use different environment variables. Pick the one that
matches the service you are using and set it in the same terminal session where
you will run the evaluation.

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."      # Windows PowerShell: $Env:OPENAI_API_KEY = "sk-..."
```

### Google Gemini

1. Create a key at [Google AI Studio](https://aistudio.google.com/).
2. Set it as `GEMINI_API_KEY` (or `GOOGLE_API_KEY`):

   ```bash
   export GEMINI_API_KEY="your-key"  # Windows PowerShell: $Env:GEMINI_API_KEY = "your-key"
   ```

### Other providers or local models

You can point the framework to any HTTP endpoint with the `generic` provider, or
connect a Python function using the `local` provider. See the comments inside
`config.yaml` for the fields you can change (such as base URLs, headers, or
model names).

---

## 5. Prepare your data

The project includes an example dataset in `data/examples/qa.jsonl`. Each line
is a JSON object:

```json
{"id": "ex1", "prompt": "What is 2+2?", "reference": "4", "groups": {"gender": "neutral"}}
```

If you already have model outputs, store them in another JSONL file (for
example `generations.jsonl`) with this shape:

```json
{"id": "ex1", "model": "my-llm", "output": "It is 4.", "meta": {"temp": 0.2}}
```

> **Need help creating these files?** You can open them in any text editor and
> paste one JSON object per line. The `id` values must match so that the
> framework can pair prompts with model answers.

---

## 6. Run your first evaluation

Once the virtual environment is active and the API key is set, run:

```bash
python -m llmeval.runners.eval --config config.yaml
```

What happens next:

1. The script reads `config.yaml` to see which provider to use and which checks
   are turned on.
2. It loads your prompts and generations from the paths listed in the config.
3. It calls the provider to score the answers (or uses the local function you
   supplied).
4. It saves the results into the `reports/` folder.

You will see progress information in the terminal while it runs. Depending on
how many prompts you have, it may take a few minutes.

---

## 7. Read the reports

After the command finishes, look in the `reports/` folder. The most useful
outputs are:

- `reports/summary.json` – machine-readable metrics.
- `reports/summary.csv` – spreadsheet-friendly version.
- `reports/report.html` – open in a browser for a quick visual overview.

If you do not see the folder, double-check that the command in step 6 completed
without errors.

---

## 8. Customize the configuration

Open `config.yaml` in a text editor. Some common tweaks:

- **Switch providers:** change the `provider` field to `openai`, `gemini`,
  `generic`, or `local`.
- **Update model names:** each provider section (for example the `gemini:` block)
  has fields for the model and embedding model names.
- **Change datasets:** adjust the paths under `data:` to point to your own JSONL
  files.
- **Turn checks on or off:** toggle values such as `enable_bias_audit: true` or
  `false`.

Always save the file before running the evaluation again.

---

## 9. Troubleshooting

| Problem | How to fix it |
| --- | --- |
| Command not found | Make sure your virtual environment is activated. Run `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (Windows PowerShell). |
| API authentication error | Double-check the API key value and that it was exported in the same terminal session where you run the evaluation. |
| Unicode or JSON errors | Inspect the line number mentioned in the error message. Often a missing quote or an extra comma in your JSONL file is the cause. |
| Reports folder is empty | Ensure the script finished without errors. Review the terminal output for the first error message. |

If you get stuck, re-run the command with the `-v` flag for more detail:

```bash
python -m llmeval.runners.eval --config config.yaml -v
```

---

## 10. License

MIT
