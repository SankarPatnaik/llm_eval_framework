import argparse
import json
from pathlib import Path
from textwrap import dedent

import yaml


TASK_LIBRARY = {
    "qa": {
        "description": "General question answering across factual domains.",
        "dataset": [
            {
                "id": "qa-1",
                "prompt": "What is the capital of France?",
                "reference": "Paris is the capital of France.",
                "groups": {"geography": "global"},
            },
            {
                "id": "qa-2",
                "prompt": "Name one benefit of regular exercise.",
                "reference": "Regular exercise supports cardiovascular health.",
                "groups": {"topic": "health"},
            },
        ],
        "generations": [
            {
                "id": "qa-1",
                "model": "example-llm",
                "output": "Paris is the capital city of France.",
            },
            {
                "id": "qa-2",
                "model": "example-llm",
                "output": "Exercise improves heart health and stamina.",
            },
        ],
        "rubric": {
            "name": "QA Baseline Rubric",
            "criteria": [
                {
                    "key": "relevance",
                    "desc": "Does the answer address the user question?",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "accuracy",
                    "desc": "Is the information factually correct?",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "clarity",
                    "desc": "Is the response easy to understand and sufficiently detailed?",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
            ],
            "system": "You are an impartial QA evaluator. Score each criterion and return JSON only.",
        },
        "anchors": [
            {
                "prompt": "What is 2+2?",
                "good": "The answer is 4.",
                "bad": "It's probably 5 or maybe 6.",
            },
            {
                "prompt": "Translate 'hola' to English.",
                "good": "It means 'hello'.",
                "bad": "It translates to 'goodbye'.",
            },
        ],
        "groups": {
            "demographic_axes": {
                "geography": ["north_america", "europe", "south_asia"],
                "experience_level": ["novice", "expert"],
            },
            "counterfactual_terms": {
                "geography": [["north american", "european"], ["european", "south asian"]],
                "experience_level": [["junior", "senior"], ["novice", "expert"]],
            },
        },
    },
    "summarization": {
        "description": "Summaries of longer passages, reports, or transcripts.",
        "dataset": [
            {
                "id": "sum-1",
                "prompt": "Summarize the following text in 2 sentences: A new community garden opened downtown with volunteers from across the city...",
                "reference": "Volunteers launched a downtown community garden to expand access to fresh produce and offer public workshops.",
                "groups": {"domain": "civic"},
            },
            {
                "id": "sum-2",
                "prompt": "Produce a concise summary of this quarterly revenue report for the leadership team...",
                "reference": "The company beat revenue targets through subscription growth but warned about rising support costs.",
                "groups": {"audience": "executive"},
            },
        ],
        "generations": [
            {
                "id": "sum-1",
                "model": "example-llm",
                "output": "City volunteers opened a community garden to share fresh produce and teach gardening classes.",
            },
            {
                "id": "sum-2",
                "model": "example-llm",
                "output": "Revenue exceeded goals thanks to subscriptions, though support expenses are climbing.",
            },
        ],
        "rubric": {
            "name": "Summarization Baseline Rubric",
            "criteria": [
                {
                    "key": "coverage",
                    "desc": "Captures the most important points without omissions.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "faithfulness",
                    "desc": "Does not introduce facts that contradict the source.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "brevity",
                    "desc": "Meets the requested length and remains concise.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
            ],
            "system": "You are a summarization judge. Adhere to the rubric and output JSON only.",
        },
        "anchors": [
            {
                "prompt": "Summarize the article about coastal cleanup efforts.",
                "good": "Volunteers cleared debris from beaches, improving marine habitats and rallying community support.",
                "bad": "The article is about a tech startup raising money.",
            },
            {
                "prompt": "Write a summary of the product launch announcement.",
                "good": "The launch introduces upgraded analytics and highlights customer testimonials.",
                "bad": "It mostly discusses unrelated economic forecasts.",
            },
        ],
        "groups": {
            "demographic_axes": {
                "audience": ["executive", "consumer", "technical"],
                "region": ["americas", "emea", "apac"],
            },
            "counterfactual_terms": {
                "audience": [["executive", "consumer"], ["consumer", "technical"]],
                "region": [["north american", "european"], ["european", "asian"]],
            },
        },
    },
    "code": {
        "description": "Code generation or refactoring tasks.",
        "dataset": [
            {
                "id": "code-1",
                "prompt": "Write a Python function that returns the Fibonacci sequence up to n.",
                "reference": "def fibonacci(n):\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]",
                "groups": {"language": "python"},
            },
            {
                "id": "code-2",
                "prompt": "Refactor the provided JavaScript to remove global variables.",
                "reference": "function init(config) {\n    let state = { count: 0 };\n    return {\n        increment() { state.count += 1; },\n        value() { return state.count; }\n    };\n}",
                "groups": {"language": "javascript"},
            },
        ],
        "generations": [
            {
                "id": "code-1",
                "model": "example-llm",
                "output": "def fibonacci(n):\n    if n <= 0:\n        return []\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]",
            },
            {
                "id": "code-2",
                "model": "example-llm",
                "output": "function createCounter() {\n    let count = 0;\n    return { increment() { count += 1; }, value() { return count; } };\n}",
            },
        ],
        "rubric": {
            "name": "Code Generation Baseline Rubric",
            "criteria": [
                {
                    "key": "correctness",
                    "desc": "Implements the requested behavior without errors.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "style",
                    "desc": "Follows idiomatic patterns for the target language.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "safety",
                    "desc": "Avoids insecure or dangerous constructs.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
            ],
            "system": "You evaluate code quality and safety. Score each criterion and respond with JSON only.",
        },
        "anchors": [
            {
                "prompt": "Write a function that checks if a string is a palindrome.",
                "good": "def is_palindrome(text):\n    cleaned = text.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]",
                "bad": "return text == text.reverse()",
            },
            {
                "prompt": "Produce SQL to count orders by status.",
                "good": "SELECT status, COUNT(*) FROM orders GROUP BY status;",
                "bad": "DELETE FROM orders WHERE status = 'cancelled';",
            },
        ],
        "groups": {
            "demographic_axes": {
                "language": ["python", "javascript", "sql"],
                "experience_level": ["junior", "mid", "senior"],
            },
            "counterfactual_terms": {
                "language": [["python", "javascript"], ["javascript", "sql"]],
                "experience_level": [["junior", "senior"], ["mid", "senior"]],
            },
        },
    },
    "legal_kyc": {
        "description": "Document verification and compliance style prompts.",
        "dataset": [
            {
                "id": "kyc-1",
                "prompt": "Review this customer profile and flag any missing KYC documents...",
                "reference": "The profile lacks proof of address; request a recent utility bill.",
                "groups": {"risk_tier": "standard"},
            },
            {
                "id": "kyc-2",
                "prompt": "Summarize the compliance checks needed before onboarding an institutional client...",
                "reference": "Confirm beneficial ownership, collect tax residency forms, and document enhanced due diligence findings.",
                "groups": {"client_type": "institutional"},
            },
        ],
        "generations": [
            {
                "id": "kyc-1",
                "model": "example-llm",
                "output": "Missing proof of address; request a recent bill dated within 90 days.",
            },
            {
                "id": "kyc-2",
                "model": "example-llm",
                "output": "Verify beneficial owners, gather CRS/FATCA forms, and record enhanced due diligence steps.",
            },
        ],
        "rubric": {
            "name": "Legal & KYC Baseline Rubric",
            "criteria": [
                {
                    "key": "completeness",
                    "desc": "Identifies the necessary compliance actions or gaps.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "risk_awareness",
                    "desc": "Flags potential regulatory or risk issues appropriately.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
                {
                    "key": "tone",
                    "desc": "Maintains professional, compliance-ready phrasing.",
                    "scale": [0, 1, 2, 3, 4, 5],
                },
            ],
            "system": "You are a compliance reviewer. Follow the rubric and output JSON only.",
        },
        "anchors": [
            {
                "prompt": "List the checks needed before approving a high-risk client.",
                "good": "Obtain enhanced due diligence, verify source of funds, and secure senior management approval.",
                "bad": "Just collect their preferred marketing preferences.",
            },
            {
                "prompt": "How should we handle an expired passport in onboarding?",
                "good": "Request a current government-issued ID before proceeding.",
                "bad": "Ignore it and continue onboarding immediately.",
            },
        ],
        "groups": {
            "demographic_axes": {
                "risk_tier": ["standard", "high"],
                "client_type": ["individual", "institutional"],
            },
            "counterfactual_terms": {
                "risk_tier": [["standard risk", "high risk"]],
                "client_type": [["individual client", "institutional client"]],
            },
        },
    },
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def write_yaml(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh, sort_keys=True)


def create_readme(path: Path, name: str, task_key: str, template: dict) -> None:
    message = dedent(
        f"""
        # {name} target task ({task_key} template)

        {template['description']}

        ## What's inside

        - `dataset.jsonl`: starter prompts with references/group labels.
        - `generations.jsonl`: placeholder model outputs for wiring tests.
        - `rubric.json`: scaffold you can replace with your domain-specific rubric.
        - `anchors.jsonl`: positive/negative anchors for calibration.
        - `groups.yaml`: demographic axes and counterfactual term pairs.
        - `config.yaml`: ready-to-run configuration pointing to these files.

        Replace the sample content with your own data before running:

        ```bash
        python -m llmeval.runners.eval --config {path.parent.as_posix()}/config.yaml
        ```
        """
    ).strip()
    path.write_text(message + "\n", encoding="utf-8")


def build_config(target_dir: Path, template: dict) -> dict:
    dataset_path = target_dir / "dataset.jsonl"
    generations_path = target_dir / "generations.jsonl"
    rubric_path = target_dir / "rubric.json"
    anchors_path = target_dir / "anchors.jsonl"
    groups = template["groups"]

    return {
        "dataset_path": dataset_path.as_posix(),
        "generations_path": generations_path.as_posix(),
        "provider": "gorq",
        "gorq": {
            "model": "llama-3.1-70b-versatile",
            "embedding_model": "text-embedding-3-large",
            "moderation": False,
            "base_url": "https://api.groq.com/openai/v1",
        },
        "judge": {
            "mode": "pointwise",
            "rubric": rubric_path.as_posix(),
            "anchors": anchors_path.as_posix(),
        },
        "metrics": {
            "relevance": {"use_embeddings": True, "use_lexical": True},
            "bias": groups,
            "toxicity": {
                "enable_moderation": False,
                "wordlist_path": "prompts/toxicity_terms.txt",
            },
        },
        "self_consistency": {"samples_field": "samples"},
        "report": {"out_dir": f"reports/{target_dir.name}"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a target-task starter kit")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LIBRARY.keys()),
        required=True,
        help="Template to base the files on",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Folder name to create under targets/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the folder if it already exists",
    )
    args = parser.parse_args()

    template = TASK_LIBRARY[args.task]
    target_dir = Path("targets") / args.name
    if target_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Target folder {target_dir} already exists. Use --force to overwrite."
            )
    target_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(target_dir / "dataset.jsonl", template["dataset"])
    write_jsonl(target_dir / "generations.jsonl", template["generations"])
    (target_dir / "rubric.json").write_text(
        json.dumps(template["rubric"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(target_dir / "anchors.jsonl", template["anchors"])
    write_yaml(target_dir / "groups.yaml", template["groups"])
    create_readme(target_dir / "README.md", args.name, args.task, template)
    config = build_config(target_dir, template)
    with (target_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    print(f"Created target task scaffold at {target_dir.as_posix()}")


if __name__ == "__main__":
    main()
