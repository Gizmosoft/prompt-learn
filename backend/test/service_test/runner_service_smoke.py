'''
    Handles smoke tests for runner service
'''

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="backend/.env")

from backend.services.prompt_runner.runner_service import RunnerService

runner = RunnerService(
    default_model="gemini-2.5-flash",
    default_temperature=0.0,
    # Explicit keys; reads from env if None
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

config = {
    "name": "Phase2 Smoke",
    "model": "gemini-2.5-flash",
    "temperature": 0,
    "variables": {
        "text": "LangChain helps developers orchestrate LLM calls; Gemini provides fast reasoning."
    },
    "variants": [
        {"label": "P1", "template": "Summarize in 2 lines:\n{text}"},
        {"label": "P2", "template": "Give a concise 2-sentence summary:\n{text}"}
    ]
}

summary = runner.run_experiment(config)
print("=== Phase 2 Summary ===")
print("Experiment ID:",summary["experiment_id"])
print("Input: ", summary["input_preview"])
for r in summary["results"]:
    print(
        f'{r["variant_label"]} {r["variant_template"]} {r["status"]} {r.get("latency_ms")}ms '
        f'tokens={r["tokens"]} cost=${r["cost_usd"]} source={r["metrics_confidence"]}'
    )
