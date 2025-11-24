import asyncio
from typing import Any, Dict, List, Optional
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.database.db import connect
from backend.services.prompt_runner.runner_service import RunnerService
from fastapi.middleware.cors import CORSMiddleware

# Load env (so RunnerService + Gemini + Langfuse get their keys)
# Try multiple possible .env locations
env_loaded = False
possible_paths = [
    ".env",  # Current directory (backend/)
    "../.env",  # Parent directory (project root)
    "backend/.env",  # If running from project root
]
for env_path in possible_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        env_loaded = True
        break

if not env_loaded:
    # Fallback: try without path (uses default search)
    load_dotenv()

app = FastAPI(
    title="Prompt Performance Analyzer API",
    version="0.1.0",
    description="Backend API for running and inspecting prompt performance experiments.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Pydantic Models ----
class PromptVariantRequest(BaseModel):
    label: str = Field(..., description="Short identifier for the prompt variant, e.g. 'P1'")
    template: str = Field(..., description="Prompt template string with variables like {text}")
    notes: Optional[str] = Field(None, description="Optional notes about this variant")

class ExperimentRunRequest(BaseModel):
    name: Optional[str] = Field(None, description="Human-readable name for the experiment")
    model: Optional[str] = Field("gemini-2.5-flash", description="LLM model name")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature")
    variables: Dict[str, Any] = Field(..., description="Input variables to fill into the templates")
    variants: List[PromptVariantRequest] = Field(..., description="List of prompt variants to test")

class ExperimentListItem(BaseModel):
    experiment_id: str
    name: Optional[str]
    model: str
    temperature: float
    created_at: str

class ExperimentResultItem(BaseModel):
    variant_label: str
    latency_ms: Optional[int]
    tokens: Optional[Dict[str, Optional[int]]]
    metrics_confidence: Optional[str]
    cost_usd: Optional[float]
    status: str
    trace_url: Optional[str]
    response_text: Optional[str]

class ExperimentDetailResponse(BaseModel):
    experiment_id: str
    name: Optional[str]
    model: str
    temperature: float
    created_at: str
    input_preview: Optional[str]
    results: List[ExperimentResultItem]

# ---- Initialize Runner Service ----
runner = RunnerService(
    default_model="gemini-2.5-flash",
    default_temperature=0.0,
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# ---- POST /api/experiments/run ----
@app.post("/api/experiments/run")
async def run_experiment(req: ExperimentRunRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "name": req.name,
        "model": req.model,
        "temperature": req.temperature,
        "variables": req.variables,
        "variants": [
            {
                "label": v.label,
                "template": v.template,
                "notes": v.notes,
            }
            for v in req.variants
        ],
    }

    # Offload sync method to a thread pool to prevent blocking FastAPI
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, runner.run_experiment, config)

    return summary

# ---- GET /api/experiments ----
@app.get("/api/experiments", response_model=List[ExperimentListItem])
def list_experiments() -> List[ExperimentListItem]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, name, model, temperature, created_at
            FROM experiments
            ORDER BY datetime(created_at) DESC
            """
        ).fetchall()

    return [
        ExperimentListItem(
            experiment_id=row["experiment_id"],
            name=row["name"],
            model=row["model"],
            temperature=row["temperature"],
            created_at=row["created_at"],
        )
        for row in rows
    ]

# ---- GET /api/experiments/{experiment_id} ----
@app.get("/api/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    with connect() as conn:
        exp_row = conn.execute(
            """
            SELECT experiment_id, name, model, temperature, created_at
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        if exp_row is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        rows = conn.execute(
            """
            SELECT
              pv.label AS variant_label,
              r.run_id,
              r.status,
              r.started_at,
              r.finished_at,
              r.error_message,
              rm.latency_ms,
              rm.input_tokens,
              rm.output_tokens,
              rm.total_tokens,
              rm.cost_usd,
              rm.trace_url,
              rm.metrics_confidence,
              ro.response_text
            FROM prompt_variants pv
            JOIN runs r ON r.variant_id = pv.variant_id
            LEFT JOIN run_metrics rm ON rm.run_id = r.run_id
            LEFT JOIN run_outputs ro ON ro.run_id = r.run_id
            WHERE pv.experiment_id = ?
            ORDER BY datetime(r.started_at) ASC
            """,
            (experiment_id,),
        ).fetchall()

    results: List[ExperimentResultItem] = []
    for row in rows:
        results.append(
            ExperimentResultItem(
                variant_label=row["variant_label"],
                latency_ms=row["latency_ms"],
                tokens={
                    "input": row["input_tokens"],
                    "output": row["output_tokens"],
                    "total": row["total_tokens"],
                }
                if row["input_tokens"] is not None or row["output_tokens"] is not None
                else None,
                metrics_confidence=row["metrics_confidence"],
                cost_usd=row["cost_usd"],
                status=row["status"],
                trace_url=row["trace_url"],
                response_text=row["response_text"],
            )
        )

    input_preview = None
    if results and results[0].response_text:
        input_preview = results[0].response_text[:120]

    return ExperimentDetailResponse(
        experiment_id=exp_row["experiment_id"],
        name=exp_row["name"],
        model=exp_row["model"],
        temperature=exp_row["temperature"],
        created_at=exp_row["created_at"],
        input_preview=input_preview,
        results=results,
    )

# ---- Graceful Shutdown ----
@app.on_event("shutdown")
def shutdown_event():
    print("🛑 Server is shutting down gracefully...")

# ---- Local Dev Runner ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
        log_level="info"
    )
