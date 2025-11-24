'''
    This service is called Prompt Runner which uses LangChain and Langfuse to run one prompt on the provided
    user_content and returns the response from the LLM
'''
import os
import json
import time
import uuid
import hashlib

from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from backend.database.db import connect, init_db

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
# Try multiple possible .env locations
env_loaded = False
possible_paths = [
    ".env",  # Current directory
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

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

# Approximate prices per 1K tokens in USD
PRICES = {
    "gemini-2.5-flash": {
        "input_per_1k": 0.15,   # $ per 1K input tokens
        "output_per_1k": 0.60   # $ per 1K output tokens
    },
    "gemini-2.5-pro": {
        "input_per_1k": 0.50,
        "output_per_1k": 1.50
    }
}

#### Helper functions ####
def _now_iso() -> str:
    return datetime.utcnow().strftime(ISO)


def _uuid() -> str:
    return str(uuid.uuid4())


def _input_hash(variables: Dict[str, Any]) -> str:
    # Stable hash of the input variables for idempotency
    dump = json.dumps(variables, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


class RunnerService():
    """
    Orchestrates multi-variant prompt execution with Gemini + LangFuse,
    persists results to SQLite, returns an aggregated summary.
    """

    def __init__(
        self,
        default_model: str = "gemini-2.5-flash",
        default_temperature: float = 0.0,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
    ):
        init_db()

        self.model = default_model
        self.temperature = default_temperature

        # Initialize Langfuse only if credentials are available
        # Langfuse reads from environment variables, so we set them if provided as parameters
        public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key:
            try:
                # Set environment variables if provided as parameters (temporarily override)
                if langfuse_public_key:
                    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
                if langfuse_secret_key:
                    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret_key
                if langfuse_host:
                    os.environ["LANGFUSE_HOST"] = langfuse_host
                
                # Langfuse reads from environment variables automatically
                self.langfuse = get_client()
                self.langfuse_handler = CallbackHandler()
            except Exception as e:
                print(f"⚠️ Warning: Langfuse initialization failed: {e}. Continuing without Langfuse tracing.")
                self.langfuse = None
                self.langfuse_handler = None
        else:
            # Skip Langfuse initialization entirely if credentials are missing
            self.langfuse = None
            self.langfuse_handler = None

        # Initialize Gemini LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("⚠️ Warning: GOOGLE_API_KEY not found in environment. LLM calls will fail.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            google_api_key=google_api_key
        )

    # ---------- Persistence helpers ----------

    def _ensure_experiment(self, exp: Dict[str, Any]) -> None:
        with connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO experiments (experiment_id, name, model, temperature, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (exp["experiment_id"], exp.get("name"), exp["model"],
                 exp.get("temperature", 0.0), _now_iso()),
            )

    def _ensure_variant(self, experiment_id: str, label: str, template: str, notes: Optional[str]) -> str:
        variant_id = _uuid()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO prompt_variants (variant_id, experiment_id, label, template, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (variant_id, experiment_id, label, template, notes),
            )
        return variant_id

    def _create_run_row(self, experiment_id: str, variant_id: str, input_hash: str) -> str:
        run_id = _uuid()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, experiment_id, variant_id, input_hash, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, variant_id,
                 input_hash, _now_iso(), "running"),
            )
        return run_id

    def _complete_run_row(self, run_id: str, status: str, error_message: Optional[str] = None) -> None:
        with connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = ?, error_message = ?
                WHERE run_id = ?
                """,
                (_now_iso(), status, error_message, run_id),
            )

    def _persist_metrics(
        self,
        run_id: str,
        latency_ms: int,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        cost_usd: Optional[float],
        trace_url: Optional[str],
        metrics_confidence: str = "unknown"
    ) -> None:
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO run_metrics (run_id, latency_ms, input_tokens, output_tokens, total_tokens, cost_usd, trace_url, metrics_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, latency_ms, input_tokens, output_tokens,
                 total_tokens, cost_usd, trace_url, metrics_confidence),
            )

    def _persist_output(self, run_id: str, response_text: str, raw_json: Optional[Dict[str, Any]]) -> None:
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO run_outputs (run_id, response_text, raw_json)
                VALUES (?, ?, ?)
                """,
                (run_id, response_text, json.dumps(raw_json,
                 ensure_ascii=False) if raw_json is not None else None),
            )

    # ---------- Cost helpers ----------
    @staticmethod
    def _estimate_cost_usd(
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        model: str
    ) -> Optional[float]:
        """
        Compute cost from measured token usage and a simple per-1K pricing table.
        Returns None if token usage or pricing is missing.
        """
        if input_tokens is None or output_tokens is None:
            return None

        pricing = PRICES.get(model)
        if not pricing:
            # Unknown model -> we don't know its price
            return None

        input_cost = (input_tokens / 1000.0) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000.0) * pricing["output_per_1k"]
        return round(input_cost + output_cost, 6)

    def _extract_gemini_usage(self, resp) -> tuple[int | None, int | None, int | None, str]:
        '''
        Returns (input_tokens, output_tokens, total_tokens, confidence)
        confidence: "measured" when all are ints, otherwise "unknown"
        '''
        usage = None

        # Preferred path (most LangChain builds)
        rm = getattr(resp, "response_metadata", None)
        if isinstance(rm, dict):
            usage = rm.get("usage_metadata") or rm.get("token_usage")

        # Fallbacks
        if usage is None:
            ak = getattr(resp, "additional_kwargs", None)
            if isinstance(ak, dict):
                usage = ak.get("usage_metadata") or ak.get("token_usage")

        if usage is None:
            usage = getattr(resp, "usage_metadata", None)

        # Normalize keys
        inp = usage.get("prompt_token_count") if usage else None
        out = usage.get("candidates_token_count") if usage else None
        tot = usage.get("total_token_count") if usage else None

        # Some variants use different names
        if inp is None and usage:
            inp = usage.get("input_tokens") or usage.get("prompt_tokens")
        if out is None and usage:
            out = usage.get("output_tokens") or usage.get("completion_tokens")
        if tot is None and usage:
            tot = usage.get("total_tokens") or ( (inp or 0) + (out or 0) if (inp is not None and out is not None) else None )

        all_ints = all(isinstance(x, int) for x in (inp, out, tot) if x is not None)
        return (inp if all_ints else None,
                out if all_ints else None,
                tot if all_ints else None,
                "measured" if all_ints else "unknown")


    # ---------- Core execution ----------
    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        config = {
          "experiment_id": "...",            # optional (auto if missing)
          "name": "My first test",           # optional
          "model": "gemini-2.5-flash",       # optional override
          "temperature": 0,                  # optional override
          "variables": {"text": "..."},
          "variants": [
            {"label": "P1", "template": "Summarize:\n{text}", "notes": ""},
            {"label": "P2", "template": "Two-sentence summary:\n{text}"}
          ]
        }
        """
        experiment_id = config.get("experiment_id") or _uuid()
        model = config.get("model", self.model)
        temperature = float(config.get("temperature", self.temperature))
        variables = config["variables"]
        variants = config["variants"]
        name = config.get("name")

        # Basic validation
        if not variants or not isinstance(variants, list):
            raise ValueError("config.variants must be a non-empty list")
        if not variables or not isinstance(variables, dict):
            raise ValueError("config.variables must be a dict")

        # Update runner model settings if overridden
        if model != self.model or temperature != self.temperature:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

        # Create experiment row if missing
        self._ensure_experiment({
            "experiment_id": experiment_id,
            "name": name,
            "model": model,
            "temperature": temperature
        })

        # Stable hash for this input
        in_hash = _input_hash(variables)

        aggregated_results: List[Dict[str, Any]] = []

        for v in variants:
            label = v["label"]
            template = v["template"]
            notes = v.get("notes")

            # Insert variant row
            variant_id = self._ensure_variant(
                experiment_id, label, template, notes)

            # Create run row
            run_id = self._create_run_row(experiment_id, variant_id, in_hash)

            # Create a manual LangFuse trace for easy linking (independent of callback internals)
            # trace = self.langfuse.trace(
            #     name=f"prompt_analyzer/{experiment_id}/{label}",
            #     metadata={
            #         "experiment_id": experiment_id,
            #         "variant_label": label,
            #         "model": model,
            #         "temperature": temperature,
            #         "input_hash": in_hash,
            #     }
            # )
            # trace_id = trace.id
            # host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            # trace_url = f"{host}/trace/{trace_id}"

            # Compose final prompt text
            try:
                final_text = template.format(**variables)
            except KeyError as ke:
                self._complete_run_row(
                    run_id, "error", f"Missing variable in template: {ke}")
                # trace.log_event(name="format_error", input=v,
                #                 output={"error": str(ke)})
                # trace.end()
                aggregated_results.append({
                    "variant_label": label,
                    "variant_template": template,
                    "status": "error",
                    "error_message": f"Missing variable {ke}",
                })
                continue

            # Execute with timing + callback
            start = time.perf_counter()
            status = "success"
            error_msg = None
            response_text = ""
            raw_json = None
            input_tokens = None
            output_tokens = None
            total_tokens = None
            trace_url = None

            try:
                msg = [HumanMessage(content=final_text)]
                # Attach LangFuse callback for granular spans (if available)
                invoke_config = {}
                if self.langfuse_handler is not None:
                    invoke_config["callbacks"] = [self.langfuse_handler]
                resp = self.llm.invoke(msg, config=invoke_config)
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                # Get Metrics
                input_tokens, output_tokens, total_tokens, metrics_confidence = self._extract_gemini_usage(resp)

                response_text = getattr(resp, "content", "") or ""
                raw_json = {
                    "ai_message": response_text
                }

                # Get trace URL from Langfuse handler if available
                if self.langfuse_handler is not None:
                    get_url = getattr(self.langfuse_handler, "get_trace_url", None)
                    if callable(get_url):
                        trace_url = get_url()
                    # fallback: build from a trace id if the handler exposes it
                    if not trace_url:
                        trace_id = getattr(self.langfuse_handler, "trace_id", None) or getattr(
                            self.langfuse_handler, "root_observation_id", None)
                        if trace_id:
                            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                            trace_url = f"{host}/trace/{trace_id}"

                # If usage becomes available in your env, extract here:
                # usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
                # if usage:
                #     input_tokens = usage.get("prompt_token_count")
                #     output_tokens = usage.get("candidates_token_count")
                #     total_tokens = usage.get("total_token_count")

                # Log input/output to our manual trace for consistent linking
                # trace.log_input({"prompt": final_text})
                # trace.log_output({"response": response_text})
                # trace.end()

                # Compute (optional) cost
                cost_usd = self._estimate_cost_usd(
                    input_tokens, output_tokens, model)
                # metrics_confidence = "unknown" if input_tokens is None else "measured"

                # Persist metrics and outputs
                self._persist_metrics(
                    run_id=run_id,
                    latency_ms=elapsed_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    trace_url=trace_url,
                    metrics_confidence=metrics_confidence
                )
                self._persist_output(run_id, response_text, raw_json)
                self._complete_run_row(run_id, "success", None)

                aggregated_results.append({
                    "variant_label": label,
                    "variant_template": template,
                    "latency_ms": elapsed_ms,
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens
                    },
                    "metrics_confidence": metrics_confidence,
                    "cost_usd": cost_usd,
                    "status": status,
                    "trace_url": trace_url,
                    "response_text": response_text
                })

            except Exception as e:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                status = "error"
                error_msg = str(e)

                # End trace with error context
                # trace.log_event(name="run_error", input={
                #                 "prompt": final_text}, output={"error": error_msg})
                # trace.end()

                self._persist_metrics(
                    run_id=run_id,
                    latency_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    cost_usd=None,
                    trace_url=trace_url,
                    metrics_confidence="unknown"
                )
                self._complete_run_row(run_id, status, error_msg)

                aggregated_results.append({
                    "variant_label": label,
                    "variant_template": template,
                    "status": status,
                    "error_message": error_msg,
                    "trace_url": trace_url
                })

        return {
            "experiment_id": experiment_id,
            "model": model,
            "temperature": temperature,
            "input_preview": str(variables)[:120],
            "results": aggregated_results
        }
