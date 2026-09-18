"""OpenAI-compatible LLM client with per-call cost tracking and model fallback.

The API key is read from the environment (optionally loaded from app-build/.env) and is never
printed, logged or saved. Every request/response goes to data_private/llm_log/{log_name}.jsonl.

Model fallback (owner instruction 2026-09-17,: models are tried in the order
LLM_MODEL, then LLM_MODEL_FALLBACKS. A model whose quota is used up is recorded in model_state.json
and skipped from then on, so resumed runs do not retry it. A model that is not activated or not found
is skipped for the current process only.
"""
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import openai

from lcbank.common import config
from lcbank.common.paths import LLM_LOG

LEDGER = LLM_LOG / "cost_ledger.jsonl"
MODEL_STATE = LLM_LOG / "model_state.json"

# Ark error codes. Observed 2026-09-17: ModelNotOpen, InvalidEndpointOrModel.NotFound. The quota codes are
# unverified; the raw error is logged on every switch.
# Setup errors (fixable by the project in the Ark Console) are skipped for this process only.
SETUP_CODES = {"ModelNotOpen", "InvalidEndpointOrModel.NotFound"}
# Quota used up: recorded in model_state.json so resumed runs skip the model.
QUOTA_CODES = {"SetLimitExceeded", "AccountOverdueError", "InsufficientQuota", "QuotaExhausted"}
UNAVAILABLE_CODES = SETUP_CODES | QUOTA_CODES
UNAVAILABLE_TEXT = re.compile(r"quota.*(used up|exhausted|exceeded)|reached the set.*limit|"
                              r"overdue|insufficient (balance|quota)|额度|余额", re.I)


class BudgetExceeded(RuntimeError):
    pass


class AllModelsUnavailable(RuntimeError):
    pass


def _prices():
    p = config.load("prices")
    budget = os.environ.get("LLM_BUDGET_CNY", p["budget_cny"])  # None: owner waived the stop (D-008)
    return (float(os.environ.get("LLM_PRICE_IN_CNY_PER_M", p["price_in_cny_per_m"])),
            float(os.environ.get("LLM_PRICE_OUT_CNY_PER_M", p["price_out_cny_per_m"])),
            float(budget) if budget not in (None, "") else None,
            float(p["pause_fraction"]))


def spent_cny():
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER, encoding="utf-8") as f:
        return sum(json.loads(line)["cny"] for line in f if line.strip())


def _error_code(err):
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("code"):
            return str(inner["code"])
    return None


def is_model_unavailable(err):
    """True if the error means this model is permanently unusable (quota used up / not activated)."""
    if isinstance(err, openai.RateLimitError) and _error_code(err) not in UNAVAILABLE_CODES \
            and not UNAVAILABLE_TEXT.search(str(err)):
        return False  # ordinary rate limiting is transient: back off, do not switch
    if not isinstance(err, openai.API:
        return False
    return _error_code(err) in UNAVAILABLE_CODES or bool(UNAVAILABLE_TEXT.search(str(err)))


class LLMClient:
    _lock = threading.Lock()

    def __init__(self, prefix="LLM"):
        config.load_dotenv()
        missing = [v for v in (f"{prefix}_API_KEY", f"{prefix}_BASE_URL", f"{prefix}_MODEL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"missing environment variables: {missing}")
        fallbacks = [m.strip() for m in os.environ.get(f"{prefix}_MODEL_FALLBACKS", "").split(",") if m.strip()]
        self.models = [os.environ[f"{prefix}_MODEL"]] + [m for m in fallbacks if m != os.environ[f"{prefix}_MODEL"]]
        self.client = openai.OpenAI(api_key=os.environ[f"{prefix}_API_KEY"],
                                    base_url=os.environ[f"{prefix}_BASE_URL"], timeout=180, max_retries=0)
        extra = os.environ.get(f"{prefix}_EXTRA_BODY")
        self.extra_body = json.loads(extra) if extra else None
        self.price_in, self.price_out, self.budget, self.pause_fraction = _prices()
        self.skipped = {}  # model -> setup error code, this process only
        self.pin = None    # if set, only this model may be used (a full run must not mix models, D-026)

    @staticmethod
    def _state():
        return json.loads(MODEL_STATE.read_text()) if MODEL_STATE.exists() else {"unavailable": {}}

    def _mark_unavailable(self, model, err):
        if _error_code(err) in SETUP_CODES:
            self.skipped[model] = _error_code(err)
            return
        with self._lock:
            state = self._state()
            state["unavailable"][model] = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                           ",
                                           "code": _error_code(err), "message": str(err)[:300]}
            LLM_LOG.mkdir(parents=True, exist_ok=True)
            MODEL_STATE.write_text(json.dumps(state, indent=1))

    @property
    def model(self):
        """The model the next call will use."""
        dead = self._state()["unavailable"]
        return next((m for m in self.models if m not in dead and m not in self.skipped), None)

    def check_budget(self):
        spent = spent_cny()
        if self.budget is not None and spent >= self.budget * self.pause_fraction:
            raise BudgetExceeded(f"spent {spent:.2f} CNY >= {self.pause_fraction:.0%} of budget {self.budget:.0f}")
        return spent

    def chat(self, messages, log_name, purpose, **kwargs):
        self.check_budget()
        if self.extra_body and "extra_body" not in kwargs:
            kwargs["extra_body"] = self.extra_body
        while True:
            model = self.model
            if self.pin and model != self.pin:
                raise AllModelsUnavailable(f"pinned model {self.pin} is unavailable (next would be {model})")
            if model is None:
                raise AllModelsUnavailable(f"no usable model; setup errors {self.skipped}, "
                                           f"quota used up {sorted(self._state()['unavailable'])}")
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
            except openai.API
                if is_model_unavailable(e):
                    self._mark_unavailable(model, e)
                    continue
                raise
            return resp, self._record(resp, model, messages, kwargs, log_name, purpose, time.time() - t0)

    def _record(self, resp, model, messages, kwargs, log_name, purpose, seconds):
        usage = resp.usage
        tin, tout = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
        # Estimated at the quoted §4 prices; other models may be priced differently (D-009).
        cny = tin / 1e6 * self.price_in + tout / 1e6 * self.price_out
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        LLM_LOG.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": now, "model": model, "purpose": purpose, "log": log_name,
                                    "tokens_in": tin, "tokens_in_cached": cached, "tokens_out": tout,
                                    "cny": cny}) + "\n")
            with open(LLM_LOG / f"{log_name}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": now, "model": model, "served_model": resp.model, "purpose": purpose,
                                    "request": messages,
                                    "kwargs": {k: v for k, v in kwargs.items() if k != "extra_body"},
                                    "response": resp.model_dump(), "seconds": round(seconds, 2),
                                    "cny": cny}, ensure_ascii=False) + "\n")
        return cny
