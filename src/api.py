"""
FastAPI service: synchronous scoring + read APIs over the lake.

    uvicorn src.api:app --port 8000          # docs at http://localhost:8000/docs

Endpoints
---------
GET  /health                 liveness + readiness (model loaded, lake freshness)
POST /score                  score one transaction synchronously (same features + model as the stream)
GET  /transactions/recent    latest scored transactions (filterable)
GET  /alerts                 latest review/decline decisions
GET  /stats                  rolling KPIs over the last N minutes
GET  /metrics                Prometheus exposition format (scrape target)
GET  /rings                  fraud rings from the latest card-device graph snapshot
GET  /rings/{ring_id}        one ring with members and shared devices
GET  /model-health           drift (PSI) + champion-vs-challenger report
POST /copilot                analyst copilot: case brief for a txn / ring, or a free-form question

The /score endpoint keeps its own in-memory FeatureEngine warmed from the
consumer's last checkpoint. It is *eventually consistent* with the stream
(state is as of the last checkpoint, typically < 30 s old).
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse

from src.config import get_logger
from src.detection import Scorer
from src.features import FeatureEngine
from src.schemas import Decision, ScoreResponse, Transaction
from src.storage import lake_summary, query

log = get_logger("api")
_state: dict = {}
_lock = threading.Lock()  # FeatureEngine state is not thread-safe; serialise /score


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["scorer"] = Scorer()
        eng = FeatureEngine()
        eng.restore()
        _state["engine"] = eng
        log.info("API ready · model=%s", _state["scorer"].bundle.version)
    except FileNotFoundError as exc:
        log.error("Startup degraded: %s", exc)
        _state["startup_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(title="Fraud Risk API", version="2.0.0", lifespan=lifespan,
              description="Real-time card fraud scoring and monitoring endpoints.")


@app.get("/health")
def health() -> dict:
    ready = "scorer" in _state
    fresh = query("SELECT max(processed_at) AS last FROM scored")
    last = fresh["last"].iloc[0] if not fresh.empty else None
    age = (time.time() - last.timestamp()) if last is not None and str(last) != "NaT" else None
    return {"status": "ok" if ready else "degraded", "model_version": _state["scorer"].bundle.version if ready else None,
            "error": _state.get("startup_error"), "lake_last_write_age_s": round(age, 1) if age is not None else None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    if "scorer" not in _state:
        raise HTTPException(503, detail=_state.get("startup_error", "model not loaded"))
    t0 = time.perf_counter()
    ev = txn.model_dump(mode="json")
    with _lock:
        feats = _state["engine"].compute_batch([ev], [txn.event_time.timestamp()])
    res = _state["scorer"].score(feats).iloc[0]
    return ScoreResponse(
        txn_id=txn.txn_id, risk_score=float(res["risk_score"]), decision=Decision(res["decision"]),
        if_score=float(res["if_anomaly"]), rule_points=float(res["rule_points"]),
        rules_triggered=[r for r in res["rules_triggered"].split("|") if r],
        top_features=[{"feature": k, "robust_z": v} for k, v in json.loads(res["top_features"]).items()],
        model_version=res["model_version"], latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.get("/transactions/recent")
def recent(limit: int = Query(50, ge=1, le=1000), min_risk: float = Query(0, ge=0, le=100)) -> list[dict]:
    df = query("SELECT txn_id, event_time, card_id, merchant_category, amount_usd, currency, country, txn_type, "
               "risk_score, decision, rules_triggered FROM scored WHERE risk_score >= ? "
               "ORDER BY event_time DESC LIMIT ?", [min_risk, limit])
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/alerts")
def alerts(limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    df = query("SELECT txn_id, event_time, card_id, amount_usd, merchant_category, country, risk_score, decision, "
               "rules_triggered FROM scored WHERE decision <> 'approve' ORDER BY event_time DESC LIMIT ?", [limit])
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/stats")
def stats(minutes: int = Query(60, ge=1, le=7 * 24 * 60)) -> dict:
    df = query(f"""
        WITH w AS (SELECT * FROM scored WHERE event_time >= (SELECT max(event_time) FROM scored) - INTERVAL {int(minutes)} MINUTE)
        SELECT count(*) AS txns, sum(amount_usd) AS volume_usd, avg(risk_score) AS avg_risk,
               avg((decision <> 'approve')::INT) AS flag_rate, sum(CASE WHEN decision='decline' THEN amount_usd END) AS declined_usd
        FROM w""")
    return {k: (None if v != v else v) for k, v in df.iloc[0].to_dict().items()} if not df.empty else {}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus() -> str:
    """Prometheus text exposition – point a scrape job at /metrics."""
    lines = []
    tot = query("SELECT decision, count(*) n FROM scored GROUP BY 1")
    lines += ["# HELP fraud_transactions_total Scored transactions by decision", "# TYPE fraud_transactions_total counter"]
    lines += [f'fraud_transactions_total{{decision="{r.decision}"}} {int(r.n)}' for r in tot.itertuples()]
    m = query("SELECT * FROM metrics ORDER BY event_time DESC LIMIT 1")
    if not m.empty:
        r = m.iloc[0]
        lines += ["# TYPE fraud_pipeline_latency_p95_ms gauge", f"fraud_pipeline_latency_p95_ms {r['latency_p95_ms']:.1f}",
                  "# TYPE fraud_consumer_lag gauge", f"fraud_consumer_lag {int(r['lag_total'])}"]
    for name, s in lake_summary().items():
        lines.append(f'fraud_lake_files{{table="{name}"}} {s["files"]}')
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- 2026: rings, model health, copilot
@app.get("/rings")
def rings(limit: int = Query(25, ge=1, le=200), min_score: float = Query(0, ge=0, le=100)) -> list[dict]:
    from src.graph import load_rings
    r, _, _ = load_rings()
    if r.empty:
        return []
    r = r[r["ring_score"] >= min_score].head(limit)
    return json.loads(r.to_json(orient="records", date_format="iso"))


@app.get("/rings/{ring_id}")
def ring(ring_id: str) -> dict:
    from src.copilot import LakeTools
    out = LakeTools().get_ring(ring_id=ring_id)
    if "error" in out:
        raise HTTPException(404, detail=out["error"])
    return out


@app.get("/model-health")
def model_health() -> dict:
    from src.monitoring import read_report
    r = read_report()
    if not r:
        raise HTTPException(404, detail="no monitoring report yet - run `python -m src.monitoring`")
    return r


class CopilotRequest(BaseModel):
    kind: str = Field(pattern="^(txn|ring|ask)$", description="txn = case brief for a transaction, ring = ring brief, ask = question")
    value: str = Field(min_length=1, max_length=2000)
    offline: bool = False


@app.post("/copilot")
def copilot(req: CopilotRequest) -> dict:
    from src.copilot import run
    return run(req.kind, req.value, prefer_claude=not req.offline).to_dict()
