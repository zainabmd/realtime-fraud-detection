# Operations

How to deploy, configure and run the platform, and what to do when something goes wrong.
The code is the same on a laptop and in production. Only configuration changes.

## 1. Deploy
**Local:** `python run_local.py --fresh --hours 6 --tps 25` (SQLite broker, local Parquet lake).

**Single host / staging (Docker Compose, Kafka in KRaft mode):**
```bash
python -m src.train_model          # writes the model files that are baked into the image
docker compose up -d --build       # kafka, consumer, producer, graph, monitor, api, dashboard
docker compose logs -f consumer
```

**Production (Kubernetes):**
- **Kafka:** 3 brokers; `transactions.v1` with 12 partitions, replication 3, `min.insync.replicas=2`. Create
  topics with infrastructure-as-code, never with auto-create.
- **Consumer:** one pod per 2–3 partitions (never more pods than partitions). Events are keyed by `card_id`, so
  each card's state lives in one pod. Scale by adding partitions and pods. Keep checkpoints on a persistent
  volume so a moved pod restarts warm. On shutdown the consumer flushes, checkpoints, then commits.
- **Graph and monitor:** one instance each (`python -m src.graph --watch 120`, `python -m src.monitoring --watch 300`).
- **API:** 2+ replicas with autoscaling. **Dashboard:** internal only, behind SSO (it shows card-level data).
- **Lake:** move to object storage (S3/GCS/ADLS). pandas writes and DuckDB `httpfs` reads work unchanged.
  Run compaction as a scheduled job instead of inside the consumer.
- Tag each image with the git SHA and model version.

## 2. Configuration and secrets
All settings are environment variables (`src/config.py`, `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `BROKER_BACKEND` | `sqlite` | `kafka` in production |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | broker list |
| `NUM_PARTITIONS` | 4 | must match the topic |
| `FRAUD_DATA_DIR` | `./data` | mounted volume or bucket |
| `REVIEW_THRESHOLD` / `DECLINE_THRESHOLD` | 50 / 80 | owned by the risk team |
| `CHECKPOINT_EVERY_S` | 30 | limits replay after a crash |
| `COPILOT_MODEL` / `COPILOT_EFFORT` | `claude-opus-5-5` / `medium` | copilot only |

**Secrets** (Kafka credentials, bucket keys, `ANTHROPIC_API_KEY`) come only from the environment, filled by
the platform's secret manager. Never put them in images, files or Git. Mount the Anthropic key only into the API
and dashboard; the consumer, graph job and monitor never need it. Copilot cost grows with analyst use, not
with traffic.

**Security basics:** no card numbers on the stream (`card_id` is a token); TLS and ACLs on Kafka; encrypted
buckets with least-privilege roles; every decline keeps its `rules_triggered` and `top_features` for audit.

## 3. Monitoring and alerts
| Signal | Target | Where |
|---|---|---|
| Latency (broker → scored) | p95 ≤ 2 s | Pipeline Health, `/metrics` |
| Freshness | lake written ≤ 30 s ago | `/health` |
| Consumer lag | < 5,000 messages | Pipeline Health |
| Dead-letter rate | < 0.5% of events | Pipeline Health |
| Flag rate | 3–6% at the review threshold | Command Center |
| Drift | max PSI < 0.25 | Model Health, `/model-health` |

Suggested alerts: lag > 5,000 for 5 min (page); p95 latency > 2 s for 10 min (page); flag rate > 8% for
15 min (ticket); max PSI > 0.25 for 30 min (ticket).

## 4. Incident playbooks
**Drift alert (PSI > 0.25).**
1. Open Model Health and see which features moved, and in which direction.
2. Business change (new prices, new product) or data defect (a broken field shows as a spike in one bin)?
3. Data defect: fix it upstream and tighten validation.
4. Real shift on a score: check precision on matured labels. If it fell, retrain (`python -m src.train_model`)
   and release as below.

**Model promotion.** When the monitor says `promote`:
1. Confirm the verdict holds over two reports in a row (24 h of matured labels).
2. Check `recall_by_type`: the new model must not lose a fraud type the old one catches.
3. Release the three files together: `fraud_model.joblib`, `challenger.joblib`, `reference_profile.json`.
   Keep the old champion as the new challenger for a week and watch the flag rate.

**Model rollback.** Redeploy the previous image tag. Every scored row has `model_version`, so no data
migration is needed and before/after checks are a simple filter.

**Copilot outage.** The dashboard and API keep working: the copilot falls back to the offline analyst and
says so. Check that `ANTHROPIC_API_KEY` is set, the Anthropic API status and rate limits. `src.copilot` logs the
error class.

**Consumer crash.** Recovery is automatic: offsets are committed only with a state checkpoint, so a restart
replays at most 30 s and the lake views remove duplicates by `txn_id`. If the checkpoint is corrupt, delete
`data/state/feature_state.pkl`; the consumer starts cold and recovers within about an hour.

## 5. Useful commands
```bash
curl localhost:8000/health                  # liveness, model version, lake freshness
curl localhost:8000/model-health            # drift + promotion verdict
python -m src.copilot --ring R-2DA4A0       # copilot from the CLI (--txn, --ask)
```
