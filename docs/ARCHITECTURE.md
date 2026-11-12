# Architecture

## Event flow
```mermaid
flowchart LR
    SIM[simulator.py<br/>traffic, attacks, rings,<br/>late + malformed events] --> T1[(transactions.v1<br/>key = card_id)]
    SIM --> T2[(fraud-labels.v1<br/>delayed 5–90 min)]
    T1 --> V{schema v1}
    V -- invalid --> DLQ[(lake: dlq)]
    V -- valid --> F[FeatureEngine<br/>per-card state<br/>22 + 2 graph features]
    F --> CH[Champion<br/>rules + Isolation Forest<br/>decides]
    F --> SH[Challenger<br/>gradient boosting<br/>shadow only]
    CH -- review / decline --> T3[(fraud-alerts.v1)]
    CH & SH --> SC[(lake: scored)]
    T2 --> LB[(lake: labels)]
    F <-. checkpoint .-> CK[(feature_state.pkl)]
    SC --> G[graph job · 2 min] --> GR[(lake/graph)]
    SC & LB --> M[monitoring · 5 min] --> MR[(report.json)]
    SC & GR & MR --> Q[DuckDB] --> API[FastAPI] & UI[Streamlit] & CP[Claude copilot]
```
The graph job, monitoring and copilot only read the lake, so they can never block or change a live decision.

## Delivery semantics
- The consumer polls up to 2,000 events, validates, computes features, scores and publishes alerts.
- Every ≤ 5 s it flushes buffered rows to the lake as atomic Parquet files.
- Every 30 s it flushes, checkpoints feature state atomically, **then** commits broker offsets.
- After a crash it restores the checkpoint and replays from the committed offsets. State matches the offsets;
  replayed lake rows are removed by `txn_id` in the views.

## Scoring
```
if_points   = 60 × clip((percentile − 0.85) / 0.15, 0, 1) ^ 1.5    (Isolation Forest percentile)
rule_points = sum of fired rule points, capped at 70
risk        = min(100, if_points + rule_points)
decision    = decline if risk ≥ 80 · review if risk ≥ 50 · approve otherwise
```

| Rule | Points | Fires when |
|---|---|---|
| `IMPOSSIBLE_TRAVEL` | 45 | implied speed > 900 km/h over > 500 km |
| `SMALL_TXN_BURST` | 30 | ≥ 3 charges under $5 in 10 min |
| `ATO_PATTERN` | 30 | new device + foreign IP, card not present |
| `VELOCITY_1M` | 25 | ≥ 4 transactions in 60 s |
| `AMOUNT_SPIKE` | 25 | amount z-score > 3.5 and > $200 |
| `MERCHANT_SPRAY` | 25 | ≥ 5 distinct merchants in 10 min |
| `HIGH_RISK_MCC` | 15 | crypto, gift cards or gambling at > 3× usual spend |
| `NIGHT_CASH` | 10 | ATM at local night at > 2× usual spend |

## Graph features and rings
- **Streaming features.** `FeatureEngine` keeps a `device → {card: last_seen}` index. For each card-not-present
  payment it emits `device_cards_24h` and `card_devices_24h`. Store terminals are excluded. Only the challenger
  uses these two features.
- **Ring detection** (`src/graph.py`, every 2 min). Take the last 24 h of card-not-present payments. Link cards
  to devices, keeping only devices used by 2+ cards (a victim's own phone is not ring infrastructure). Keep
  connected components with 3+ cards. Score = 35% size + 35% collector concentration + 20% cards per device +
  10% challenger view. ≥ 70 is high, ≥ 45 medium. Snapshots are rewritten atomically.

## Champion / challenger
The **champion** (Isolation Forest + 8 rules, no labels needed) decides approve / review / decline. The
**challenger** (`HistGradientBoostingClassifier` on 22 + 2 graph features, trained on labels) writes only a
shadow `challenger_score`. Only the challenger can see rings.

**Promotion rule** (`src/monitoring.py::compare_models`): use only events older than 90 min, so labels have
arrived. Set the challenger's threshold to flag the same share of traffic as the champion. Then require at
least `PROMOTE_MIN_FRAUD` labelled frauds, a PR-AUC lead of at least `PROMOTE_MIN_GAIN`, and recall no worse.
Otherwise the verdict is `keep_champion`, `retire_challenger` or `insufficient_data`. A human does the release
([OPERATIONS.md](OPERATIONS.md)).

## Drift monitoring
Training saves decile edges for every input and both scores in `models/reference_profile.json`. Every 5 min
the last 3 h of events are binned on those edges and PSI is computed: below 0.10 is stable, 0.10–0.25 is
watch, above 0.25 is drift. Three fixes keep it honest:
- Values and edges are rounded to the lake's 4-decimal grid, so a point mass cannot straddle a bin edge.
- Training starts after a 2 h warm-up, so the reference has no cold-start skew.
- Time-of-day features are reported as seasonal and never alert.

## Analyst copilot (`src/copilot.py`)
- Claude (`COPILOT_MODEL`, default `claude-opus-5-5`), effort `medium`, server-side refusal fallbacks.
- A tool loop of at most 8 rounds over six strict, typed, read-only tools: `get_transaction`,
  `get_card_history`, `get_ring`, `list_rings`, `list_alerts`, `get_model_health`. No free-form SQL. Output is
  capped at 6 kB per call.
- Merchant names and device ids can be set by attackers, so the system prompt says tool output is data, never
  instructions.
- Without credentials, or on an API error, `OfflineAnalyst` writes a template brief from the same tools.
- The copilot recommends; analysts decide. It never changes a decision or a model.

## Key design decisions
| Decision | Why | Trade-off |
|---|---|---|
| SQLite commit log by default, Kafka adapter | runs anywhere with Kafka semantics | single host; use Kafka in production |
| Partition by `card_id` | per-card state needs no shared store | hot cards cause skew |
| Score-then-update, event-time state | no leakage; late events judged on their own past | state must be checkpointed |
| One FeatureEngine for training and serving | no train/serve skew in features | training is slower |
| Commit offsets only with a checkpoint | state is effectively processed once | up to 30 s replay, de-duplicated |
| Parquet lake + DuckDB | no database server, cheap path to object storage | small files, so compaction |
| Per-partition `MAX()` head-offset lookup | 2,459 ms → 0.9 ms vs `GROUP BY` | one query per partition |
| Compare models at equal flag rate | "better" means more fraud for the same analyst work | ignores calibration |
| Graph features streaming, rings in batch | flat per-event latency | rings appear up to 2 min late |
| Typed read-only copilot tools | small blast radius, simpler injection story | new questions may need new tools |

**Throughput** (laptop, SQLite broker): median 3.75k events/s per consumer process over 234 batches, with
challenger and graph features; the graph job analyses 320k payments in about 4 s.
