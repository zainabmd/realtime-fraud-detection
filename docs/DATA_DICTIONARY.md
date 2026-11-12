# Data Dictionary

## Topic `transactions.v1` (`src/schemas.py::Transaction`)
Key: `card_id`. Value: JSON.

| Field | Type | Rule | Meaning |
|---|---|---|---|
| `schema_version` | int | = 1 | Contract version; bumped on breaking changes |
| `txn_id` | str | 8–64 chars | Unique payment id (idempotency key) |
| `event_time` | ISO-8601 UTC | valid datetime | When the payment happened. ~1% arrive 30–180 s late |
| `card_id` | str | required | Card token (never a card number) |
| `merchant_id`, `merchant_name` | str | required | |
| `merchant_category` | str | one of 17 | Grocery … Crypto Exchange, Gambling, ATM Cash, P2P Wallet |
| `txn_type` | enum | POS · ECOM · ATM · TRANSFER · P2P | |
| `channel` | enum | card_present · card_not_present | |
| `amount` | float | > 0 | In `currency` |
| `currency` | str(3) | ISO 4217 | 17 currencies |
| `amount_usd` | float | 0 < x < 1,000,000 | Converted with the FX snapshot |
| `country`, `city`, `lat`, `lon` | | ISO-3, lat in [-90, 90] | Terminal location (card present) or IP location (card not present) |
| `device_id` | str | | Customer device (card not present); `term_<merchant>` for card present |
| `ip_country` | str(3) | | IP country (card not present) |

## Topic `fraud-labels.v1` (`FraudLabel`)
| Field | Meaning |
|---|---|
| `txn_id`, `card_id` | The payment confirmed as fraud |
| `is_fraud` | Always true. Unlabelled payments are treated as legitimate |
| `fraud_type` | `card_testing` · `account_takeover` · `impossible_travel` · `velocity_burst` · `high_amount_anomaly` · `fraud_ring` |
| `label_time` | When the confirmation arrived (event_time + 5–90 min) |

## Topic `fraud-alerts.v1`
One JSON message per review or decline: `txn_id, card_id, event_time, amount_usd, merchant_category, country,
risk_score, decision, rules_triggered`.

## Lake table `scored` (`data/lake/scored/dt=YYYY-MM-DD/hr=HH/*.parquet`)
All event fields above (except `schema_version`), plus:

| Group | Columns | Notes |
|---|---|---|
| Card context | `home_country, home_lat, home_lon` | From the card reference table |
| Model features (22) | `amount_usd_log, amount_zscore, amount_to_avg_ratio, txn_count_1m, txn_count_10m, txn_count_1h, amount_sum_1h_log, distinct_merchants_10m, small_txn_count_10m, secs_since_last_log, km_from_last_log, speed_kmh_log, km_from_home_log, geo_mismatch, ip_mismatch, new_device, card_not_present, hour_sin, hour_cos, is_night, mcc_risk, is_atm` | From the card's past only |
| Graph features (2) | `device_cards_24h, card_devices_24h` | Other cards on this device / devices on this card in the last 24 h (card not present). Challenger only |
| Raw values | `speed_kmh, km_from_last, km_from_home` | Readable values used by rules, not by the model |
| Champion scores | `if_anomaly, if_percentile, if_points, rule_points, rules_triggered, risk_score, decision, top_features, model_version` | `rules_triggered` is pipe-separated; `top_features` is JSON `{feature: robust_z}` |
| Challenger | `challenger_score` (0–100), `challenger_version` | Shadow score; never drives the decision |
| Lineage | `partition, offset, broker_ts, processed_at, pipeline_latency_ms` | Trace a row back to its broker record |

## Lake table `labels`
`txn_id, card_id, is_fraud, fraud_type, label_time`. Partitioned by `label_time`.

## Lake table `metrics` (one row per micro-batch)
| Column | Meaning |
|---|---|
| `event_time`, `max_event_time` | Batch completion time; latest event time in the batch |
| `batch_events, valid_events, dlq_events, alerts` | Counts |
| `processing_ms, events_per_s` | Batch compute time and rate |
| `latency_p50_ms, latency_p95_ms, latency_p99_ms` | Broker append → scored |
| `lag_total, lag_by_partition` | Unprocessed messages (JSON per partition) |
| `cards_in_state` | Cards in the online feature store |

## Lake table `dlq`
`received_at, partition, offset, error, raw`. `error` is `<field>: <message>`; `raw` is the original payload
(up to 2 KB) for replay.

## Graph snapshots (`data/lake/graph/`, rewritten every run)
| Table | Grain | Main columns |
|---|---|---|
| `rings.parquet` | ring | `ring_id, status` (high/medium/low)`, ring_score, n_cards, n_devices, n_txns, amount_usd, top_merchant, top_category, collector_concentration, champion_flag_rate, challenger_mean, confirmed_fraud_share, first_seen, last_seen, detected_at` |
| `ring_members.parquet` | ring × card | `n_txns, amount_usd, home_country, max_risk, max_challenger, confirmed_fraud` |
| `ring_edges.parquet` | ring × card × device | `device_id, n_txns` |

## Reference data (`data/reference/`)
- `cards.parquet`: `card_id, customer_id, home_city, home_country, home_lat, home_lon, home_currency, profile_avg_usd, enrolled_device, tier`
- `merchants.parquet`: `merchant_id, merchant_name, merchant_category`

## Model files (`models/`)
| File | Content |
|---|---|
| `fraud_model.joblib` | Champion `ModelBundle`: Isolation Forest, version, training anomaly scores, feature scaling, importances, metrics |
| `challenger.joblib` | Challenger bundle: model, version, feature list, holdout comparison |
| `reference_profile.json` | Per feature and score: decile `edges`, `expected` share per bin, training `mean` |
| `model_card.json` | Champion metrics and the `champion_vs_challenger` holdout comparison |

## Monitoring report (`data/monitoring/report.json`)
`generated_at`, `window_hours`, `drift_window_hours`, `drift[]` (`feature, psi, status, ref_mean, live_mean,
kind`), `drift_summary`, `matured_rows`, `comparison` (PR-AUC, ROC-AUC, precision and recall for both models at
the shared `flag_rate`, plus `recall_by_type`, `pr_auc_gain`, `verdict`, `reason`), `status`.
