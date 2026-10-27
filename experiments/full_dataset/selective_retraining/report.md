# Phase 5 — PSI-triggered selective retraining

**Created:** 2026-10-26T11:43:04.840026+00:00  
**Runtime:** 6.26 seconds.  
**Scope:** Phase 4 no-drift and synthetic-shift scenarios only; no natural temporal drift was tested.

## Policy

- PSI threshold **0.25**, read unchanged from Phase 4.
- Trigger after **2 consecutive alert windows** for one client.
- Cooldown: **3 windows after a trigger**.
- Maximum **1 action(s) per client/scenario**.
- State is isolated per client and per scenario. Trigger decisions are logged separately from training and candidate selection.

## Scenario decisions

| Scenario | Alerts | Pending | Triggers | Cooldown-suppressed | Max-frequency-suppressed | Actual training runs | Accepted candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_drift | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| synthetic_shift | 5 | 1 | 1 | 3 | 0 | 1 | 0 |

### Alert-level decisions

| Scenario | Client | Window | PSI | Decision | Reason | Training run |
|---|---:|---:|---:|---|---|---|
| synthetic_shift | 2 | 4 | 12.433835 | pending_consecutive_alerts | streak 1; requires 2 | no |
| synthetic_shift | 2 | 5 | 12.433835 | retraining_triggered | 2 consecutive alerts reached policy requirement | yes |
| synthetic_shift | 2 | 6 | 12.433835 | suppressed_cooldown | cooldown active through window 8 | no |
| synthetic_shift | 2 | 7 | 12.433835 | suppressed_cooldown | cooldown active through window 8 | no |
| synthetic_shift | 2 | 8 | 12.433835 | suppressed_cooldown | cooldown active through window 8 | no |

## Candidate: Client 2 — synthetic_shift

A real local training run was performed after the trigger at window 5. It fitted on 8,192 rows from the affected client's training partition, taken from the two consecutive alert windows. No validation or test rows were used for fitting.
Synthetic shift applied to fitting rows: **yes** (Rate, +5.0 standardized units). Only Client 2 was retrained; clients 1, 3 were skipped. Their parameters remained at the Phase 3 global checkpoint and were used unchanged in the weighted candidate aggregate.
Validation target macro-F1: **0.521020 → 0.518312** (-0.002707). Clean validation macro-F1: **0.576287 → 0.569188** (drop 0.007099).
Candidate: **rejected** under the configured validation rules. It was not deployed. Selected checkpoint remains the original Phase 3 FedAvg checkpoint.
No Phase 5 test evaluation was performed because the candidate was rejected. Existing Phase 3 test results remain unchanged.

## Data use and limitations

- Decisions consume Phase 4 `drift_scores.csv` and `event_log.json`; the PSI threshold and detector are unchanged.
- Candidate fitting uses only affected-client training rows from the alert windows. Validation selection uses prepared validation, with a client-2 synthetic shifted view when evaluating the synthetic-shift scenario.
- The test loader is guarded and is called only after validation accepts a candidate.
- Dataset records have no reliable device/session/time identifiers. All streams and drift events here are synthetic; natural temporal drift is not validated.
- Drift indicates a feature-distribution change and does not confirm an intrusion. Simulated clients on one computer do not provide formal privacy or network isolation.
