# Phase 4 — PSI-based drift detection

**Created:** 2026-10-26T11:32:44.159069+00:00  
**Input:** 5,492,542 prepared training rows; 44 features. Validation and test were not used.  
**Runtime:** 0.88 seconds after input mapping and partition verification (the end-to-end command also includes startup/input loading).

## Detector

Each of the three Phase 3 simulated clients has its own reference distributions, fitted from 50,000 rows in its training partition. The 10 requested quantile bins are fitted feature by feature; repeated quantile cut-points collapse, so tied/discrete features can have fewer bins. Current windows use those fixed training-derived edges. Empty-bin proportions are floored at 1e-6 and renormalized before the PSI logarithm. A window alerts when its maximum feature PSI is at least 0.25.

The 0.25 cutoff is a predeclared operational heuristic, not a threshold tuned on held-out data or a statistical guarantee. No validation or test data was used to establish reference distributions or thresholds.

## Scenarios and results

No-drift windows contain unmodified training rows disjoint from each client’s reference sample. The synthetic scenario replays the same streams and adds +5.0 standardized units to `Rate` for Client 2 starting at window 4. Clients 1 and 3 are unshifted controls. This is a deliberately injected synthetic distribution change, not observed temporal drift.

| Scenario | Windows | Alerts | False-alarm windows | Shift windows | Detected shift windows | Missed shifts | Delay (windows) | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_drift | 24 | 0 | 0 | 0 | 0 | 0 | — | 0.000 |
| synthetic_shift | 24 | 5 | 0 | 5 | 5 | 0 | 0 | 0.000 |

Detection delay is measured from the injected shift’s first window (window 4); a value of zero means the first shifted window alerted. False alarms are alerts in windows without an injected shift. These rates characterize this finite synthetic replay only.

## Client-specific alert windows

| Scenario | Client | Window | Max PSI | Threshold | Top feature | Injected shift | Alert |
|---|---:|---:|---:|---:|---|---|---|
| synthetic_shift | 2 | 4 | 12.433835 | 0.25 | Rate | yes | yes |
| synthetic_shift | 2 | 5 | 12.433835 | 0.25 | Rate | yes | yes |
| synthetic_shift | 2 | 6 | 12.433835 | 0.25 | Rate | yes | yes |
| synthetic_shift | 2 | 7 | 12.433835 | 0.25 | Rate | yes | yes |
| synthetic_shift | 2 | 8 | 12.433835 | 0.25 | Rate | yes | yes |

Full window scores are in `drift_scores.csv`; all per-feature scores and bin counts are in `feature_psi_scores.csv`. Reference edges and histograms are in `reference_distributions.json`; triggered events are in `event_log.json`.

## Interpretation and limitations

**A detected distribution change is not proof of a cyberattack.** PSI detects changes in feature frequencies; benign traffic changes, sensor changes, or preprocessing differences can also trigger it.
- This is a synthetic train-derived replay. The dataset has no device/session/time IDs to establish a natural temporal stream, so the experiment does not validate real-world drift.
- The PSI threshold is a heuristic. The maximum across 44 feature tests can create false alarms; observed false-alarm windows are reported rather than hidden.
- Per-client references account for client-specific baseline distributions but can be weak for rare classes and small windows.
- The clients remain a single-computer simulation; this phase does not implement retraining or provide a privacy guarantee.
