# Phase 2 — Centralized baseline report

**Model:** CPU scikit-learn MLPClassifier, hidden layers [128, 64].  
**Created:** 2026-10-26T11:04:12.713343+00:00  
**Training rows:** 5,492,542; validation rows: 1,176,701; test rows: 1,176,430.  
**Seed:** 20261026

## Method

The classifier uses ReLU hidden layers and Adam with 8,192-row minibatches. It consumes the Phase 1 float32 memmaps without materializing the complete feature matrix in RAM. Class imbalance is handled with normalized inverse-square-root training-frequency sample weights. This moderates the majority-class dominance without duplicating rare records; it does not make the rare classes adequately represented, and the weighting can affect calibration and false positives.

The best checkpoint is selected by validation macro-F1. Training stops after 3 epochs without improvement, up to 12 epochs. The test split was not used for training, checkpoint selection, or tuning.

## Validation checkpoint selection

Selected epoch: **12**; validation macro-F1: **0.6697**. Test used for selection: **No**.

## Final test evaluation

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.9709 |
| Macro precision | 0.6674 |
| Macro recall | 0.6956 |
| Macro-F1 | 0.6733 |
| Weighted-F1 | 0.9708 |

All 34 classes were present in the final test split according to the Phase 1 support audit. The per-class table and confusion matrix are saved as separate artifacts. Do not interpret the high-support weighted score as uniform performance across classes.

### Lowest per-class F1 among supported test classes

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Uploading_Attack | 0.0000 | 0.0000 | 0.0000 | 33 |
| XSS | 0.0286 | 0.0490 | 0.0361 | 102 |
| Backdoor_Malware | 0.0909 | 0.0761 | 0.0828 | 92 |
| Recon-PingSweep | 0.1538 | 0.1091 | 0.1277 | 55 |
| BrowserHijacking | 0.1795 | 0.2105 | 0.1938 | 133 |
| Recon-OSScan | 0.3622 | 0.1449 | 0.2070 | 2,450 |
| SqlInjection | 0.1672 | 0.3649 | 0.2293 | 148 |
| CommandInjection | 0.3133 | 0.2203 | 0.2587 | 118 |

## Runtime and resources

- Training plus per-epoch validation: 885.1 seconds.
- Selected-checkpoint validation recheck: not separately timed.
- Final test prediction and metric calculation: not separately timed.
- Post-selection validation, test, and artifact finalization: 7.9 seconds.
- Total elapsed through evaluation artifacts: 893.1 seconds.
- Peak measured working set: 1.35 GiB where available.
- Scikit-learn warned on the smaller final minibatch and clipped its configured size; the minibatch was processed and no rows were skipped. The preparation code now sets that final size explicitly to suppress the warning with equivalent processing.
- Parameter count: 16,226; estimated float64 parameter storage: 129,808 bytes.
- Training feature memmap: 966,687,520 bytes; each minibatch input: 1,441,792 bytes.
- Peak working-set samples are recorded in run_metadata.json when available.

## Limitations

1. The class distribution is highly imbalanced. The rarest training class has only 138 examples, so its metrics are uncertain. Weighting cannot replace missing diversity.
2. The split policy groups exact serialized feature-token fingerprints; it is not device-, session-, or time-aware.
3. The baseline is a single-machine CPU experiment, not a federated or privacy-preserving deployment.
4. This run evaluates the supplied Phase 1 derived split, not the original sampled experiment.

## Artifacts

See checkpoints/best_model.joblib, training_history.csv, validation_metrics_by_epoch.csv, best_validation_metrics.json, per_class_validation_metrics.csv, confusion_matrix_validation.csv, test_metrics.json, per_class_test_metrics.csv, confusion_matrix_test.csv, confusion_matrix_test.npy, test_predictions.npy, run_metadata.json, and training.log.
