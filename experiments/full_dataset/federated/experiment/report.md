# Phase 3 — Full-dataset FedAvg report

**Status:** completed  
**Created:** 2026-10-26T11:24:41.801085+00:00  
**Rows:** 5,492,542 train; 1,176,701 validation; 1,176,430 test.  
**Model:** same [128, 64] sklearn MLP shape as Phase 2, 1 local epoch(s), 12 rounds.  
**Selection:** round 12 by validation macro-F1; test was not used for selection.

## Method and compatibility

The Phase 2 checkpoint was verified to have compatible layer shapes, but its learned parameters were not used because they were trained on all training rows. FedAvg starts from one fresh initialization with the same architecture/seed; all clients begin each round from the same global parameters. The server averages only weight and bias arrays using each client’s fraction of training rows. Client Adam state is reset each round and is not averaged.

Clients are simulated in one process using disjoint row-index partitions and minibatches from the prepared training memmap. This does not provide machine-level isolation, secure aggregation, or a formal privacy guarantee.

## Client distribution

| Client | Rows | Classes | Uploading_Attack | XSS | Recon-PingSweep |
|---|---:|---:|---:|---:|---:|
| 1 | 1,783,912 | 33 | 85 | 125 | 38 |
| 2 | 1,685,958 | 34 | 37 | 119 | 176 |
| 3 | 2,022,672 | 34 | 16 | 170 | 33 |

Full client class counts are in `client_partitions.json`.

## Validation checkpoint

| Metric | Validation |
|---|---:|
| Accuracy | 0.804130 |
| Macro Precision | 0.634753 |
| Macro Recall | 0.598006 |
| Macro F1 | 0.576287 |
| Weighted F1 | 0.782445 |

### Validation comparison (checkpoint selection split)

Both approaches use the same prepared validation rows; these values guide checkpoint selection.

| Metric | FedAvg selected round | Centralized selected epoch | FedAvg - centralized |
|---|---:|---:|---:|
| Accuracy | 0.804130 | 0.971271 | -0.167141 |
| Macro Precision | 0.634753 | 0.663704 | -0.028952 |
| Macro Recall | 0.598006 | 0.690892 | -0.092886 |
| Macro F1 | 0.576287 | 0.669651 | -0.093364 |
| Weighted F1 | 0.782445 | 0.971076 | -0.188631 |

## Final test evaluation

The validation-selected checkpoint was evaluated on the prepared test split after the protocol was fixed.

| Metric | FedAvg | Centralized | Difference |
|---|---:|---:|---:|
| Accuracy | 0.803753 | 0.970913 | -0.167160 |
| Macro Precision | 0.624161 | 0.667388 | -0.043227 |
| Macro Recall | 0.596092 | 0.695633 | -0.099542 |
| Macro F1 | 0.575443 | 0.673290 | -0.097847 |
| Weighted F1 | 0.781980 | 0.970753 | -0.188774 |

All 34 classes occur in the test split by Phase 1 audit. Rare-class scores remain uncertain; Uploading_Attack has 33 test rows and XSS 102. See per-class metrics and confusion matrix artifacts.

## Runtime and communication

Training and validation took 261.9 seconds across 12 rounds. Stop reason: `configured_round_limit`.
Estimated parameter upload/download traffic: 4,673,088 bytes; transport/security overhead is excluded.

## Limitations

- Class-wise Dirichlet partitioning is a reproducible simulation, not evidence of real organizational data separation.
- The training distribution is highly imbalanced. Global inverse-square-root weights cannot create rare-class diversity.
- Phase 1 grouping uses exact feature-token fingerprints; no authentic device, session, or time IDs exist.
- sklearn’s private `_initialize` API is version-sensitive; the run records versions and tests this initialization.
- One fixed-seed run does not establish that FedAvg outperforms the centralized baseline.
