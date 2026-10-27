# Phase 3 — Full-dataset FedAvg report

**Status:** pilot_only_no_test_evaluation  
**Created:** 2026-10-26T11:19:55.709793+00:00  
**Rows:** 5,492,542 train; 1,176,701 validation; 1,176,430 test.  
**Model:** same [128, 64] sklearn MLP shape as Phase 2, 1 local epoch(s), 1 rounds.  
**Selection:** round 1 by validation macro-F1; test was not used for selection.

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
| Accuracy | 0.731014 |
| Macro Precision | 0.466146 |
| Macro Recall | 0.461970 |
| Macro F1 | 0.416797 |
| Weighted F1 | 0.680174 |

## Final test evaluation

Validation-only resource pilot: no test arrays or labels were evaluated.

## Runtime and communication

Training and validation took 22.1 seconds across 1 rounds. Stop reason: `configured_round_limit`.
Estimated parameter upload/download traffic: 389,424 bytes; transport/security overhead is excluded.

## Limitations

- Class-wise Dirichlet partitioning is a reproducible simulation, not evidence of real organizational data separation.
- The training distribution is highly imbalanced. Global inverse-square-root weights cannot create rare-class diversity.
- Phase 1 grouping uses exact feature-token fingerprints; no authentic device, session, or time IDs exist.
- sklearn’s private `_initialize` API is version-sensitive; the run records versions and tests this initialization.
- One fixed-seed run does not establish that FedAvg outperforms the centralized baseline.
