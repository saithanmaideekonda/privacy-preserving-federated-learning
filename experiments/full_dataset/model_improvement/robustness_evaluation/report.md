# Stage 3 — Validation robustness and final candidate evaluation

## Protocol

Two additional predeclared seeds (20261028, 20261029) used the unchanged Stage 2 run_candidate implementation: learning rate 0.002, 12-epoch maximum, patience 3, Phase 2 class weights, 8,192-row batches, [128, 64] MLP, and identical prepared train/validation splits. Training updates used training only; validation selected checkpoints. Test arrays were not opened before selection was frozen.
Four seeds are a small exploratory check, not formal evidence of robustness.

## Validation results

| Seed | Epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261026 | 12 | 0.982985 | 0.685251 | 0.714297 | 0.691748 | 0.982779 | 191.9 |
| 20261027 | 11 | 0.977941 | 0.683965 | 0.701971 | 0.684908 | 0.978125 | 205.4 |
| 20261028 | 12 | 0.974075 | 0.676646 | 0.713317 | 0.682134 | 0.973946 | 190.3 |
| 20261029 | 10 | 0.979248 | 0.672562 | 0.703898 | 0.681391 | 0.979164 | 196.2 |

| Metric | Mean | Sample SD | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.978562 | 0.003677 | 0.974075 | 0.982985 | 0.008911 |
| macro_precision | 0.679606 | 0.006034 | 0.672562 | 0.685251 | 0.012688 |
| macro_recall | 0.708371 | 0.006339 | 0.701971 | 0.714297 | 0.012325 |
| macro_f1 | 0.685045 | 0.004718 | 0.681391 | 0.691748 | 0.010357 |
| weighted_f1 | 0.978504 | 0.003635 | 0.973946 | 0.982779 | 0.008833 |

Phase 2 reproduction validation macro-F1 was 0.669651; Stage 2 learning-rate 0.002 seeds scored 0.691748 and 0.684908. Additional seeds are reported above.

## Validation-only freeze

Selection rule: passed. Frozen source: Stage 2 preselected learning-rate 0.002 checkpoint, seed 20261026. Validation macro-F1: 0.691748. Frozen checkpoint SHA-256: 47984db438d22952a8624c675cfc31cae67885f1e0e016f2186679fffd52b88a.

| Predeclared gate | Result |
|---|:---:|
| mean macro f1 exceeds baseline | Pass |
| three of four macro f1 exceed baseline | Pass |
| mean macro recall not lower | Pass |
| accuracy within 1pp | Pass |
| weighted f1 within 1pp | Pass |
| 12 weak class mean f1 within 1pp | Pass |
| uploading recall positive in two runs | Pass |

Uploading_Attack has 37 validation examples. Per-seed support, recall, F1, predicted counts, and leading confusion destinations are in validation_weak_classes_by_seed.csv. These small supports do not establish stable detection. Full per-class metrics are in validation_per_class_by_seed.csv.

## Final test evaluation after freeze

The frozen checkpoint was evaluated once on 1,176,430 prepared test rows. The Phase 2 baseline evaluation was reused from its saved metrics and confusion matrix, after checking that it covers the same row count and class mapping.

| Approach | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Frozen candidate | 0.982648 | 0.693131 | 0.714255 | 0.692903 | 0.982458 |
| Phase 2 centralized baseline | 0.970913 | 0.667388 | 0.695633 | 0.673290 | 0.970753 |

Per-class test metrics, confusion counts, rare-class prediction counts, and deltas are stored in adjacent machine-readable files.

## Limitations

- Four seeds are too few for formal reliability claims; rare-class validation support is especially small.
- Accuracy and weighted-F1 are dominated by common classes; macro metrics provide a complementary view.
- A single test evaluation does not establish dependable deployment detection.
- Previous Phase 1–5, Phase 2/3, and Stage 2 artifacts were checked with SHA-256 audits.

## Per-class comparison

Supports match because both confusion matrices summarize the same test rows. Undefined metrics remain blank; they do not mean zero performance.

| Class | Support | Candidate P/R/F1 | Phase 2 P/R/F1 | Delta F1 |
|---|---:|---:|---:|---:|
| Backdoor_Malware | 92 | 0.1757 / 0.1413 / 0.1566 | 0.0909 / 0.0761 / 0.0828 | 0.0738 |
| BrowserHijacking | 133 | 0.1637 / 0.2105 / 0.1842 | 0.1795 / 0.2105 / 0.1938 | -0.0096 |
| SqlInjection | 148 | 0.1617 / 0.3311 / 0.2173 | 0.1672 / 0.3649 / 0.2293 | -0.0120 |
| Uploading_Attack | 33 | 0.4000 / 0.0606 / 0.1053 | 0.0000 / 0.0000 / 0.0000 | 0.1053 |
| XSS | 102 | 0.0887 / 0.1078 / 0.0973 | 0.0286 / 0.0490 / 0.0361 | 0.0612 |

### Largest supported-class F1 changes

| Class | Support | Candidate F1 | Phase 2 F1 | Delta F1 |
|---|---:|---:|---:|---:|
| Uploading_Attack | 33 | 0.1053 | 0.0000 | +0.1053 |
| DDoS-HTTP_Flood | 702 | 0.7822 | 0.6943 | +0.0878 |
| VulnerabilityScan | 909 | 0.7303 | 0.6466 | +0.0837 |
| Backdoor_Malware | 92 | 0.1566 | 0.0828 | +0.0738 |
| DoS-HTTP_Flood | 1796 | 0.8813 | 0.8199 | +0.0614 |
| Recon-PingSweep | 55 | 0.0513 | 0.1277 | -0.0764 |
| SqlInjection | 148 | 0.2173 | 0.2293 | -0.0120 |
| BrowserHijacking | 133 | 0.1842 | 0.1938 | -0.0096 |
| DDoS-ICMP_Fragmentation | 11393 | 0.9796 | 0.9834 | -0.0039 |
| DDoS-ACK_Fragmentation | 7268 | 0.9837 | 0.9863 | -0.0026 |

## Runtime, memory and warnings

The two additional training runs took 386.6 seconds combined; peak sampled working set was 1.35 GiB.
The final test inference pass took 1.20 seconds.
No training warnings were recorded.
