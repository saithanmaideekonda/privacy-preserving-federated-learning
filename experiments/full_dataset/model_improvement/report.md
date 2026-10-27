# Controlled class-weighting experiment — validation-only selection

The four runs used 1203.2 seconds in total for training plus epoch validation. Peak sampled working set was 1.36 GiB.

All candidates use the same Phase 1 training/validation partitions, 44-input [128, 64] MLP, seed, initialization, minibatch order, epoch budget, and early-stopping rule. Only class weighting varies. Checkpoint selection uses validation macro-F1. Test arrays and labels were not loaded for these candidates, and no test-set evaluation was run.

## Validation comparison

| Rank | Strategy | Epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Runtime (s) | Uploading R/F1 | XSS R/F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | reproduction_control | 12 | 0.9713 | 0.6637 | 0.6909 | 0.6697 | 0.9711 | 293.8 | 0.000/0.000 | 0.043/0.032 |
| 2 | effective_number_beta_09999 | 9 | 0.9726 | 0.6763 | 0.6946 | 0.6647 | 0.9722 | 366.7 | 0.135/0.020 | 0.076/0.039 |
| 3 | inverse_capped_1_cap10 | 12 | 0.8929 | 0.6710 | 0.6419 | 0.6302 | 0.8897 | 274.4 | 0.000/0.000 | 0.000/0.000 |
| 4 | inverse_power_075 | 9 | 0.8663 | 0.6149 | 0.6805 | 0.6185 | 0.8649 | 268.3 | 0.324/0.026 | 0.120/0.045 |

## Stage A weak-class results

The following table reports validation precision/recall/F1 and support for every class ranked among the 12 weakest in Stage A. Full 34-class per-class results and confusion matrices remain in each strategy folder. These rows use validation data only.

| Strategy | Class | Support | Precision | Recall | F1 | Δ F1 vs control |
|---|---|---:|---:|---:|---:|---:|
| reproduction_control | Uploading_Attack | 37 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| effective_number_beta_09999 | Uploading_Attack | 37 | 0.0106 | 0.1351 | 0.0196 | +0.0196 |
| inverse_capped_1_cap10 | Uploading_Attack | 37 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| inverse_power_075 | Uploading_Attack | 37 | 0.0134 | 0.3243 | 0.0257 | +0.0257 |
| reproduction_control | Recon-PingSweep | 45 | 0.0169 | 0.0222 | 0.0192 | +0.0000 |
| effective_number_beta_09999 | Recon-PingSweep | 45 | 0.0776 | 0.2000 | 0.1118 | +0.0926 |
| inverse_capped_1_cap10 | Recon-PingSweep | 45 | 0.0000 | 0.0000 | 0.0000 | -0.0192 |
| inverse_power_075 | Recon-PingSweep | 45 | 0.0475 | 0.3778 | 0.0844 | +0.0651 |
| reproduction_control | XSS | 92 | 0.0258 | 0.0435 | 0.0324 | +0.0000 |
| effective_number_beta_09999 | XSS | 92 | 0.0260 | 0.0761 | 0.0388 | +0.0064 |
| inverse_capped_1_cap10 | XSS | 92 | 0.0000 | 0.0000 | 0.0000 | -0.0324 |
| inverse_power_075 | XSS | 92 | 0.0281 | 0.1196 | 0.0455 | +0.0131 |
| reproduction_control | Backdoor_Malware | 96 | 0.1220 | 0.1042 | 0.1124 | +0.0000 |
| effective_number_beta_09999 | Backdoor_Malware | 96 | 0.0431 | 0.4583 | 0.0787 | -0.0336 |
| inverse_capped_1_cap10 | Backdoor_Malware | 96 | 0.0000 | 0.0000 | 0.0000 | -0.1124 |
| inverse_power_075 | Backdoor_Malware | 96 | 0.0538 | 0.4062 | 0.0950 | -0.0174 |
| reproduction_control | SqlInjection | 148 | 0.1258 | 0.2568 | 0.1689 | +0.0000 |
| effective_number_beta_09999 | SqlInjection | 148 | 0.0839 | 0.4257 | 0.1402 | -0.0287 |
| inverse_capped_1_cap10 | SqlInjection | 148 | 0.1429 | 0.0068 | 0.0129 | -0.1560 |
| inverse_power_075 | SqlInjection | 148 | 0.0738 | 0.3986 | 0.1246 | -0.0443 |
| reproduction_control | BrowserHijacking | 155 | 0.1706 | 0.1871 | 0.1785 | +0.0000 |
| effective_number_beta_09999 | BrowserHijacking | 155 | 0.0966 | 0.1097 | 0.1027 | -0.0757 |
| inverse_capped_1_cap10 | BrowserHijacking | 155 | 0.6000 | 0.0581 | 0.1059 | -0.0726 |
| inverse_power_075 | BrowserHijacking | 155 | 0.0684 | 0.2065 | 0.1027 | -0.0757 |
| reproduction_control | Recon-OSScan | 2,488 | 0.4114 | 0.1568 | 0.2270 | +0.0000 |
| effective_number_beta_09999 | Recon-OSScan | 2,488 | 0.3657 | 0.1258 | 0.1872 | -0.0398 |
| inverse_capped_1_cap10 | Recon-OSScan | 2,488 | 0.3375 | 0.2267 | 0.2712 | +0.0442 |
| inverse_power_075 | Recon-OSScan | 2,488 | 0.2377 | 0.2548 | 0.2460 | +0.0190 |
| reproduction_control | DictionaryBruteForce | 290 | 0.3090 | 0.1897 | 0.2350 | +0.0000 |
| effective_number_beta_09999 | DictionaryBruteForce | 290 | 0.1922 | 0.1862 | 0.1891 | -0.0459 |
| inverse_capped_1_cap10 | DictionaryBruteForce | 290 | 0.6133 | 0.1586 | 0.2521 | +0.0170 |
| inverse_power_075 | DictionaryBruteForce | 290 | 0.0818 | 0.3034 | 0.1288 | -0.1062 |
| reproduction_control | CommandInjection | 126 | 0.2778 | 0.2381 | 0.2564 | +0.0000 |
| effective_number_beta_09999 | CommandInjection | 126 | 0.5000 | 0.2222 | 0.3077 | +0.0513 |
| inverse_capped_1_cap10 | CommandInjection | 126 | 0.9583 | 0.1825 | 0.3067 | +0.0503 |
| inverse_power_075 | CommandInjection | 126 | 0.3684 | 0.2222 | 0.2772 | +0.0208 |
| reproduction_control | Recon-PortScan | 2,081 | 0.3591 | 0.3926 | 0.3751 | +0.0000 |
| effective_number_beta_09999 | Recon-PortScan | 2,081 | 0.4462 | 0.3128 | 0.3678 | -0.0073 |
| inverse_capped_1_cap10 | Recon-PortScan | 2,081 | 0.3614 | 0.4296 | 0.3925 | +0.0174 |
| inverse_power_075 | Recon-PortScan | 2,081 | 0.3679 | 0.4142 | 0.3897 | +0.0146 |
| reproduction_control | DNS_Spoofing | 4,543 | 0.4804 | 0.4026 | 0.4381 | +0.0000 |
| effective_number_beta_09999 | DNS_Spoofing | 4,543 | 0.5689 | 0.2864 | 0.3810 | -0.0571 |
| inverse_capped_1_cap10 | DNS_Spoofing | 4,543 | 0.4207 | 0.5140 | 0.4627 | +0.0246 |
| inverse_power_075 | DNS_Spoofing | 4,543 | 0.4172 | 0.5364 | 0.4693 | +0.0312 |
| reproduction_control | Recon-HostDiscovery | 3,459 | 0.5597 | 0.7696 | 0.6481 | +0.0000 |
| effective_number_beta_09999 | Recon-HostDiscovery | 3,459 | 0.7422 | 0.5953 | 0.6607 | +0.0126 |
| inverse_capped_1_cap10 | Recon-HostDiscovery | 3,459 | 0.5479 | 0.7765 | 0.6425 | -0.0056 |
| inverse_power_075 | Recon-HostDiscovery | 3,459 | 0.6794 | 0.6635 | 0.6713 | +0.0233 |

## Finding and next-step recommendation

The reproduction control ranked first at validation macro-F1 0.669651; no alternative weighting strategy improved it. Effective-number weighting was closest at 0.664704, but it remains below the control.

Keep the Phase 2 weighting/model as the current validation reference. If approved next, evaluate training-parameter changes in a separate validation-only experiment; do not tune on the test split.

## Interpretation and limitations

- The reproduction control exactly matches the Phase 2 saved validation macro-F1 (0.6696508820), which verifies the reproduced weighting/training path.
- Uploading_Attack has 138 training and 37 validation examples; XSS has 414 and 92. Small support makes per-class validation estimates noisy.
- Stronger weighting improved recall for some rare classes but reduced overall validation macro-F1 and, in some candidates, sharply reduced other weak-class F1. Reweighting cannot add rare-class variation and may increase false positives.
- No candidate was evaluated on the test split in this stage. The original Phase 2 and Phase 3 checkpoints, reports, prepared arrays, and Phase 1–5 artifacts were not modified.
- See `weak_class_validation_comparison.csv` and the per-strategy `per_class_validation_metrics.csv` files for auditable detail.
