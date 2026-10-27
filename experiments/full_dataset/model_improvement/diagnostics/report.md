# Stage A — Baseline rare-class diagnostics

**Ranking uses Phase 2 validation F1 only.** Saved Phase 2 test per-class metrics and confusion counts are included only as retrospective diagnostics requested by the project owner; no test arrays or labels were opened, and test results are not used for experiment configuration or selection.

Verified: 34 contiguous class IDs; 44 retained features; train/validation/test rows 5,492,542/1,176,701/1,176,430. Existing centralized test accuracy/macro-F1/weighted-F1 = 0.970913/0.673290/0.970753. The test split is otherwise reserved for the separately authorized final evaluation stage.

## Weakest classes by validation F1

| Rank | Class | Train n | Val n | Val P | Val R | Val F1 | Top saved test error destinations (diagnostic only) |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Uploading_Attack | 138 | 37 | 0.0000 | 0.0000 | 0.0000 | Recon-HostDiscovery:12; DNS_Spoofing:8; BenignTraffic:6 |
| 2 | Recon-PingSweep | 247 | 45 | 0.0169 | 0.0222 | 0.0192 | BenignTraffic:13; Recon-OSScan:10; MITM-ArpSpoofing:7 |
| 3 | XSS | 414 | 92 | 0.0258 | 0.0435 | 0.0324 | Recon-HostDiscovery:30; DNS_Spoofing:24; BenignTraffic:13 |
| 4 | Backdoor_Malware | 386 | 96 | 0.1220 | 0.1042 | 0.1124 | DNS_Spoofing:19; Recon-HostDiscovery:17; BenignTraffic:16 |
| 5 | SqlInjection | 590 | 148 | 0.1258 | 0.2568 | 0.1689 | Recon-PortScan:23; Recon-HostDiscovery:21; BenignTraffic:16 |
| 6 | BrowserHijacking | 669 | 155 | 0.1706 | 0.1871 | 0.1785 | BenignTraffic:44; DNS_Spoofing:19; Recon-HostDiscovery:19 |
| 7 | Recon-OSScan | 11,582 | 2,488 | 0.4114 | 0.1568 | 0.2270 | BenignTraffic:1065; Recon-PortScan:583; Recon-HostDiscovery:209 |
| 8 | DictionaryBruteForce | 1,543 | 290 | 0.3090 | 0.1897 | 0.2350 | BenignTraffic:84; Recon-HostDiscovery:41; Recon-PortScan:38 |
| 9 | CommandInjection | 627 | 126 | 0.2778 | 0.2381 | 0.2564 | BenignTraffic:22; Recon-HostDiscovery:21; DNS_Spoofing:20 |
| 10 | Recon-PortScan | 9,663 | 2,081 | 0.3591 | 0.3926 | 0.3751 | BenignTraffic:421; Recon-HostDiscovery:330; Recon-OSScan:230 |
| 11 | DNS_Spoofing | 21,224 | 4,543 | 0.4804 | 0.4026 | 0.4381 | BenignTraffic:963; MITM-ArpSpoofing:867; Recon-HostDiscovery:554 |
| 12 | Recon-HostDiscovery | 15,743 | 3,459 | 0.5597 | 0.7696 | 0.6481 | BenignTraffic:226; DNS_Spoofing:224; MITM-ArpSpoofing:110 |

## Interpretation

The current MLP uses inverse-square-root class weighting, normalized to a training-support-weighted mean of one. Training support spans 138 to 847,902 rows (about 6,144:1); weighting changes gradient influence but cannot supply missing rare-class variation.
The baseline shuffles row indices reproducibly each epoch, then trains in 8,192-row batches with `shuffle=False` inside scikit-learn. Early stopping/checkpoint selection uses validation macro-F1, which is appropriate for class coverage but can be noisy for validation classes with fewer than 100 examples. Training has a 12-epoch maximum and patience 3; the saved run selected epoch 12, so it did not stop before the budget ended.
The saved confusion matrices show whether weak classes are confused with BenignTraffic, frequent attacks, or other rare classes. See `weak_class_diagnostics.csv`; its validation confusion destinations inform training decisions, while test destinations are labeled diagnostic-only.
New comparisons retain 44 inputs, [128,64] hidden layers, 34 outputs, seed 20261026, the same row splits, Adam, batch size 8,192, 12-epoch maximum, and patience 3. Only class weighting changes. No PyTorch or focal-loss dependency will be added.
