# Phase 1 — Full-dataset preparation report

**Preparation date:** 2026-10-26  
**Status:** split construction and preprocessing only; no training was run.  
**Seed:** 20261026

## Summary

Created derived arrays from all 7,845,673 rows in the supplied source CSVs. The original files and sampled experiments were read only. Every well-formed, non-empty row was retained once. Derived sizes: **5,492,542 train**, **1,176,701 validation**, and **1,176,430 test**.

The schema has no stable device, capture, session, timestamp, or flow ID. This is an **exact-feature-group-aware split**, not a claim of device-, capture-, session-, or time-level independence. Related but non-identical flows may still cross splits.

## Duplicate and overlap audit

The earlier SQLite fingerprint audit stores one row per distinct fingerprint with occurrence counters. Its overlap values are counts of **distinct shared fingerprints**, not all matching row occurrences. Pair counts are not additive because a fingerprint present in all three files appears in more than one pair.

| Source split | Repeated exact-row fingerprints | Extra exact-row occurrences | Repeated feature vectors |
| --- | ---: | ---: | ---: |
| train | 134565 | 134565 | 134565 |
| validation | 6297 | 6297 | 6297 |
| test | 6384 | 6384 | 6384 |

| Pair | Shared exact-row fingerprints | Shared feature-vector fingerprints |
| --- | ---: | ---: |
| train ↔ validation | 57,744 | 57,744 |
| train ↔ test | 57,847 | 57,847 |
| validation ↔ test | 12,422 | 12,422 |

The streamed Phase 1 check examined 275,259 repeated or shared feature groups for label consistency and found **0 groups with conflicting labels**. Repeated vectors are not assumed erroneous: none were deleted, and all repeats follow their group's assignment.

## Split methodology

For each SHA-256 fingerprint of the original 46 feature tokens (label excluded), assign the entire group to the source split with the largest occurrence count. Resolve ties with a deterministic SHA-256 choice and seed 20261026. Groups unique to one source split remain there. No records are duplicated or oversampled.

| Original file | To train | To validation | To test |
| --- | ---: | ---: | ---: |
| train | 5,434,461 | 28,785 | 28,725 |
| validation | 28,959 | 1,141,693 | 6,199 |
| test | 29,122 | 6,223 | 1,141,506 |

All occurrences of one serialized-token feature fingerprint have one destination, so overlap for that exact fingerprint definition is **0 by construction**. The digest is computed from the original numeric text tokens, not canonicalized float values; alternate text encodings of equal numeric values were not separately merged. The procedure retains all rows and conservatively reconciles the supplied splits; it is not a new random split from a known population.

## Derived class support

| Split | Rows | Class coverage | Missing classes |
| --- | ---: | ---: | --- |
| train | 5,492,542 | 34/34 | None |
| validation | 1,176,701 | 34/34 | None |
| test | 1,176,430 | 34/34 | None |

Training support ranges from Uploading_Attack (138) to DDoS-ICMP_Flood (847,902), about 6,144:1.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Backdoor_Malware | 386 | 96 | 92 |
| BenignTraffic | 129,502 | 27,574 | 27,690 |
| BrowserHijacking | 669 | 155 | 133 |
| CommandInjection | 627 | 126 | 118 |
| DDoS-ACK_Fragmentation | 33,622 | 7,177 | 7,268 |
| DDoS-HTTP_Flood | 3,391 | 693 | 702 |
| DDoS-ICMP_Flood | 847,902 | 182,265 | 180,379 |
| DDoS-ICMP_Fragmentation | 53,023 | 11,462 | 11,393 |
| DDoS-PSHACK_Flood | 481,124 | 103,101 | 103,340 |
| DDoS-RSTFINFlood | 475,551 | 101,470 | 101,802 |
| DDoS-SYN_Flood | 478,747 | 102,515 | 102,243 |
| DDoS-SlowLoris | 2,749 | 626 | 624 |
| DDoS-SynonymousIP_Flood | 422,137 | 90,726 | 90,495 |
| DDoS-TCP_Flood | 528,545 | 113,898 | 113,679 |
| DDoS-UDP_Flood | 637,636 | 136,453 | 136,652 |
| DDoS-UDP_Fragmentation | 34,181 | 7,188 | 7,245 |
| DNS_Spoofing | 21,224 | 4,543 | 4,582 |
| DictionaryBruteForce | 1,543 | 290 | 317 |
| DoS-HTTP_Flood | 8,491 | 1,838 | 1,796 |
| DoS-SYN_Flood | 237,621 | 51,085 | 51,098 |
| DoS-TCP_Flood | 314,444 | 66,924 | 67,559 |
| DoS-UDP_Flood | 390,514 | 83,356 | 83,625 |
| MITM-ArpSpoofing | 36,357 | 7,742 | 7,798 |
| Mirai-greeth_flood | 116,202 | 24,869 | 24,906 |
| Mirai-greip_flood | 88,773 | 18,913 | 19,281 |
| Mirai-udpplain | 104,800 | 22,365 | 22,499 |
| Recon-HostDiscovery | 15,743 | 3,459 | 3,348 |
| Recon-OSScan | 11,582 | 2,488 | 2,450 |
| Recon-PingSweep | 247 | 45 | 55 |
| Recon-PortScan | 9,663 | 2,081 | 2,069 |
| SqlInjection | 590 | 148 | 148 |
| Uploading_Attack | 138 | 37 | 33 |
| VulnerabilityScan | 4,404 | 901 | 909 |
| XSS | 414 | 92 | 102 |

No examples were added to force class coverage. Low-support classes will still have uncertain later per-class metrics.

## Data quality

| Source | Missing feature values | Infinite values | Invalid numeric tokens | Rows |
| --- | ---: | ---: | ---: | ---: |
| train | 0 | 0 | 0 | 5,491,971 |
| validation | 0 | 0 | 0 | 1,176,851 |
| test | 0 | 0 | 0 | 1,176,851 |

The three files were streamed and checked for schema width, numeric parseability, missing feature values, and infinities. Non-finite or non-numeric feature values are treated as missing for the train-fitted imputer. Malformed rows cause a clear failure instead of being silently discarded.

## Preprocessing

- Separate the label from the 46 numeric feature columns.
- Fit label mapping using derived training labels only. A label not seen in derived train is recorded with ID -1 in validation/test.
- Fit per-feature medians on derived training rows only; apply fixed medians to validation/test.
- Fit StandardScaler incrementally on imputed derived training rows only; apply the saved transform unchanged to validation/test.
- Identify constant or entirely missing columns using derived training only. Removed: Telnet (constant in derived training data); IRC (constant in derived training data)
- Save X arrays as float32 memory-mappable NumPy files and y arrays as int32.

Test fingerprints are used only to reconcile identical-vector overlap. Test labels do not fit preprocessing or select models. Output metadata records the fitted mapping and numeric transforms.

## Limitations

1. Exact-feature grouping cannot replace missing device/capture/session/time identifiers; similar traffic with different values may still cross splits.
2. Majority-source assignment retains the provided split structure where possible, but does not prove the original assignments were random or independent.
3. Class imbalance is severe and later per-class metrics must be interpreted with support.
4. No classifier, FedAvg, drift detector, selective retraining, or dashboard was run in this phase.

## Next step

Given the available schema, exact-token feature grouping is the strongest defensible local leakage mitigation; do not invent device or event IDs. For stronger group-aware generalization, obtain authentic capture/device/session identifiers and split by those groups. Review and approve this policy before Phase 2. Keep the derived test arrays reserved for final evaluation.
