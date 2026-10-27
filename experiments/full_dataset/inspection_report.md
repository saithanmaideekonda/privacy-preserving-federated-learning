# Full-dataset CICIoT2023 inspection report

**Inspection date:** 2026-10-26  
**Scope:** Read-only inspection of the dataset path provided by the user. No training was run. No file under the dataset directory or the prior project was modified.  
**Project workspace:** `C:\Users\saith\Documents\Codex\project-privacy-preserving-federated-learning-full-dataset`

## Executive summary

The local directory has three large CSV splits containing **7,845,673 rows total**, **47 columns** each (46 numeric features and `label`), and 34 labels in every split. All classes present in training also appear in validation and test. Missing values, infinities, and non-numeric feature tokens were not observed.

A disk-backed SHA-256 audit found substantial repeated records both within and across the supplied splits. Pairwise cross-split overlap is about 4.9% of validation/test rows against train and 1.1% between validation and test. Treat the supplied splits as **not leakage-free** until a group-aware split is prepared. The source CSVs remain unchanged.

The `sampled_data` files are verified smaller samples: 20,000 train rows and 5,000 rows each for validation/test. The included `sample.py` documents that those are class-proportional reservoir samples drawn from the three larger split files. The local evidence supports that the split files are the larger source inputs for sampling; it does **not** prove completeness relative to every record in the upstream Kaggle release because no upstream manifest or expected record count was available locally.

## Dataset inventory

Top-level entries found: `train/`, `validation/`, `test/`, `sampled_data/`, and `sample.py`. The three split directories each contain the named CSV shown below. All are comma-separated UTF-8 CSVs with the same header. Rows parsed by pandas matched physical record-line counts in the fingerprint audit; every physical data row had the expected 47 fields.

### Large split files

| Split | Exact path | Rows | Columns | Bytes | Size |
| --- | --- | --- | --- | --- | --- |
| Train | `C:\Users\saith\Downloads\archive (1)\CICIoT23\train\train.csv` | 5,491,971 | 47 | 1,623,446,296 | 1548.2 MiB |
| Validation | `C:\Users\saith\Downloads\archive (1)\CICIoT23\validation\validation.csv` | 1,176,851 | 47 | 347,921,865 | 331.8 MiB |
| Test | `C:\Users\saith\Downloads\archive (1)\CICIoT23\test\test.csv` | 1,176,851 | 47 | 347,958,926 | 331.8 MiB |

**Combined large-split size:** 2,319,327,087 bytes (2.16 GiB).  
**Combined rows:** 7,845,673.

### Prepared sample files found alongside the splits

| Sample split | Exact path | Rows checked from sample.py target | Bytes |
| --- | --- | --- | --- |
| Train | `C:\Users\saith\Downloads\archive (1)\CICIoT23\sampled_data\train_sample.csv` | 20,000 | 5,907,385 |
| Validation | `C:\Users\saith\Downloads\archive (1)\CICIoT23\sampled_data\validation_sample.csv` | 5,000 | 1,476,659 |
| Test | `C:\Users\saith\Downloads\archive (1)\CICIoT23\sampled_data\test_sample.csv` | 5,000 | 1,477,116 |

The sample target sizes and the sampling method are verified by reading `sample.py`; the sample CSVs have the expected headers and sizes. Their small row counts distinguish them from the large split files.

### Schema and target

- **Target column:** `label`
- **Feature columns:** 46
- **Columns per file:** 47
- **Numeric feature dtypes:** `float64` across all chunks in all three splits
- **Label dtype:** string/object across all chunks
- **Column names:** `flow_duration, Header_Length, Protocol Type, Duration, Rate, Srate, Drate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number, ack_count, syn_count, fin_count, urg_count, rst_count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitue, Radius, Covariance, Variance, Weight, label`

## Class distribution and coverage

All 34 training labels are present in validation and test; no class is absent from either split. Counts below are from chunked scans of the complete local split files.

| Class | Train | Validation | Test |
| --- | --- | --- | --- |
| Backdoor_Malware | 392 | 93 | 89 |
| BenignTraffic | 129,538 | 27,519 | 27,709 |
| BrowserHijacking | 665 | 158 | 134 |
| CommandInjection | 620 | 132 | 119 |
| DDoS-ACK_Fragmentation | 33,581 | 7,194 | 7,292 |
| DDoS-HTTP_Flood | 3,371 | 706 | 709 |
| DDoS-ICMP_Flood | 848,088 | 182,011 | 180,447 |
| DDoS-ICMP_Fragmentation | 53,046 | 11,430 | 11,402 |
| DDoS-PSHACK_Flood | 481,254 | 102,985 | 103,326 |
| DDoS-RSTFINFlood | 475,441 | 101,563 | 101,819 |
| DDoS-SlowLoris | 2,757 | 620 | 622 |
| DDoS-SYN_Flood | 478,653 | 102,644 | 102,208 |
| DDoS-SynonymousIP_Flood | 422,083 | 90,795 | 90,480 |
| DDoS-TCP_Flood | 528,499 | 113,888 | 113,735 |
| DDoS-UDP_Flood | 637,558 | 136,466 | 136,717 |
| DDoS-UDP_Fragmentation | 34,169 | 7,221 | 7,224 |
| DictionaryBruteForce | 1,541 | 290 | 319 |
| DNS_Spoofing | 21,214 | 4,565 | 4,570 |
| DoS-HTTP_Flood | 8,487 | 1,833 | 1,805 |
| DoS-SYN_Flood | 237,573 | 51,115 | 51,116 |
| DoS-TCP_Flood | 314,174 | 67,056 | 67,697 |
| DoS-UDP_Flood | 390,422 | 83,446 | 83,627 |
| Mirai-greeth_flood | 116,133 | 24,910 | 24,934 |
| Mirai-greip_flood | 88,821 | 18,867 | 19,279 |
| Mirai-udpplain | 104,814 | 22,314 | 22,536 |
| MITM-ArpSpoofing | 36,316 | 7,741 | 7,840 |
| Recon-HostDiscovery | 15,737 | 3,482 | 3,331 |
| Recon-OSScan | 11,587 | 2,500 | 2,433 |
| Recon-PingSweep | 249 | 45 | 53 |
| Recon-PortScan | 9,648 | 2,083 | 2,082 |
| SqlInjection | 590 | 148 | 148 |
| Uploading_Attack | 140 | 35 | 33 |
| VulnerabilityScan | 4,396 | 905 | 913 |
| XSS | 414 | 91 | 103 |

The largest training class is `DDoS-ICMP_Flood` (848,088); the smallest is `Uploading_Attack` (140). The ratio is about 6058:1. `Uploading_Attack` has 35 validation rows and 33 test rows. All classes are represented, but minority-class metrics will have substantially higher uncertainty.

## Data-quality checks

### Missing, infinite, and parseability checks

The scan used pandas CSV chunks of 100,000 rows; files were processed one at a time. Each chunk was checked for missing values, numeric infinities, and non-numeric tokens in non-label columns.

| Split | Missing values | Infinite numeric values | Non-numeric feature tokens |
| --- | --- | --- | --- |
| Train | 0 | 0 | 0 |
| Validation | 0 | 0 | 0 |
| Test | 0 | 0 | 0 |

### Duplicate records and cross-split overlap

To avoid holding rows in memory, a temporary SQLite database stored SHA-256 fingerprints and per-split counts only. Exact-row fingerprints hash the serialized record bytes excluding line endings. Feature-vector fingerprints hash the first 46 comma-separated fields, excluding `label`. Every source row had the expected field count. Hash matches have negligible cryptographic collision risk, but original rows were not re-read for byte-by-byte verification of matching digests.

Within-split duplicate counts are unique fingerprints that repeat and the excess occurrences beyond the first. No duplicate records were deleted.

| Split | Repeated exact-row fingerprints | Extra exact-row occurrences | Repeated feature-vector fingerprints |
| --- | --- | --- | --- |
| Train | 134,565 | 134,565 | 134,565 |
| Validation | 6,297 | 6,297 | 6,297 |
| Test | 6,384 | 6,384 | 6,384 |

Pairwise overlap counts are **unique fingerprints shared by the named pair**. Do not add pair counts together: a fingerprint present in all three files is counted in each relevant pair.

| Split pair | Shared exact-row fingerprints | Shared feature-vector fingerprints | Matching row instances | Share of right-hand split |
| --- | --- | --- | --- | --- |
| Train ↔ Validation | 57,744 | 57,744 | 57,744 | 4.91% of validation rows |
| Train ↔ Test | 57,847 | 57,847 | 57,847 | 4.92% of test rows |
| Validation ↔ Test | 12,422 | 12,422 | 12,422 | 1.06% of test rows |

The identical pairwise row and feature-vector counts mean the independent fingerprint summaries reported the same number of matches for each pair. This audit did not transform or rewrite source rows. The measured overlap creates a material evaluation-leakage risk for naïve use of the supplied validation/test split assignments.

### Other observations

- All three split files have the same 47-column schema and all 34 labels.
- `sample.py` is a Python sampling script (6,736 bytes); it defines a fixed seed and per-class reservoir sampling. It was read but not executed.
- The feature set contains duration/interval fields but no obvious absolute event timestamp. The files alone do not establish a natural chronological traffic stream.
- Identical feature vectors can repeat because this dataset contains duplicate traffic records; preserve the raw evidence and document any derived split/deduplication policy.

## Hardware feasibility (Intel i5-13420H, 16 GB RAM, CPU-only)

### Memory and temporary storage

- The combined CSV input is 2.16 GiB on disk.
- Materializing all 46 feature columns for every row as float32 would require about **1376.7 MiB**; int32 labels add about **29.9 MiB**. As float64, feature arrays alone are about **2753.5 MiB**.
- The train feature matrix alone as float32 is about **963.7 MiB**; int64 training labels add about **41.9 MiB**.
- Full preprocessing and training are **feasible within 16 GB RAM** if data are processed in chunks and transformed arrays use float32 memory-mapped files or streamed batches. Loading multiple float64 DataFrames and temporary copies at once is unnecessary and could consume several gigabytes.
- A reasonable derived-data disk budget is **5–10 GB** for float32 feature arrays, labels, preprocessing outputs, model checkpoints, logs, and scratch files. A single extra full copy of all CSVs is not needed. The duplicate audit used a 664.4 MiB fingerprint database, retained under `work/`. It contains hashes and counts, not raw dataset rows.
- At inspection time, C: had 249.94 GiB free of 474.72 GiB total. This is ample for the planned derived arrays, although available space can change.

### Compute

The row scan of all three files completed in 33.3 seconds; the full duplicate-fingerprint pass took 174.4 seconds. These are observed inspection timings, **not training-time estimates**.

A small MLP is computationally manageable, but training over 5.49 million rows is CPU-intensive. Centralized training across multiple epochs and repeated validation can take significant time. FedAvg multiplies this work by communication rounds and local epochs; validation evaluation each round is another cost. Integrated Intel graphics should not be assumed to accelerate PyTorch. Start with a measured, short profiling run only after approval, then choose batch size, epoch count, and round count from observed throughput. Use early stopping and avoid per-client copies of the full feature matrix.

## Verified facts versus unverified assumptions

### Verified locally

- The exact paths, file byte sizes, row counts, column names, dtypes, label counts, and missing/infinite/nonnumeric counts above.
- The three large split files are much larger than the `sampled_data` CSVs.
- `sample.py` describes sampling those large split files into the sample target sizes.
- The large files contain duplicate row/feature fingerprints within and across split folders.

### Not verified

- Whether these three large split files include every row in the original upstream CICIoT2023 release. Local filenames and folder structure are not a release manifest.
- Whether the source split assignments were made by time, device, source capture, or random row selection.
- Whether the overlap fingerprints represent identical original flow records or repeated feature values from separate but indistinguishable flows. The CSV itself does not supply stable record IDs.

## Limitations and risks

1. The cross-split overlap is non-trivial. Evaluation on the supplied test CSV without a group-aware check can overstate generalization.
2. Class imbalance is severe, despite all 34 classes appearing in each split. Macro metrics and per-class estimates for labels with tens of test rows will be noisy.
3. There is no demonstrated temporal ordering, so naturally occurring drift cannot be claimed from these files alone.
4. A three-client experiment on one laptop is a simulation, not deployment across organizations. No differential privacy or secure aggregation is present unless separately implemented.
5. Full-dataset CPU training is feasible from a memory/disk standpoint but may be lengthy; exact wall-clock time depends on batch size, epochs, implementation, and other laptop load.

## Recommended next steps

1. Before training, approve a derived-data split policy that groups identical feature-vector fingerprints into a single split. Keep all original CSVs untouched. Check for label conflicts inside each feature-vector group, then create deterministic class-aware train/validation/test manifests with no group crossing. Do not duplicate rare examples; report absent or low-support classes.
2. Fit imputers/scalers and label maps on derived training data only. Transform validation/test with the frozen train-fitted objects. Store float32 features in chunked files or memory maps.
3. After split integrity passes, build and select a centralized MLP using train and validation only; access test once after the configuration is fixed.
4. Partition only derived training rows into three reproducible client sets. Use an IID baseline and a label-skewed Dirichlet partition as a non-IID comparison; each row belongs to exactly one client and class totals are preserved without oversampling. Record per-client counts and class coverage.
5. Run FedAvg with identical model architecture/preprocessing and aggregate client parameters weighted by local sample count. Keep validation/test out of local updates.
6. Calibrate PSI reference bins/thresholds from training-derived client data, test no-drift and clearly labelled synthetic-shift streams using training/validation only, and keep drift detection separate from attack classification.
7. For selective retraining, update only the flagged simulated client using permitted local training/replay data. Use validation to accept/reject candidates; reserve the leakage-safe test set for final evaluation only. Build a read-only dashboard after saved results exist.

## Proposed staged experiment plan (not executed)

| Stage | Planned work | Leakage and rare-class safeguards |
|---|---|---|
| 1. Data preparation | Create group-aware derived split manifests, train-only preprocessing, float32/memmap outputs, schema and split audit. | Keep raw source files read-only; assign duplicate feature groups to one split; no synthetic duplication; inspect label conflicts and support. |
| 2. Central baseline | Train lightweight MLP on derived train; select architecture/epoch/checkpoint using derived validation. | Class weighting or balanced loss from training counts only; report supported-class metrics and rare-class uncertainty. |
| 3. Three-client FedAvg | Deterministic 3-client IID and non-IID training partitions; local mini-batch updates and sample-weighted parameter average. | Training rows only; each row assigned once; validation for round selection; no raw row aggregation. Simulation only, no formal privacy guarantee. |
| 4. PSI drift detection | Establish client-specific training-derived PSI reference; test no-drift plus documented synthetic shifts. | Thresholds chosen without test; if no chronological source exists, call all shift experiments synthetic. Drift is not an attack verdict. |
| 5. Selective retraining | Trigger only the affected client, use permitted local train/replay stream, aggregate compatible update. | No test labels or rows used for trigger or training; unaffected clients skipped and logged; validation chooses checkpoint. |
| 6. Final evaluation/dashboard | Freeze protocol, evaluate centralized/FedAvg/selective models once on leakage-safe test, save metrics and visualizations, then render dashboard. | Test never tunes thresholds/models; confusion/per-class metrics note low support; preserve synthetic-drift caveat. |

## Inspection outputs

The chunked split summary and SHA-256 overlap summary are retained as JSON in this new workspace under `work/`; scripts are retained for reproducibility. The SQLite fingerprint database remains under `work/` as generated audit evidence; it stores hashes and counts, not copied CSV rows.
