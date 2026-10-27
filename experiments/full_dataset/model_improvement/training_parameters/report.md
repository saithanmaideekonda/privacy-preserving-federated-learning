# Stage 2 — Controlled centralized training-parameter study

Four runs used 872.1 seconds in total for training plus validation; peak sampled working set was 1.35 GiB.

All runs use the prepared Phase 1 train/validation arrays, identical preprocessing/feature order/label mapping, the Phase 2 MLP (44 inputs, hidden layers 128/64, 34 outputs), the original inverse-square-root class weights, 8,192-row minibatches, Adam, alpha 1e-4, and seed 20261026. Only the named learning-rate or epoch-budget factor changes. Validation macro-F1 selects the checkpoint. Test arrays and labels were not loaded.

## Reproduction controls

| Reference | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Saved Phase 2 validation | 0.971271 | 0.663704 | 0.690892 | 0.669651 | 0.971076 |
| Weighting-study reproduction | 0.971271 | 0.663704 | 0.690892 | 0.669651 | 0.971076 |
| Stage 2 reproduction run | 0.971271 | 0.663704 | 0.690892 | 0.669651 | 0.971076 |

## Validation comparison

| Rank | Candidate | Learning rate | Epoch budget | Epochs completed | Selected epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Δ Macro-F1 | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | learning_rate_double_epochs_12 | 0.002 | 12 | 12 | 12 | 0.9830 | 0.6853 | 0.7143 | 0.6917 | 0.9828 | +0.0221 | 191.9 |
| 2 | budget_18_lr_baseline | 0.001 | 18 | 18 | 17 | 0.9798 | 0.6806 | 0.7056 | 0.6827 | 0.9797 | +0.0130 | 301.5 |
| 3 | phase2_reproduction_lr_0p001_epochs_12 | 0.001 | 12 | 12 | 12 | 0.9713 | 0.6637 | 0.6909 | 0.6697 | 0.9711 | +0.0000 | 188.5 |
| 4 | learning_rate_half_epochs_12 | 0.0005 | 12 | 12 | 12 | 0.8758 | 0.6404 | 0.6416 | 0.6274 | 0.8723 | -0.0423 | 190.2 |

All primary runs completed their configured epoch caps; the patience-3 rule was monitored but did not terminate a run early. The 18-epoch candidate selected epoch 17 because epoch 18 had lower validation macro-F1.

## Weak-class validation results

The following table includes all 12 classes ranked weakest in Stage A. Support is shown for both training and validation; candidate scores are validation-only.

| Candidate | Class | Train n | Val n | Precision | Recall | F1 | Δ F1 vs Phase 2 |
|---|---|---:|---:|---:|---:|---:|---:|
| learning_rate_double_epochs_12 | Uploading_Attack | 138 | 37 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| budget_18_lr_baseline | Uploading_Attack | 138 | 37 | 0.0095 | 0.0270 | 0.0141 | +0.0141 |
| phase2_reproduction_lr_0p001_epochs_12 | Uploading_Attack | 138 | 37 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| learning_rate_half_epochs_12 | Uploading_Attack | 138 | 37 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| learning_rate_double_epochs_12 | Recon-PingSweep | 247 | 45 | 0.0432 | 0.1778 | 0.0696 | +0.0503 |
| budget_18_lr_baseline | Recon-PingSweep | 247 | 45 | 0.1207 | 0.1556 | 0.1359 | +0.1167 |
| phase2_reproduction_lr_0p001_epochs_12 | Recon-PingSweep | 247 | 45 | 0.0169 | 0.0222 | 0.0192 | +0.0000 |
| learning_rate_half_epochs_12 | Recon-PingSweep | 247 | 45 | 0.0270 | 0.0222 | 0.0244 | +0.0052 |
| learning_rate_double_epochs_12 | XSS | 414 | 92 | 0.0982 | 0.1196 | 0.1078 | +0.0755 |
| budget_18_lr_baseline | XSS | 414 | 92 | 0.0274 | 0.0217 | 0.0242 | -0.0081 |
| phase2_reproduction_lr_0p001_epochs_12 | XSS | 414 | 92 | 0.0258 | 0.0435 | 0.0324 | +0.0000 |
| learning_rate_half_epochs_12 | XSS | 414 | 92 | 0.0192 | 0.0326 | 0.0242 | -0.0082 |
| learning_rate_double_epochs_12 | Backdoor_Malware | 386 | 96 | 0.2250 | 0.1875 | 0.2045 | +0.0922 |
| budget_18_lr_baseline | Backdoor_Malware | 386 | 96 | 0.2778 | 0.1042 | 0.1515 | +0.0392 |
| phase2_reproduction_lr_0p001_epochs_12 | Backdoor_Malware | 386 | 96 | 0.1220 | 0.1042 | 0.1124 | +0.0000 |
| learning_rate_half_epochs_12 | Backdoor_Malware | 386 | 96 | 0.3548 | 0.1146 | 0.1732 | +0.0609 |
| learning_rate_double_epochs_12 | SqlInjection | 590 | 148 | 0.1265 | 0.2770 | 0.1737 | +0.0048 |
| budget_18_lr_baseline | SqlInjection | 590 | 148 | 0.1379 | 0.3243 | 0.1935 | +0.0247 |
| phase2_reproduction_lr_0p001_epochs_12 | SqlInjection | 590 | 148 | 0.1258 | 0.2568 | 0.1689 | +0.0000 |
| learning_rate_half_epochs_12 | SqlInjection | 590 | 148 | 0.1667 | 0.2568 | 0.2021 | +0.0332 |
| learning_rate_double_epochs_12 | BrowserHijacking | 669 | 155 | 0.1320 | 0.1677 | 0.1477 | -0.0307 |
| budget_18_lr_baseline | BrowserHijacking | 669 | 155 | 0.2771 | 0.1484 | 0.1933 | +0.0148 |
| phase2_reproduction_lr_0p001_epochs_12 | BrowserHijacking | 669 | 155 | 0.1706 | 0.1871 | 0.1785 | +0.0000 |
| learning_rate_half_epochs_12 | BrowserHijacking | 669 | 155 | 0.2679 | 0.0968 | 0.1422 | -0.0363 |
| learning_rate_double_epochs_12 | Recon-OSScan | 11,582 | 2,488 | 0.4353 | 0.1813 | 0.2560 | +0.0290 |
| budget_18_lr_baseline | Recon-OSScan | 11,582 | 2,488 | 0.4367 | 0.1427 | 0.2151 | -0.0119 |
| phase2_reproduction_lr_0p001_epochs_12 | Recon-OSScan | 11,582 | 2,488 | 0.4114 | 0.1568 | 0.2270 | +0.0000 |
| learning_rate_half_epochs_12 | Recon-OSScan | 11,582 | 2,488 | 0.3296 | 0.1415 | 0.1980 | -0.0290 |
| learning_rate_double_epochs_12 | DictionaryBruteForce | 1,543 | 290 | 0.3356 | 0.1724 | 0.2278 | -0.0073 |
| budget_18_lr_baseline | DictionaryBruteForce | 1,543 | 290 | 0.2080 | 0.1966 | 0.2021 | -0.0329 |
| phase2_reproduction_lr_0p001_epochs_12 | DictionaryBruteForce | 1,543 | 290 | 0.3090 | 0.1897 | 0.2350 | +0.0000 |
| learning_rate_half_epochs_12 | DictionaryBruteForce | 1,543 | 290 | 0.2948 | 0.1759 | 0.2203 | -0.0147 |
| learning_rate_double_epochs_12 | CommandInjection | 627 | 126 | 0.2513 | 0.3968 | 0.3077 | +0.0513 |
| budget_18_lr_baseline | CommandInjection | 627 | 126 | 0.1567 | 0.3333 | 0.2132 | -0.0432 |
| phase2_reproduction_lr_0p001_epochs_12 | CommandInjection | 627 | 126 | 0.2778 | 0.2381 | 0.2564 | +0.0000 |
| learning_rate_half_epochs_12 | CommandInjection | 627 | 126 | 0.3084 | 0.2619 | 0.2833 | +0.0269 |
| learning_rate_double_epochs_12 | Recon-PortScan | 9,663 | 2,081 | 0.3435 | 0.4589 | 0.3929 | +0.0178 |
| budget_18_lr_baseline | Recon-PortScan | 9,663 | 2,081 | 0.4294 | 0.4118 | 0.4204 | +0.0453 |
| phase2_reproduction_lr_0p001_epochs_12 | Recon-PortScan | 9,663 | 2,081 | 0.3591 | 0.3926 | 0.3751 | +0.0000 |
| learning_rate_half_epochs_12 | Recon-PortScan | 9,663 | 2,081 | 0.3696 | 0.4339 | 0.3992 | +0.0241 |
| learning_rate_double_epochs_12 | DNS_Spoofing | 21,224 | 4,543 | 0.5297 | 0.4046 | 0.4588 | +0.0207 |
| budget_18_lr_baseline | DNS_Spoofing | 21,224 | 4,543 | 0.4318 | 0.5494 | 0.4836 | +0.0455 |
| phase2_reproduction_lr_0p001_epochs_12 | DNS_Spoofing | 21,224 | 4,543 | 0.4804 | 0.4026 | 0.4381 | +0.0000 |
| learning_rate_half_epochs_12 | DNS_Spoofing | 21,224 | 4,543 | 0.4604 | 0.3993 | 0.4277 | -0.0104 |
| learning_rate_double_epochs_12 | Recon-HostDiscovery | 15,743 | 3,459 | 0.6001 | 0.7606 | 0.6709 | +0.0228 |
| budget_18_lr_baseline | Recon-HostDiscovery | 15,743 | 3,459 | 0.6621 | 0.7097 | 0.6851 | +0.0370 |
| phase2_reproduction_lr_0p001_epochs_12 | Recon-HostDiscovery | 15,743 | 3,459 | 0.5597 | 0.7696 | 0.6481 | +0.0000 |
| learning_rate_half_epochs_12 | Recon-HostDiscovery | 15,743 | 3,459 | 0.5809 | 0.7508 | 0.6550 | +0.0069 |
| learning_rate_double_epochs_12 | DDoS-SlowLoris | 2,749 | 626 | 0.5707 | 0.9089 | 0.7012 | +0.0369 |
| budget_18_lr_baseline | DDoS-SlowLoris | 2,749 | 626 | 0.5729 | 0.8978 | 0.6994 | +0.0351 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-SlowLoris | 2,749 | 626 | 0.5261 | 0.9010 | 0.6643 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-SlowLoris | 2,749 | 626 | 0.4663 | 0.8946 | 0.6130 | -0.0513 |
| learning_rate_double_epochs_12 | VulnerabilityScan | 4,404 | 901 | 0.6288 | 0.9156 | 0.7456 | +0.0805 |
| budget_18_lr_baseline | VulnerabilityScan | 4,404 | 901 | 0.4795 | 0.9623 | 0.6401 | -0.0250 |
| phase2_reproduction_lr_0p001_epochs_12 | VulnerabilityScan | 4,404 | 901 | 0.5107 | 0.9534 | 0.6651 | +0.0000 |
| learning_rate_half_epochs_12 | VulnerabilityScan | 4,404 | 901 | 0.4853 | 0.9523 | 0.6429 | -0.0222 |
| learning_rate_double_epochs_12 | DDoS-HTTP_Flood | 3,391 | 693 | 0.6857 | 0.8658 | 0.7653 | +0.0795 |
| budget_18_lr_baseline | DDoS-HTTP_Flood | 3,391 | 693 | 0.6468 | 0.8456 | 0.7330 | +0.0471 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-HTTP_Flood | 3,391 | 693 | 0.5859 | 0.8268 | 0.6858 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-HTTP_Flood | 3,391 | 693 | 0.6298 | 0.7388 | 0.6799 | -0.0059 |
| learning_rate_double_epochs_12 | MITM-ArpSpoofing | 36,357 | 7,742 | 0.7443 | 0.6788 | 0.7100 | +0.0123 |
| budget_18_lr_baseline | MITM-ArpSpoofing | 36,357 | 7,742 | 0.7663 | 0.6421 | 0.6987 | +0.0009 |
| phase2_reproduction_lr_0p001_epochs_12 | MITM-ArpSpoofing | 36,357 | 7,742 | 0.7104 | 0.6856 | 0.6978 | +0.0000 |
| learning_rate_half_epochs_12 | MITM-ArpSpoofing | 36,357 | 7,742 | 0.7611 | 0.6510 | 0.7018 | +0.0040 |
| learning_rate_double_epochs_12 | DoS-HTTP_Flood | 8,491 | 1,838 | 0.8761 | 0.8808 | 0.8785 | +0.0642 |
| budget_18_lr_baseline | DoS-HTTP_Flood | 8,491 | 1,838 | 0.8654 | 0.8221 | 0.8432 | +0.0289 |
| phase2_reproduction_lr_0p001_epochs_12 | DoS-HTTP_Flood | 8,491 | 1,838 | 0.8489 | 0.7824 | 0.8143 | +0.0000 |
| learning_rate_half_epochs_12 | DoS-HTTP_Flood | 8,491 | 1,838 | 0.7704 | 0.8215 | 0.7952 | -0.0191 |
| learning_rate_double_epochs_12 | BenignTraffic | 129,502 | 27,574 | 0.8571 | 0.8989 | 0.8775 | +0.0060 |
| budget_18_lr_baseline | BenignTraffic | 129,502 | 27,574 | 0.8542 | 0.8960 | 0.8746 | +0.0031 |
| phase2_reproduction_lr_0p001_epochs_12 | BenignTraffic | 129,502 | 27,574 | 0.8547 | 0.8890 | 0.8715 | +0.0000 |
| learning_rate_half_epochs_12 | BenignTraffic | 129,502 | 27,574 | 0.8374 | 0.8998 | 0.8675 | -0.0041 |
| learning_rate_double_epochs_12 | DoS-SYN_Flood | 237,621 | 51,085 | 0.9786 | 0.9903 | 0.9845 | +0.0350 |
| budget_18_lr_baseline | DoS-SYN_Flood | 237,621 | 51,085 | 0.9778 | 0.9902 | 0.9839 | +0.0345 |
| phase2_reproduction_lr_0p001_epochs_12 | DoS-SYN_Flood | 237,621 | 51,085 | 0.9371 | 0.9622 | 0.9495 | +0.0000 |
| learning_rate_half_epochs_12 | DoS-SYN_Flood | 237,621 | 51,085 | 0.8232 | 0.5470 | 0.6573 | -0.2922 |
| learning_rate_double_epochs_12 | DDoS-SYN_Flood | 478,747 | 102,515 | 0.9976 | 0.9790 | 0.9882 | +0.0303 |
| budget_18_lr_baseline | DDoS-SYN_Flood | 478,747 | 102,515 | 0.9968 | 0.9553 | 0.9756 | +0.0177 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-SYN_Flood | 478,747 | 102,515 | 0.9817 | 0.9352 | 0.9579 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-SYN_Flood | 478,747 | 102,515 | 0.6789 | 0.9144 | 0.7792 | -0.1787 |
| learning_rate_double_epochs_12 | DoS-UDP_Flood | 390,514 | 83,356 | 0.9951 | 0.9954 | 0.9953 | +0.0319 |
| budget_18_lr_baseline | DoS-UDP_Flood | 390,514 | 83,356 | 0.9898 | 0.9958 | 0.9928 | +0.0294 |
| phase2_reproduction_lr_0p001_epochs_12 | DoS-UDP_Flood | 390,514 | 83,356 | 0.9466 | 0.9808 | 0.9634 | +0.0000 |
| learning_rate_half_epochs_12 | DoS-UDP_Flood | 390,514 | 83,356 | 0.7812 | 0.7296 | 0.7545 | -0.2089 |
| learning_rate_double_epochs_12 | Mirai-greip_flood | 88,773 | 18,913 | 0.9649 | 0.9668 | 0.9658 | +0.0021 |
| budget_18_lr_baseline | Mirai-greip_flood | 88,773 | 18,913 | 0.9618 | 0.9714 | 0.9666 | +0.0029 |
| phase2_reproduction_lr_0p001_epochs_12 | Mirai-greip_flood | 88,773 | 18,913 | 0.9714 | 0.9561 | 0.9637 | +0.0000 |
| learning_rate_half_epochs_12 | Mirai-greip_flood | 88,773 | 18,913 | 0.9678 | 0.9500 | 0.9589 | -0.0048 |
| learning_rate_double_epochs_12 | Mirai-greeth_flood | 116,202 | 24,869 | 0.9759 | 0.9736 | 0.9747 | +0.0014 |
| budget_18_lr_baseline | Mirai-greeth_flood | 116,202 | 24,869 | 0.9820 | 0.9698 | 0.9759 | +0.0026 |
| phase2_reproduction_lr_0p001_epochs_12 | Mirai-greeth_flood | 116,202 | 24,869 | 0.9699 | 0.9768 | 0.9733 | +0.0000 |
| learning_rate_half_epochs_12 | Mirai-greeth_flood | 116,202 | 24,869 | 0.9662 | 0.9733 | 0.9697 | -0.0036 |
| learning_rate_double_epochs_12 | DDoS-UDP_Fragmentation | 34,181 | 7,188 | 0.9709 | 0.9855 | 0.9782 | +0.0011 |
| budget_18_lr_baseline | DDoS-UDP_Fragmentation | 34,181 | 7,188 | 0.9727 | 0.9850 | 0.9788 | +0.0017 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-UDP_Fragmentation | 34,181 | 7,188 | 0.9687 | 0.9855 | 0.9770 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-UDP_Fragmentation | 34,181 | 7,188 | 0.9703 | 0.9857 | 0.9779 | +0.0009 |
| learning_rate_double_epochs_12 | DDoS-UDP_Flood | 637,636 | 136,453 | 0.9977 | 0.9965 | 0.9971 | +0.0199 |
| budget_18_lr_baseline | DDoS-UDP_Flood | 637,636 | 136,453 | 0.9988 | 0.9929 | 0.9959 | +0.0187 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-UDP_Flood | 637,636 | 136,453 | 0.9890 | 0.9656 | 0.9772 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-UDP_Flood | 637,636 | 136,453 | 0.8421 | 0.8746 | 0.8580 | -0.1191 |
| learning_rate_double_epochs_12 | DDoS-SynonymousIP_Flood | 422,137 | 90,726 | 0.9889 | 0.9976 | 0.9932 | +0.0149 |
| budget_18_lr_baseline | DDoS-SynonymousIP_Flood | 422,137 | 90,726 | 0.9634 | 0.9967 | 0.9797 | +0.0014 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-SynonymousIP_Flood | 422,137 | 90,726 | 0.9628 | 0.9944 | 0.9783 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-SynonymousIP_Flood | 422,137 | 90,726 | 0.9385 | 0.7411 | 0.8282 | -0.1501 |
| learning_rate_double_epochs_12 | DDoS-ICMP_Fragmentation | 53,023 | 11,462 | 0.9806 | 0.9803 | 0.9805 | -0.0022 |
| budget_18_lr_baseline | DDoS-ICMP_Fragmentation | 53,023 | 11,462 | 0.9802 | 0.9802 | 0.9802 | -0.0025 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-ICMP_Fragmentation | 53,023 | 11,462 | 0.9859 | 0.9795 | 0.9827 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-ICMP_Fragmentation | 53,023 | 11,462 | 0.9921 | 0.9783 | 0.9852 | +0.0025 |
| learning_rate_double_epochs_12 | DoS-TCP_Flood | 314,444 | 66,924 | 0.9936 | 0.9943 | 0.9940 | +0.0093 |
| budget_18_lr_baseline | DoS-TCP_Flood | 314,444 | 66,924 | 0.9851 | 0.9959 | 0.9904 | +0.0058 |
| phase2_reproduction_lr_0p001_epochs_12 | DoS-TCP_Flood | 314,444 | 66,924 | 0.9767 | 0.9926 | 0.9846 | +0.0000 |
| learning_rate_half_epochs_12 | DoS-TCP_Flood | 314,444 | 66,924 | 0.9289 | 0.5240 | 0.6701 | -0.3145 |
| learning_rate_double_epochs_12 | DDoS-ACK_Fragmentation | 33,622 | 7,177 | 0.9807 | 0.9851 | 0.9829 | -0.0033 |
| budget_18_lr_baseline | DDoS-ACK_Fragmentation | 33,622 | 7,177 | 0.9788 | 0.9848 | 0.9818 | -0.0044 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-ACK_Fragmentation | 33,622 | 7,177 | 0.9887 | 0.9837 | 0.9862 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-ACK_Fragmentation | 33,622 | 7,177 | 0.9830 | 0.9838 | 0.9834 | -0.0027 |
| learning_rate_double_epochs_12 | DDoS-TCP_Flood | 528,545 | 113,898 | 0.9972 | 0.9957 | 0.9965 | +0.0058 |
| budget_18_lr_baseline | DDoS-TCP_Flood | 528,545 | 113,898 | 0.9988 | 0.9898 | 0.9943 | +0.0036 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-TCP_Flood | 528,545 | 113,898 | 0.9966 | 0.9849 | 0.9907 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-TCP_Flood | 528,545 | 113,898 | 0.7779 | 0.9759 | 0.8657 | -0.1250 |
| learning_rate_double_epochs_12 | Mirai-udpplain | 104,800 | 22,365 | 0.9937 | 0.9958 | 0.9948 | -0.0008 |
| budget_18_lr_baseline | Mirai-udpplain | 104,800 | 22,365 | 0.9957 | 0.9950 | 0.9954 | -0.0002 |
| phase2_reproduction_lr_0p001_epochs_12 | Mirai-udpplain | 104,800 | 22,365 | 0.9962 | 0.9950 | 0.9956 | +0.0000 |
| learning_rate_half_epochs_12 | Mirai-udpplain | 104,800 | 22,365 | 0.9959 | 0.9951 | 0.9955 | -0.0001 |
| learning_rate_double_epochs_12 | DDoS-PSHACK_Flood | 481,124 | 103,101 | 0.9998 | 0.9991 | 0.9995 | +0.0001 |
| budget_18_lr_baseline | DDoS-PSHACK_Flood | 481,124 | 103,101 | 0.9997 | 0.9989 | 0.9993 | -0.0000 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-PSHACK_Flood | 481,124 | 103,101 | 0.9997 | 0.9990 | 0.9993 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-PSHACK_Flood | 481,124 | 103,101 | 0.9996 | 0.9984 | 0.9990 | -0.0003 |
| learning_rate_double_epochs_12 | DDoS-ICMP_Flood | 847,902 | 182,265 | 1.0000 | 0.9988 | 0.9994 | +0.0000 |
| budget_18_lr_baseline | DDoS-ICMP_Flood | 847,902 | 182,265 | 0.9999 | 0.9988 | 0.9994 | -0.0000 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-ICMP_Flood | 847,902 | 182,265 | 1.0000 | 0.9987 | 0.9994 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-ICMP_Flood | 847,902 | 182,265 | 1.0000 | 0.9987 | 0.9994 | -0.0000 |
| learning_rate_double_epochs_12 | DDoS-RSTFINFlood | 475,551 | 101,470 | 1.0000 | 0.9991 | 0.9995 | +0.0000 |
| budget_18_lr_baseline | DDoS-RSTFINFlood | 475,551 | 101,470 | 0.9999 | 0.9991 | 0.9995 | +0.0000 |
| phase2_reproduction_lr_0p001_epochs_12 | DDoS-RSTFINFlood | 475,551 | 101,470 | 0.9999 | 0.9991 | 0.9995 | +0.0000 |
| learning_rate_half_epochs_12 | DDoS-RSTFINFlood | 475,551 | 101,470 | 1.0000 | 0.9988 | 0.9994 | -0.0001 |

## Selection and recommendation

learning_rate_double_epochs_12 ranked first at validation macro-F1 0.691748 versus Phase 2 reproduction 0.669651; inspect rare-class and overall trade-offs before deciding whether to freeze it.

Freeze learning_rate_double_epochs_12 only if the per-class trade-offs are acceptable; its checkpoint is recorded in selected_checkpoint.json. Test evaluation was not run. Do not tune FedAvg or use test results to revise this selection.

## Controls and limitations

- The Phase 2 reproduction and completed weighting-study reproduction are both used as validation controls; their selected macro-F1 values match exactly.
- The longer-budget candidate uses the same learning rate and patience; a longer maximum does not guarantee better generalization.
- Rare-class metrics are noisy where validation support is small. Improvements for individual classes must be weighed against macro-F1, common-class performance, and the complete per-class table.
- No candidate received test evaluation. The selected checkpoint metadata is frozen for a separately approved final evaluation.

## Artifacts

Each candidate directory contains its checkpoint, configuration, class support, training history, validation metrics by epoch, per-class validation metrics, validation confusion matrix, run metadata, and training log.

## Second-seed robustness check

Because the fixed-seed 0.002 learning-rate run exceeded the Phase 2 validation macro-F1, that configuration alone was repeated with seed 20261027. This repeat is a robustness check and does not alter the primary four-run ranking or the frozen primary checkpoint.

| Seed | Epochs completed | Selected epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Δ Macro-F1 vs Phase 2 | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261026 | 12 | 12 | 0.9830 | 0.6853 | 0.7143 | 0.6917 | 0.9828 | +0.0221 | 191.9 |
| 20261027 | 12 | 11 | 0.9779 | 0.6840 | 0.7020 | 0.6849 | 0.9781 | +0.0153 | 205.4 |

Across these two seeds, validation macro-F1 mean was 0.688328, range 0.684908–0.691748, and absolute spread 0.006840. Both runs are above the reproduced Phase 2 validation macro-F1, but the observed spread is material and this is only a two-seed check. Total recorded training-plus-validation time including the repeat was 1077.5 seconds.

The rare-class results also vary by seed (notably Uploading_Attack); this does not establish stable rare-attack detection. Full metrics and support for all Stage A weak classes are saved in `seed_repeat_weak_class_validation.csv`.

| Class | Val support | Seed 20261026 recall/F1 | Seed 20261027 recall/F1 |
|---|---:|---:|---:|
| Uploading_Attack | 37 | 0.0000/0.0000 | 0.0541/0.0678 |
| XSS | 92 | 0.1196/0.1078 | 0.0978/0.0896 |
| Backdoor_Malware | 96 | 0.1875/0.2045 | 0.0625/0.0945 |
| SqlInjection | 148 | 0.2770/0.1737 | 0.2635/0.1797 |
| BrowserHijacking | 155 | 0.1677/0.1477 | 0.1419/0.1991 |
| Recon-PingSweep | 45 | 0.1778/0.0696 | 0.1778/0.0833 |

No test data was used in this repeat.
