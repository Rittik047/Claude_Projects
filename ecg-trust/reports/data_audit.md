# Data audit

## Downloads

All files verified against the dataset's SHA256SUMS.txt.

| dataset | version | files selected | verified | failed | MB |
|---|---|---|---|---|---|
| ptbxl | 1.0.3 | 43600 | 43600 | 0 | 542.9 |
| ptbxl_plus | 1.0.1 | 6 | 6 | 0 | 189.2 |
| external | 1.0.0 | 90757 | 90757 | 0 | 5450.2 |

## PTB-XL

| item | value |
|---|---|
| records | 21799 |
| patients | 18869 |
| no superclass label (excluded) | 411 |
| diagnostic SCP codes with likelihood 0 (still counted, benchmark convention) | 213 |
| age coded 300 (de-identified >89) | 293 |
| pacemaker flag | 291 |
| patient leakage across splits | none (checked) |
| noise flag: static_noise | 3260 |
| noise flag: baseline_drift | 1598 |
| noise flag: burst_noise | 613 |
| noise flag: electrodes_problems | 30 |

### Label prevalence per split

| split | records | patients | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|---|
| train | 17084 | 14823 | 7596 (44.5%) | 4379 (25.6%) | 4186 (24.5%) | 3907 (22.9%) | 2119 (12.4%) |
| val_select | 1072 | 955 | 484 (45.1%) | 273 (25.5%) | 275 (25.7%) | 243 (22.7%) | 113 (10.5%) |
| val_calib | 1074 | 962 | 471 (43.9%) | 267 (24.9%) | 253 (23.6%) | 252 (23.5%) | 155 (14.4%) |
| test | 2158 | 1877 | 963 (44.6%) | 550 (25.5%) | 521 (24.1%) | 496 (23.0%) | 262 (12.1%) |

### Human-validated labels per fold

| fold | validated_by_human |
|---|---|
| 1 | 67.3% |
| 2 | 66.3% |
| 3 | 67.0% |
| 4 | 68.1% |
| 5 | 67.7% |
| 6 | 64.2% |
| 7 | 67.9% |
| 8 | 67.8% |
| 9 | 100.0% |
| 10 | 100.0% |

### Recording devices

| device | records |
|---|---|
| CS100    3 | 6140 |
| CS-12 | 4048 |
| AT-6 C 5.5 | 3950 |
| CS-12   E | 2878 |
| AT-6     6 | 2273 |
| AT-60    3 | 966 |
| AT-6 C 5.8 | 824 |
| AT-6 C | 514 |
| AT-6 C 5.0 | 80 |
| AT-6 C 5.3 | 67 |
| AT-6 C 5.6 | 59 |

## External

| item | value |
|---|---|
| records | 45152 |
| signal OK | 43559 |
| signal excluded | 1593 |
| signal OK but no superclass (excluded, as in PTB-XL) | 14913 |
| evaluated (signal OK and labelled) | 28646 |

### Signal exclusions by reason

| reason | records |
|---|---|
| flat_lead | 1494 |
| nan | 97 |
| read_error | 2 |

### Mapped superclass prevalence — primary (strict, PTB-XL-faithful) map

NORM = proxy: only sinus-family rhythm codes (SR, SB, ST, SA).

| class | positives | prevalence | status (min 100 positives) |
|---|---|---|---|
| NORM | 18723 | 65.4% | ok |
| MI | 204 | 0.7% | ok |
| STTC | 5587 | 19.5% | ok |
| CD | 5045 | 17.6% | ok |
| HYP | 757 | 2.6% | ok |

### Sensitivity: liberal label map

Adds isolated ST/T-wave findings → STTC and voltage-only criteria → HYP. Evaluated records: 37331.

| class | positives | prevalence | status (min 100 positives) |
|---|---|---|---|
| NORM | 18723 | 50.2% | ok |
| MI | 204 | 0.5% | ok |
| STTC | 13166 | 35.3% | ok |
| CD | 5045 | 13.5% | ok |
| HYP | 6140 | 16.4% | ok |

### Demographics (evaluated, primary map)

| sex | records |
|---|---|
| male | 15344 |
| female | 13287 |
| unknown | 15 |

| age band | records |
|---|---|
| 40-59 | 9598 |
| 60-79 | 10971 |
| 80-89 | 3257 |
| <40 | 4781 |
| unknown | 39 |

## 12SL statement vocabulary

Counts over all ECGs; feeds Task 19's mapping.

| statement | ECGs |
|---|---|
| NSR | 12631 |
| AB | 11508 |
| NML | 6012 |
| AU | 4537 |
| LPAREN | 4108 |
| RPAREN | 4108 |
| SBRAD | 3378 |
| SRTH | 2570 |
| ABR | 2553 |
| QRSV | 2464 |
| IMI | 2305 |
| CORNPROD | 2080 |
| LAD3 | 1943 |
| BORDE | 1726 |
| COMMA | 1698 |
| SOKOLYON | 1673 |
| RAVL | 1618 |
| AFIB | 1396 |
| FAV | 1240 |
| LVH2 | 1240 |
| SAR | 1195 |
| NST | 1133 |
| PVC | 1132 |
| ST& | 1103 |
| $SWITH | 1084 |
| PO | 1058 |
| NT | 958 |
| STACH | 957 |
| NSTT | 929 |
| OCC | 837 |
| 2ST | 822 |
| LT | 795 |
| CRO | 768 |
| RBBB | 721 |
| PAC | 713 |
| SMI | 708 |
| LNGQT | 636 |
| AMI | 629 |
| ASMI | 605 |
| LBBB | 566 |
| AFB | 507 |
| IRBBB | 456 |
| LAE | 456 |
| ROMESTES | 435 |
| RVR | 413 |
| LVH3 | 366 |
| FREQ | 362 |
| LOWV | 356 |
| IT | 334 |
| ABER | 288 |
| MSAR | 278 |
| QRSW-2ST | 240 |
| RAD | 229 |
| ALT | 226 |
| PSVC | 214 |
| LMI | 192 |
| $SAND | 180 |
| STEMI | 178 |
| ILT | 172 |
| IVCB | 170 |
| ALMI | 163 |
| VPR | 161 |
| AT | 161 |
| FLUT | 151 |
| $SABCOND | 143 |
| BIFB | 142 |
| UR | 142 |
| IVCD | 141 |
| AC | 135 |
| QRSW | 126 |
| JUNCT-R | 120 |
| $SFUS | 115 |
| SPR | 103 |
| RAE | 103 |
| MSBRAD | 98 |
| QCERR | 98 |
| VAVB | 96 |
| ILBBB | 94 |
| BIGEM | 85 |
| SVR | 84 |
| CRVI | 74 |
| VPCX | 72 |
| STDEP2 | 70 |
| LSBINJ | 68 |
| QRST | 68 |
| ACCEL | 67 |
| MALT | 65 |
| PULD | 64 |
| IPMI | 61 |
| CSEC | 57 |
| ISBINJ | 54 |
| RSR | 52 |
| W2T1 | 50 |
| EAR | 47 |
| CJP | 43 |
| ASBINJ | 42 |
| LVH | 38 |
| RVH | 37 |
| RAD4 | 37 |
| WPW | 35 |
| ASVPR | 35 |
| MSTDAL | 33 |
| $SWQR | 33 |
| QESPMI | 31 |
| BAE | 29 |
| PFB | 26 |
| SVT | 25 |
| ARM | 23 |
| MAFB | 23 |
| JSTN | 22 |
| RAD5 | 22 |
| SERYR2 | 21 |
| VESC | 21 |
| AVDPR | 21 |
| IIOHAI | 20 |
| $SOR | 19 |
| BLKED | 18 |
| CHB | 17 |
| MBZI | 17 |
| AIOHAI | 16 |
| MLT | 16 |
| BIVH | 16 |
| REPOL | 16 |
| MSTDIL | 15 |
| SAV | 14 |
| MILT | 13 |
| EABRAD | 13 |
| LINJ | 12 |
| PROAV | 11 |
| SERYR1 | 10 |
| $SRETC | 10 |
| MBZII | 9 |
| POSTMI | 9 |
| WPWA | 9 |
| EATACH | 8 |
| W4T1 | 8 |
| JTACH | 8 |
| WQTACH | 8 |
| RVE+ | 7 |
| SVCX | 7 |
| W3T1 | 6 |
| IRREG | 6 |
| IINJ | 6 |
| INDAX | 6 |
| $SIVR | 5 |
| SCX | 5 |
| MAT | 5 |
| WPWB | 5 |
| APR | 5 |
| JUNBRAD | 5 |
| LIOHAI | 4 |
| JST | 4 |
| JESC | 3 |
| AVDIS | 3 |
| INTRIN | 3 |
| MSTDAS | 2 |
| $SCAPTUR | 2 |
| JR | 2 |
| APCX | 2 |
| PCARD | 2 |
| SSBINJ | 2 |
| ALINJ | 1 |
| AINJ | 1 |
| ILIHAI | 1 |
| MIT | 1 |
| ILINJ | 1 |
| ALIHAI | 1 |

## Baseline (val_select)

Patient-level bootstrap 95 % CI.

| features | macro AUROC [95% CI] | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| 12sl | 0.939 [0.930, 0.948] | 0.947 | 0.943 | 0.933 | 0.946 | 0.928 |
| ecgdeli | 0.928 [0.918, 0.938] | 0.941 | 0.917 | 0.923 | 0.936 | 0.923 |
