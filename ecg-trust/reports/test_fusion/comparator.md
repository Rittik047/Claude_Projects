# GE 12SL comparator — test

Model sensitivity at 12SL's own specificity (patient-bootstrap 95% CI of the difference). Indicative: the statement mapping is published in `configs/12sl_label_map.csv`.

| class | positives | 12SL sens | 12SL spec | model sens @ 12SL spec | Δ [95% CI] |
|---|---|---|---|---|---|
| NORM | 963 | 0.552 | 0.915 | 0.821 | +0.269 [+0.213, +0.315] |
| MI | 550 | 0.575 | 0.942 | 0.727 | +0.153 [+0.098, +0.199] |
| STTC | 521 | 0.578 | 0.897 | 0.841 | +0.263 [+0.203, +0.317] |
| CD | 496 | 0.502 | 0.949 | 0.760 | +0.258 [+0.204, +0.313] |
| HYP | 262 | 0.439 | 0.955 | 0.641 | +0.202 [+0.119, +0.278] |

Unmapped 12SL statements on these records (count): NSR (1273), AB (1067), AU (446), LPAREN (400), RPAREN (400), SBRAD (348), ABR (286), SRTH (261), QRSV (251), CORNPROD (185), LAD3 (182), BORDE (171), SOKOLYON (167), RAVL (164), COMMA (159), SAR (141), AFIB (125), PO (118), PVC (115), $SWITH (113), ST& (100), STACH (95), 2ST (88), OCC (85), CRO (77), PAC (77), RVR (41), MSAR (37), LOWV (34), ABER (32), FREQ (30), PSVC (20), QRSW-2ST (19), RAD (19), $SABCOND (16), FLUT (15), SPR (14), LSBINJ (14), $SAND (13), VAVB (11), $SFUS (10), AC (9), MSBRAD (9), QRST (9), JUNCT-R (8), QCERR (8), QRSW (8), STDEP2 (8), SVR (7), UR (7), PULD (7), CJP (6), $SWQR (5), ASBINJ (5), RSR (5), JSTN (5), ACCEL (4), SERYR2 (4), CSEC (4), VESC (4), ISBINJ (4), RAD5 (4), RAD4 (4), EAR (4), BIGEM (3), REPOL (3), MSTDIL (3), SVT (3), MSTDAL (3), $SIVR (2), BLKED (2), ARM (2), MAFB (2), $SOR (2), VPCX (2), EABRAD (2), W4T1 (2), WQTACH (1), EATACH (1), SERYR1 (1), JUNBRAD (1), JST (1), SSBINJ (1), AVDIS (1)
