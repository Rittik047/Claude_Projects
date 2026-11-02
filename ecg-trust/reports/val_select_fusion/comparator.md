# GE 12SL comparator — val_select

Model sensitivity at 12SL's own specificity (patient-bootstrap 95% CI of the difference). Indicative: the statement mapping is published in `configs/12sl_label_map.csv`.

| class | positives | 12SL sens | 12SL spec | model sens @ 12SL spec | Δ [95% CI] |
|---|---|---|---|---|---|
| NORM | 484 | 0.556 | 0.912 | 0.820 | +0.264 [+0.195, +0.336] |
| MI | 273 | 0.538 | 0.960 | 0.696 | +0.158 [+0.088, +0.242] |
| STTC | 275 | 0.578 | 0.888 | 0.858 | +0.280 [+0.190, +0.344] |
| CD | 243 | 0.564 | 0.948 | 0.782 | +0.218 [+0.134, +0.289] |
| HYP | 113 | 0.416 | 0.944 | 0.708 | +0.292 [+0.168, +0.389] |

Unmapped 12SL statements on these records (count): NSR (626), AB (527), AU (193), LPAREN (186), RPAREN (186), SBRAD (173), ABR (134), QRSV (114), SRTH (110), CORNPROD (99), BORDE (90), LAD3 (89), SOKOLYON (71), AFIB (70), COMMA (68), RAVL (66), STACH (59), $SWITH (59), ST& (58), SAR (55), PO (51), PVC (43), 2ST (36), PAC (36), OCC (34), CRO (27), LOWV (22), RAD (18), RVR (17), PSVC (16), MSAR (13), MSBRAD (11), AC (10), QRSW-2ST (10), ABER (10), $SAND (9), FREQ (9), FLUT (8), VAVB (7), $SFUS (6), SPR (6), $SABCOND (6), UR (6), QRSW (5), CSEC (5), QCERR (4), LSBINJ (4), $SWQR (3), STDEP2 (3), RSR (3), QRST (3), EAR (3), SVR (2), CJP (2), RAD5 (2), JUNCT-R (2), ASBINJ (2), WQTACH (2), MSTDAL (2), RAD4 (2), ISBINJ (2), PULD (2), SERYR1 (1), BIGEM (1), REPOL (1), ACCEL (1), MSTDIL (1), ARM (1), MSTDAS (1), JSTN (1)
