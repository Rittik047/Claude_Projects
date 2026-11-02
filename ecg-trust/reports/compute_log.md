# Compute log

Container: 4 CPU cores, 15 GB RAM, no GPU. PyTorch 2.15.0 (CPU execution).

| date | measurement | result | decision |
|---|---|---|---|
| 2026-11-01 | ResNet1d width 64 (1.49 M params), batch 128 | 1.21 s/batch → 2.7 min/epoch → ~4 h for 3 seeds × 30 epochs | exceeds the plan's 3 h budget |
| 2026-11-01 | ResNet1d width 32, batch 128 | 0.19 s/batch → 0.4 min/epoch → ~36 min for 3 seeds × 30 epochs | plan rule suggested width 32 |
| 2026-11-01 | author decision | — | **width 64 for all seeds** (author override; ~4 h budget accepted) |
