# Fresh Stage1 Baseline Audit

Model: bwgnn  
Seeds: [42, 123, 456, 789, 2026]


## YELPCHI

### Artifact Presence

| Seed | Checkpoint | Split | Meta | Metrics |
|------|-----------|-------|------|---------|
| 42 | OK | OK | OK | OK |
| 123 | OK | OK | OK | OK |
| 456 | OK | OK | OK | OK |
| 789 | OK | OK | OK | OK |
| 2026 | OK | OK | OK | OK |

### Split Integrity


**Seed 42**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.1453, 'val': 0.1453, 'test': 0.1453}, max gap: 0.000077
- Stratified: True

**Seed 123**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.1453, 'val': 0.1453, 'test': 0.1453}, max gap: 0.000077
- Stratified: True

**Seed 456**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.1453, 'val': 0.1453, 'test': 0.1453}, max gap: 0.000077
- Stratified: True

**Seed 789**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.1453, 'val': 0.1453, 'test': 0.1453}, max gap: 0.000077
- Stratified: True

**Seed 2026**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.1453, 'val': 0.1453, 'test': 0.1453}, max gap: 0.000077
- Stratified: True

### Baseline Metrics

| Seed | ROC-AUC | AUPRC | F1@0.5 | Macro-F1@0.5 | Cal-Thresh | Cal-F1 | Cal-Macro-F1 |
|------|---------|-------|--------|-------------|------------|--------|-------------|
| 42 | 0.7760 | 0.4139 | 0.3264 | 0.6250 | 0.3300 | 0.4256 | 0.6565 |
| 123 | 0.8115 | 0.4780 | 0.4058 | 0.6656 | 0.3500 | 0.4637 | 0.6814 |
| 456 | 0.8068 | 0.4659 | 0.3777 | 0.6518 | 0.3800 | 0.4503 | 0.6792 |
| 789 | 0.8061 | 0.4600 | 0.3552 | 0.6396 | 0.3200 | 0.4555 | 0.6747 |
| 2026 | 0.8066 | 0.4707 | 0.2868 | 0.6076 | 0.3000 | 0.4576 | 0.6821 |

### Summary (mean ± std)

**Fixed@0.5** (5 seeds):
- ROC-AUC: 0.8014 ± 0.0129  AUPRC: 0.4577 ± 0.0227
- F1: 0.3504 ± 0.0411  Macro-F1: 0.6379 ± 0.0202
**Calibrated** (5 seeds):
- ROC-AUC: 0.8014 ± 0.0129  AUPRC: 0.4577 ± 0.0227
- F1: 0.4505 ± 0.0132  Macro-F1: 0.6748 ± 0.0095

## AMAZON

### Artifact Presence

| Seed | Checkpoint | Split | Meta | Metrics |
|------|-----------|-------|------|---------|
| 42 | OK | OK | OK | OK |
| 123 | OK | OK | OK | OK |
| 456 | OK | OK | OK | OK |
| 789 | OK | OK | OK | OK |
| 2026 | OK | OK | OK | OK |

### Split Integrity


**Seed 42**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.0687, 'val': 0.0687, 'test': 0.0688}, max gap: 0.000181
- Stratified: True

**Seed 123**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.0687, 'val': 0.0687, 'test': 0.0688}, max gap: 0.000181
- Stratified: True

**Seed 456**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.0687, 'val': 0.0687, 'test': 0.0688}, max gap: 0.000181
- Stratified: True

**Seed 789**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.0687, 'val': 0.0687, 'test': 0.0688}, max gap: 0.000181
- Stratified: True

**Seed 2026**
- Ratios: {'train': 0.4, 'val': 0.2, 'test': 0.4} (4:2:4 OK: True)
- Mask overlaps zero: True ({'train_val': 0, 'train_test': 0, 'val_test': 0})
- Pos rates: {'train': 0.0687, 'val': 0.0687, 'test': 0.0688}, max gap: 0.000181
- Stratified: True

### Baseline Metrics

| Seed | ROC-AUC | AUPRC | F1@0.5 | Macro-F1@0.5 | Cal-Thresh | Cal-F1 | Cal-Macro-F1 |
|------|---------|-------|--------|-------------|------------|--------|-------------|
| 42 | 0.9717 | 0.8688 | 0.8411 | 0.9152 | 0.5000 | 0.8411 | 0.9152 |
| 123 | 0.9820 | 0.8600 | 0.8440 | 0.9167 | 0.2500 | 0.8357 | 0.9121 |
| 456 | 0.9753 | 0.8591 | 0.8377 | 0.9134 | 0.3300 | 0.8228 | 0.9051 |
| 789 | 0.9151 | 0.8121 | 0.8361 | 0.9124 | 0.4900 | 0.8380 | 0.9135 |
| 2026 | 0.9485 | 0.8560 | 0.8641 | 0.9273 | 0.4800 | 0.8645 | 0.9276 |

### Summary (mean ± std)

**Fixed@0.5** (5 seeds):
- ROC-AUC: 0.9585 ± 0.0244  AUPRC: 0.8512 ± 0.0200
- F1: 0.8446 ± 0.0101  Macro-F1: 0.9170 ± 0.0054
**Calibrated** (5 seeds):
- ROC-AUC: 0.9585 ± 0.0244  AUPRC: 0.8512 ± 0.0200
- F1: 0.8404 ± 0.0136  Macro-F1: 0.9147 ± 0.0073