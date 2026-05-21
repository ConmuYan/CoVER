# RAER-FD Configs

Active configs live under `configs/raer_fd/`.

Covered datasets: `yelpchi`, `amazon`, `yelpnyc`, `yelpzip`, `tfinance`, and
`tsocial`.

```text
raer_fd/
  base_detectors/        # frozen base detector configs
  large_graph/           # neighbor mini-batch configs for YelpNYC/YelpZip/TSocial
  teacher/raer_hc/       # RAER teacher with hand-crafted relation evidence
  teacher/raer_lree/     # RAER teacher with LREE
  strong_base/           # PriorF-GNN saturation check
  student/               # CBR-Flash student configs
  ablations/             # final retained ablations
```

Canonical run names:

| Run | Config family |
|---|---|
| `base` | `base_detectors/` |
| `raer_hc` | `teacher/raer_hc/` |
| `raer_lree` | `teacher/raer_lree/` |
| `cbr_flash` | `student/` |
| `strong_base_priorfgnn_raer_lree_404020` | `strong_base/` |
| `base_neighbor_mb` | `large_graph/base_detectors/` |
| `raer_hc_compact` | `large_graph/teacher/raer_hc/` |
| `cbr_flash_compact` | `large_graph/student/` |
| `raer_lree_scalable` | `large_graph/teacher/raer_lree_scalable/` |
| `cbr_flash_lree_scalable` | `large_graph/student/` |

Historical configs are archived under
`archive/legacy_raer_migration_20260520/configs_legacy/`.

Scaling note: the default batch scripts only run the compact full-graph set
(`yelpchi`, `amazon`). `yelpnyc`, `yelpzip`, `tfinance`, and `tsocial` are
available as explicit configs, but should follow
`docs/plans/RAER_FD_DATASET_SCALING_PLAN.md` before large reruns. The first
full Work 2 path for `yelpnyc`, `yelpzip`, and `tsocial` is under
`configs/raer_fd/large_graph/`: `base_neighbor_mb` ->
`raer_lree_scalable` -> `cbr_flash_lree_scalable`. The `raer_hc_compact`
path is retained as a compact-evidence control.
