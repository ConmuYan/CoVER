## Handoff: Stage 0 (Foundation) → Stage A (Phase1 Baseline)

- **Decided**: 
  - 不 clone GCN/GAT 原 repo, 用 PyG GCNConv/GATConv (官方实现) + BWGNN-2022 论文 baseline 协议
  - 全部 deterministic 用 warn_only=True (GAT 必需)
  - LLM Judge 无条件全 5-seed, max_new_tokens=320
  - Per-model 重新生成 relation_features (避免 cross-model 隐式耦合)

- **Rejected**:
  - clone Kipf-2017/PetarV-2018 → PyG 已是官方, 增量价值低
  - 严格 deterministic (False) → GAT 会 raise 阻塞
  - 仅 3-seed Judge pilot → 用户要求与 BWGNN 完全对齐

- **Risks**:
  - GCN/GAT extras={} 缺少 BWGNN high_freq_response → BAND_* token absent, adapter 已守卫不会崩, 但 evidence vocab 会少 3-5 token
  - GAT cuDNN 注意力路径可能 1e-3 级别浮动 (warn_only 容忍)
  - GPU 1 被其他用户占用 13GB, 严禁使用
  - LLM Judge wall-time 估计 3-5h GPU (4 组合 × 5 seeds × 8-15 min/seed)

- **Files**:
  - configs/yelpchi_gcn.yaml (重写)
  - configs/yelpchi_gat.yaml (重写)
  - configs/amazon_gcn.yaml (新建)
  - configs/amazon_gat.yaml (新建)
  - scripts/train_stage1.py (加 deterministic + CUBLAS_WORKSPACE_CONFIG)
  - tests/test_detector_output_dim.py (17 测试全过)
  - external/README_GCN_GAT.md (论文+协议引用)

- **Remaining**:
  - Stage A: 5 seeds × 4 组合 = 20 train_stage1 runs
  - Stage B: 20 build_relation_features runs
  - Stage C: 20 generate_stage2_err --teacher rule runs
  - Stage D: 20 train_phase2_reasoner with E0_relgate runs (需先创建 model-specific phase2 config 副本)
  - Stage E: 20 build_judge_packets + 20 generate_llm_judge + 20 train_phase2 cover_judge fusion (需创建 phase2_{ds}_{model}_E2 config 副本)
  - Stage F: PROGRESS.md / AGENTS.md 更新, paper_cross_model_results.{csv,md} 生成

- **GPU Plan**:
  - cuda:0 → exec-1 (YelpChi GCN, 5 seeds)
  - cuda:2 → exec-2 (Amazon GCN, 5 seeds)
  - cuda:3 → exec-3 (YelpChi GAT 5 seeds → Amazon GAT 5 seeds, 顺序)
  - GPU 1 严禁
  - LLM Judge: GPU 0/2/3 队列化, 每个 executor 在自己 GPU 上跑 Qwen
