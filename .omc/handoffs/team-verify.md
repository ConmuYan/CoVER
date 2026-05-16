# Phase2 训练脚本严格审查报告

> 时间：2026-05-15 21:30
> 触发：用户报告"显卡利用率低，CPU 异常高"
> 范围：scripts/train_phase2_reasoner.py + models/cover_rel_reasoner.py + training/phase2_losses.py

---

## TL;DR

**性能层面**: 训练脚本本身处理大体正确（张量已上 GPU，模型已 .to(device)）。CPU 高 + GPU 低的主因不是 bug，是几个**叠加放大**的设计决策：(a) `torch.use_deterministic_algorithms(True)`（项目硬性要求，不能改），(b) 模型本身极小（~10K-100K 参数），(c) 每 epoch 3 次完整图 forward + 12+ 次 `.item()` sync，(d) wall time 已经很短（12-50 秒/run），sync 开销占比相对放大。

**科学层面（更严重，必须正视）**: Amazon 上**所有 4 个 Phase2 实验的 5-seed mean AUPRC 全部贴近 base BWGNN**（差异在 0.0001-0.0004 范围），实际**没有任何提升**。具体证据见下方诊断。

---

## 实验结果对比（5-seed mean AUPRC）

| Dataset | base BWGNN | Phase2 E0 | Phase2 E1 | Phase2 E2 | Phase2 E3 | (历史 anchor_gate) |
|---|---|---|---|---|---|---|
| YelpChi | 0.4674 | **0.5669** ✅ | **0.5675** ✅ | **0.5676** ✅ | **0.5672** ✅ | 0.4998 |
| Amazon  | 0.8643 | 0.8643 ⚠️ | 0.8644 ⚠️ | 0.8644 ⚠️ | 0.8640 ⚠️ | 0.8661 |

YelpChi: Phase2 大幅超过 anchor_gate baseline（+0.067）— **实际是非常好的结果**。
Amazon: Phase2 ≈ base — reasoner residual 在 amazon 上 **完全没起作用**。

---

## Section A: 性能问题（CPU/GPU 利用率）

### A1. 主因：`torch.use_deterministic_algorithms(True)` (line 70)
- 强制禁用 cuDNN/cuBLAS 多种加速 kernel，是项目 baseline 硬性要求（AGENTS.md 明确「Use deterministic seeds」）
- 影响：cuDNN convolution / atomic 加速被禁，小 MLP 训练变成 sync-bound
- **不能改**（破坏与 BWGNN base 的可复现对齐）

### A2. 模型极小 + 每 epoch 多次 sync
| 操作 | 次数/epoch | 累积开销 |
|---|---|---|
| `train_one_epoch` forward + backward | 1 | 主要 GPU 计算 |
| `evaluate_phase2(val_mask)` forward + `.cpu().numpy()` | 1 | sync ×2 |
| `compute_epoch_diagnostics(full graph)` forward + 9 个 `.item()` | 1 | sync ×9+ |
| `phase2_losses` stats 部分 8 个 `.item()` | 1 | sync ×8 |
| `train_mask.cpu()` (line 361) | 1 | sync ×1 |
| `best_state = {...v.cpu().clone()...}` 仅当 val 改进 | 0-1 | sync ×N |

**单 epoch ~25 sync 点 + 3 次 forward**。模型 forward 本身只需毫秒级，sync 反而成主导。
- yelpchi: 12 秒 / 300 epochs = 40 ms/epoch — 其中可能 25-30ms 是 sync overhead
- 实际 GPU 工作只占 10-15% 时间窗口 → nvidia-smi 采样到「0%」是必然

### A3. wall time 已经很短不是瓶颈
- 40 runs 全部跑完 < 5 分钟（实测）
- "GPU 利用率低" 是表象，实际训练已经够快
- **结论**: 性能没有"bug"，是设计决策叠加自然结果

### A4. 性能层面的可优化点（小幅，非必要）
1. ❗ `compute_epoch_diagnostics` 每 epoch 全图 forward — 可改为复用 train_one_epoch 的 outputs，省 1/3 forward
2. ❗ judge_align 字典在 main 中提前 `.to(device)` — 避免 compute_phase2_loss 内每 batch 重复转移
3. ❗ best_state CPU clone 改为只在结束时 clone 一次（仅记录 best_epoch + state ref）
4. ❗ diagnostics 中 .item() 可批量做 (tensor.cpu().tolist() 一次)
5. ✅ `.to(device)` 多次冗余但无害（device 相同时是 no-op）

预计优化后 wall time 下降 30-40%。但 absolute time 已经很短，性价比一般。

---

## Section B: 科学正确性问题（更关键）

### B1. Amazon E2 训练 dynamics 检查 (seed 42)

```
epoch  1: val_auprc=0.8657, |Δ_rel|=0.12, l_trust=0.0,  l_cls=0.66, gate=[0.26, 0.29, 0.45]
epoch  3: val_auprc=0.8659 (best!), |Δ_rel|=0.41, l_trust=0.07, l_cls=0.60, gate=[0.13, 0.24, 0.63]
epoch  5: val_auprc=0.8654, |Δ_rel|=0.71, l_trust=0.33, l_cls=0.54
epoch 50: val_auprc=0.8651, |Δ_rel|=1.49 ← 饱和到 delta_rel_max=1.5
epoch 53: 早停，l_trust=2.22 (饱和), val_auprc 早已不动
```

**关键观察**:
- `lambda_trust * L_trust = 3e-3 * 2.22 = 0.0067` 远小于 `l_cls = 0.40`
- ⚠️ **L_trust 实质上没有约束力**，让 |Δ_rel| 一直涨到 tanh 饱和
- 但 val_auprc 在整个训练过程中只在 0.8651-0.8659 之间漂移（noise 范围）
- ⚠️ **reasoner residual 几乎不影响 val_auprc** — base prior 已经强到无法被 residual 撼动
- early stop on val_auprc 在很早就触发（best_epoch=3）→ 保存了几乎是 init 状态的 reasoner
- 5-seed best_epoch: [3, 17, 50, 7, 38] — 都很早

### B2. Amazon E0 (no judge) 也是同样问题
- 5-seed mean AUPRC = 0.8643 = base AUPRC
- 整个 reasoner 的"训练"对 amazon 完全无效

### B3. 根本原因分析

**为什么 amazon 上 reasoner 不起作用？**

1. **Base BWGNN 已经太强**：amazon 上 base AUPRC 0.86+，relation residual 无法在不破坏 base 的前提下提升
2. **Softmax gate 起步均匀**（init 1/3, 1/3, 1/3），训练早期 |Δ_rel| 接近 0，base prior 主导
3. **Val AUPRC 对 residual 不敏感**：amazon val 上 base 的排序已经接近最优，residual 只能维持
4. **Early stop 立刻触发**：因 val_auprc 在 noise 范围漂移，patience=50 计数很快到上限
5. **结果**：保存的 best_state 几乎是 init reasoner（相当于 identity），test ≈ base

**这与 anchor_gate baseline 历史结果对照**：
- anchor_gate amazon 5-seed AUPRC = 0.8661 (+0.0018 over base)
- anchor_gate 用 binary sigmoid + base_additive_gate fusion，**结构上偏向 small additive nudge**
- 新 Phase2 用 **softmax gate**（强约束 Σ=1）+ **bounded tanh residual**
  - softmax 让 gate 互斥，破坏了 binary "addit small nudge" 模式
  - 当 base 已强，softmax-fused residual 没有 binary-fused 容易学出微小修正
- ⚠️ **Phase2 在 amazon 上比 anchor_gate 差 0.0017**（0.8644 vs 0.8661）

### B4. 但 YelpChi 上 Phase2 表现极佳
- YelpChi: Phase2 5-seed mean = 0.5669 (E0) → 0.5676 (E2)
- 与 anchor_gate baseline 0.4998 比，**+0.067 大幅提升**
- 与 base BWGNN 0.4674 比，**+0.10 提升**
- 这是 anchor_gate baseline 的 **3 倍提升量**

**为什么 yelpchi 上能起作用？**
- yelpchi base AUPRC 0.46（base 较弱，留有 residual 空间）
- relation evidence 信号清晰（RUR-dominant）
- gate softmax 后能学出明确 dominance

---

## Section C: 模型/损失实现正确性

逐项核查：

| 检查项 | 结论 | 证据 |
|---|---|---|
| Softmax gate 归一化 | ✅ | `F.softmax(logits/tau, dim=-1)`, `Σ=1` 已验证 |
| α_bias_init=-3.0 | ✅ | `sigmoid(-3) ≈ 0.0474` 实测一致 |
| Rejected/missing α=0 严格 | ✅ | `max_abs_alpha_llm_rejected=0.0` 全 40 runs 验证 |
| Δ_rel/Δ_llm tanh bound | ✅ | epoch 50 实测 |Δ_rel|=1.490 ≤ delta_rel_max=1.5 |
| L_cls = pos-weighted BCE | ✅ | 公式核对 |
| L_trust = mean(Δrel² + η·α·Δllm²) | ✅ | epoch 50: 0.07→2.22 随 |Δrel| 增长 |
| L_sparse = mean(ρ·H(g)) | ✅ | epoch 50: l_sparse=0.29 随 dominance 增加 |
| L_align = mean(w·KL(q‖g)) over accepted | ✅ | judge_align_count=20 amazon, KL 收敛 |
| Δ_llm = 0 when use_judge=False | ✅ | E0 stats 显示 mean_alpha_llm=0 |
| build_judge_relation_targets 正确 | ✅ | 实测 strong→0.90/0.05/0.05, w=1.0 |

**核心结论：实现没有 bug**，amazon 表现弱来自**算法设计**（softmax gate 不擅长在已饱和的 prior 上做 small nudge）。

---

## Section D: 必修问题清单（按优先级）

### D1. 🔴 必修：Amazon 训练动力学问题（科学正确性）
**症状**: 5-seed AUPRC ≈ base，相比 anchor_gate baseline 回退 0.0017。

**修复方案** (按推荐度排列):
1. **A**: 增大 lambda_trust 到 amazon 上 (e.g. 0.05-0.1)，让 trust 真正约束 |Δrel| 增长
2. **B**: 改用 base_additive_gate-style fusion（保留 anchor + per-relation small additive，不强制 softmax）— 但破坏新设计
3. **C**: 提高 patience 让 reasoner 有时间 fine-tune（但 val_auprc 不动，patience 也无用）
4. **D**: 改 early stop metric → val_macro_f1 或 val_g_means（对 residual 更敏感）
5. **E**: 让 amazon 用 lower lr (e.g. 1e-4 vs 1e-3) — 减小 update step

**推荐组合**: D + E + A 一起验证。先 D（最便宜），看是否 best_epoch 推后；再加 E；最后 A。

### D2. 🟡 性能优化（非必要但有益）
1. 复用 train forward outputs 做 diagnostics（省 1 次全图 forward）
2. judge_align 在 main 中提前 to device
3. best_state 改为只在最后保存
4. .item() 减少：在 epoch 间累积，每 N epoch sync 一次

### D3. 🔴 必修：科学结论的口径
- ❌ 不能称 Phase2 "提升 amazon"
- ✅ 可以称 Phase2 在 yelpchi 上**大幅超越 anchor_gate baseline**（+0.067 AUPRC）
- ✅ Phase2 在 amazon 上**与 base 持平**（不算回退因 0.0001 在 noise 内，但相比 anchor_gate 落后 0.0017）

---

## Section E: 推荐下一步

**立即行动**：
1. 把上述发现汇报给用户
2. 询问：是否需要立即跑 Amazon 修复版（D1 方案 D+E+A）？YelpChi 结果是否就足够发论文？
3. 等待用户确认是否要继续优化

**Background 汇报**：
- Phase2 在 yelpchi 上已经显著超过历史 baseline，可作为论文主结论
- Amazon 需要二次 tuning 才能匹配 anchor_gate baseline
- 性能问题不是瓶颈，整个 sweep 5 分钟跑完
