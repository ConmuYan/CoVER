# Chapter 3: Methodology

本章介绍 **CIRCUIT**（Contracted Intervention via Relation-Cued Information
Transfer），一个面向图欺诈检测的关系证据残差
增强框架。CIRCUIT 的出发点不是用更复杂的模型覆盖已有检测器，而是研究一个更
保守也更可审计的问题：在冻结基础检测器的前提下，关系证据何时有资格、以多大
幅度修正其判断？

给定基础检测器对节点 $v_i$ 的 logit $b_i$，CIRCUIT 只学习一个受约束的残差
增强项：

$$
s_i=b_i+\Delta_i,\qquad |\Delta_i|\le \delta_{\max}.
$$

这个形式把“基础检测能力”和“关系证据干预”分开。基础检测器负责提供主要判断；
关系证据只在有充分支持时进行有限修正。若基础检测器已经充分吸收了关系信息，
CIRCUIT 学到近似恒等的残差是合理的饱和现象，而不是方法失败。

从整体上看，CIRCUIT 是一条围绕冻结检测器构建的证据-干预-迁移回路。它包含三个
连续阶段：

| Stage | Paper-facing name | 核心问题 | 输出 |
|---|---|---|---|
| 1 | **Evidence Circuit** | 什么关系证据有资格进入系统？ | score-blind 关系证据 |
| 2 | **Intervention Circuit** | 何时修正基础检测器，修正多少？ | 有界残差策略 |
| 3 | **Transfer Circuit** | 如何把残差策略提取为轻量增强体？ | 可部署的残差增强器 |

这种命名刻意避免把方法写成三个松散模块。Evidence Circuit 定义证据边界，
Intervention Circuit 学习受约束的修正策略，Transfer Circuit 再把该策略压缩并
部署到同一个 final-logit 接口上。三者共同服务于一个目标：在不重训基础检测器的
前提下，形成可解释、受约束且高效的关系证据增强。

本文验证的核心假设是：在冻结基础检测器条件下，score-blind 关系证据能够以有界
残差形式提供可靠干预；并且这种残差策略可以被提取为轻量增强体，在降低部署成本
的同时保留源策略的主要行为。后续设计均围绕这一假设展开，而不是追求无约束的
端到端模型容量。

---

## 3.1 Problem Setup and Residual Contract

给定多关系图

$$
\mathcal{G}=(\mathcal{V},\{\mathcal{E}_r\}_{r=1}^{R},\mathbf{X}),
$$

其中 $\mathcal{V}$ 是节点集合，$\mathcal{E}_r$ 是第 $r$ 种关系的边集合，
$\mathbf{X}\in\mathbb{R}^{N\times d}$ 是节点特征，标签
$\mathbf{y}\in\{0,1\}^N$ 表示正常节点和欺诈节点。节点划分为
$\mathcal{V}_{\text{train}}$、$\mathcal{V}_{\text{val}}$ 和
$\mathcal{V}_{\text{test}}$；所有监督训练和原型统计只使用训练集标签。

基础检测器记为 $f_{\theta_{\text{base}}}$。训练完成后，它被视为冻结参照系，输出
节点 logit 和 embedding：

$$
b_i=f_{\theta_{\text{base}}}^{\text{logit}}(\mathcal{G})_i,\qquad
\mathbf{z}_i=f_{\theta_{\text{base}}}^{\text{emb}}(\mathcal{G})_i.
$$

CIRCUIT 不更新 $\theta_{\text{base}}$，也不把基础检测器重新训练成另一个端到端
黑盒。它学习的是残差策略

$$
\Delta_i = \pi(\mathbf{z}_i,\mathbf{E}_i),
$$

其中 $\mathbf{E}_i=\{\mathbf{e}_{i,r}\}_{r=1}^{R}$ 是节点 $i$ 的多关系证据。最终
预测为 $s_i=b_i+\Delta_i$。

这个残差接口带来四个设计约束。

| 约束 | 作用 |
|---|---|
| Frozen base | 保持基础检测器参数不变，使增强效果可归因于残差策略 |
| Score-blind evidence | 证据提取不读取 base logit，避免把预测分数包装成“证据” |
| Train-only prototype | 原型统计只来自训练标签，避免验证/测试标签泄漏 |
| Bounded intervention | 残差由架构约束在 $[-\delta_{\max},\delta_{\max}]$ 内 |

这些约束共同定义了 CIRCUIT 的方法边界：它不是替代检测器，而是一个可审计的
intervention layer。

$\delta_{\max}$ 是这一接口的关键安全阀。它定义的是 logit 空间中允许关系证据
改变基础判断的最大幅度，而不是一个事后裁剪阈值。由于不同基础检测器的 logit
标度可能不同，$\delta_{\max}$ 应作为验证集选择的全局干预预算，并在实验中通过
敏感性分析报告其影响。较小的预算强调保守修正，较大的预算允许关系证据承担更多
决策权；CIRCUIT 的设计目标是在这一预算内学习何时值得干预。

本文默认使用图学习中常见的 transductive evaluation：验证和测试节点的特征与
图结构可以参与消息传递或证据构造。验证节点标签仅用于早停、模型选择和阈值校准；
测试节点标签不参与任何训练、原型统计、模型选择或阈值选择过程。若用于
inductive deployment，类原型仍只由已标注训练节点计算，新节点证据由其可用特征
和关系邻域构造。

---

## 3.2 Evidence Circuit: Defining Admissible Relation Evidence

图欺诈检测中的错误往往是关系条件化的。同一节点在一种关系下可能看起来正常，
在另一种关系下却呈现异常。因此，在学习残差策略之前，首先要明确什么样的关系
证据可以被系统使用。Evidence Circuit 解决的正是这个问题。

对每个节点 $i$ 和关系 $r$，Evidence Circuit 构造

$$
\mathbf{e}_{i,r}
=
\operatorname{Evi}_r
\big(\mathbf{X},\mathbf{A}_r,\mathbf{m}^{\text{train}},
\mathbf{y}_{\text{train}}\big)_i.
$$

这里 $\mathbf{A}_r$ 是关系 $r$ 的邻接矩阵，$\mathbf{m}^{\text{train}}$ 是训练
掩码。该映射不读取 $b_i$，不读取基础检测器的 embedding，也不读取任何预测分数；
因此，Evidence Circuit 中的“证据”不会退化为 base prediction 的再包装。

Evidence Circuit 保留三类关系证据。

第一类是 **关系局部结构**。每个关系有自己的轻量编码器，用来捕捉节点在该关系
下的局部上下文。这样可以避免把不同语义的关系过早平均成一个无差别表示。
记该编码器输出为 $\mathbf{h}^{\text{rel}}_{i,r}$；在实验实现中，它可以是
关系特定的浅层图编码器，也可以由缓存的 score-blind relation-basis features
经浅层 MLP 变换得到。

第二类是 **邻域特征偏离**。对节点 $i$，系统比较其特征 $\mathbf{x}_i$ 与关系
邻居均值 $\bar{\mathbf{x}}_{i,r}$。它由零度行置零的行归一化聚合
$\bar{\mathbf{X}}_r=D_r^{-1}A_r\mathbf{X}$ 得到；若节点在关系 $r$ 下没有邻居，
则对应行定义为零向量，并可通过关系度或 missing-neighbor indicator 与真实零
均值情形区分。该信号回答的是：该节点在当前关系下是否像它的邻居。

第三类是 **训练集原型方向**。欺诈节点和正常节点的类原型只由训练集计算；本文中
$+$ 表示欺诈类，$-$ 表示正常类：

$$
\boldsymbol{\mu}^{+}
=
\frac{1}{|\mathcal{V}^{+}_{\text{train}}|}
\sum_{j\in\mathcal{V}^{+}_{\text{train}}}\mathbf{x}_j,
\qquad
\boldsymbol{\mu}^{-}
=
\frac{1}{|\mathcal{V}^{-}_{\text{train}}|}
\sum_{j\in\mathcal{V}^{-}_{\text{train}}}\mathbf{x}_j.
$$

节点到两个原型的欧氏距离及其 margin 被加入证据表示：
$d_i^+=\|\mathbf{x}_i-\boldsymbol{\mu}^{+}\|_2$，
$d_i^-=\|\mathbf{x}_i-\boldsymbol{\mu}^{-}\|_2$。实际输入使用当前图上的
z-score 形式，以减少不同特征尺度对证据幅度的影响。由于索引条件始终包含训练
掩码，验证集和测试集标签不会参与原型计算。

上述信号经过关系特定变换后得到最终关系证据：

$$
\mathbf{e}_{i,r}
=
\operatorname{MLP}_r
\big[
\mathbf{x}_i,\bar{\mathbf{x}}_{i,r},
\mathbf{h}^{\text{rel}}_{i,r},
d_i^+,d_i^-,d_i^+-d_i^-
\big].
$$

这里重要的不是枚举多少个统计量，而是证据接口满足三个条件：relation-aware、
score-blind 和 train-only。对于大规模图，Evidence Circuit 可以读取缓存的
score-blind relation-basis features，再学习逐关系变换，从而避免全图稀疏邻接
在 GPU 上展开。

---

## 3.3 Intervention Circuit: Learning a Bounded Residual Policy

Evidence Circuit 给出可用证据后，Intervention Circuit 学习一个高容量残差策略。
它的任务不是重新预测标签，而是判断：在当前关系证据下，基础检测器的 logit 是否
需要被有限修正。

对每个关系 $r$，Intervention Circuit 首先把证据映射为关系特定表示：

$$
\mathbf{h}_{i,r}=\operatorname{Expert}_r(\mathbf{e}_{i,r}).
$$

独立 relation expert 的作用是保留不同关系的行为语义。欺诈图中的关系并非可互换：
同一用户、同一商品、同一时间窗口等边类型可能对应不同异常机制，过早共享参数会
削弱这种差异。

每个关系产生一个候选残差信号：

$$
a_{i,r}=\operatorname{Head}_r(\mathbf{h}_{i,r}).
$$

随后，schema gate 根据冻结基础 embedding 和关系表示选择不同关系的相对贡献：

$$
\mathbf{g}_i
=
\operatorname{softmax}
\left(
\frac{
W_g[\mathbf{z}_i;\mathbf{h}_{i,1};\cdots;\mathbf{h}_{i,R}]+b_g
}{\tau}
\right),
\qquad
\sum_{r=1}^{R}g_{i,r}=1.
$$

$\mathbf{z}_i$ 在这里只作为 frozen context 使用；它不会进入 Evidence Circuit，
也不会让梯度回到基础检测器。换言之，score-blind 约束限制的是证据提取阶段：
证据本身不能来自基础预测；Intervention Circuit 可以使用冻结 embedding 来判断
不同关系证据在当前节点上的适用性，但该 embedding 不被声称为独立关系证据。

关系信号被聚合为

$$
u_i=\sum_{r=1}^{R}g_{i,r}a_{i,r},
$$

并通过有界激活得到 source residual policy：

$$
\Delta^{\text{src}}_i
=
\delta_{\max}\tanh(u_i),
\qquad
s_i^{\text{src}}=b_i+\Delta^{\text{src}}_i.
$$

由于 $\tanh$ 的值域被限制在 $(-1,1)$，残差幅度由结构保证：

$$
|\Delta^{\text{src}}_i|<\delta_{\max}.
$$

所有残差头零初始化，因此训练初始时 $s_i^{\text{src}}=b_i$。这使策略从“不干预”
状态出发，只有当训练信号表明关系证据确实能修正错误时才偏离基础检测器。

Intervention Circuit 使用单一监督目标：

$$
\mathcal{L}_{\text{int}}
=
\operatorname{BCE}(s_i^{\text{src}},y_i;w_{\text{pos}}),
\qquad i\in\mathcal{V}_{\text{train}}.
$$

gate 熵、关系贡献、残差幅度等量只作为诊断信号记录，不作为额外优化项。这样做的
目的，是让方法的主要约束来自结构本身，而不是依赖难以解释的多目标权重调参。

---

## 3.4 Transfer Circuit: Extracting a Portable Residual Enhancer

Intervention Circuit 学到的是高容量 residual policy。它适合分析关系证据如何
产生修正，但未必是最经济的部署对象。Transfer Circuit 的作用是从该策略中提取
一个轻量残差增强体，使其能够通过同一个 final-logit contract 接入冻结检测器：

$$
s_i^{\text{port}}=b_i+\Delta^{\text{port}}_i.
$$

这里的“迁移”首先是接口级迁移：只要一个基础检测器能够提供 frozen logit 和
embedding，增强体就可以通过 $b_i+\Delta_i$ 的方式接入。除非实验专门验证，不应
把它表述为跨数据集或跨模型的零样本迁移。

便携增强体接收冻结基础 embedding 和关系证据：

$$
\mathbf{h}^{\text{port}}_i
=
\operatorname{MLP}_{\text{port}}
\big[
\mathbf{z}_i;\mathbf{e}_{i,1};\cdots;\mathbf{e}_{i,R}
\big],
$$

并输出有界残差：

$$
\Delta^{\text{port}}_i
=
\delta_{\max}\tanh
\left(
w_{\text{port}}^\top \mathbf{h}^{\text{port}}_i+b_{\text{port}}
\right).
$$

它不复制 Intervention Circuit 的内部 relation gate，也不匹配逐关系贡献。真正
需要提取的是 source policy 的最终残差行为：何时修正、修正多大、何时保持基础
判断。

令

$$
p_i^{\text{src}}=\sigma(s_i^{\text{src}}),
\qquad
p_i^{\text{port}}=\sigma(s_i^{\text{port}}).
$$

以下所有 loss 项除特别说明外均在当前 mini-batch 上取平均。Transfer Circuit 使用
contract-budgeted residual policy extraction
目标。首先，final-policy matching 项对齐 source policy 与便携增强体的最终预测
分布。source policy 的归一化熵定义为

$$
\eta_i
=
\frac{
-p_i^{\text{src}}\log p_i^{\text{src}}
-(1-p_i^{\text{src}})\log(1-p_i^{\text{src}})
}{\log 2},
\qquad \eta_i\in[0,1].
$$

它只刻画 source soft target 的可靠性，不把 CIRCUIT 扩展为完整的不确定性估计
框架。由于 $p_i^{\text{src}}$ 和 $p_i^{\text{port}}$ 是二分类概率，下面的 KL
均表示 Bernoulli 分布之间的 KL。基于该权重，policy transfer 项写为

$$
\mathcal{L}_{\text{policy}}
=
(1-\eta_i)\operatorname{KL}
( \operatorname{Bern}(p_i^{\text{port}})
\Vert \operatorname{Bern}(p_i^{\text{src}}))
+\eta_i\operatorname{KL}
( \operatorname{Bern}(p_i^{\text{src}})
\Vert \operatorname{Bern}(p_i^{\text{port}})).
$$

当 source policy 熵较低时，目标更强调便携增强体不要偏离 source 的明确判断；
当 source policy 熵较高时，KL 方向的权重调整避免便携增强体在不确定 source
target 上形成过尖锐的残差响应。这里的设计目的是稳定提取源残差策略，而不是把
高熵样本解释为额外监督信号。

其次，监督项保持增强体仍服务于欺诈检测任务：

$$
\mathcal{L}_{\text{sup}}
=
\operatorname{BCE}(s_i^{\text{port}},y_i;w_{\text{pos}}),
\qquad i\in\mathcal{V}_{\text{train}}.
$$

最后，残差预算项约束增强体不要在 source policy 也接近基础检测器的位置过度
干预：

$$
\xi_i^{\text{src}}
=
\frac{|s_i^{\text{src}}-b_i|}{\delta_{\max}},
\qquad
\xi_i^{\text{port}}
=
\frac{|s_i^{\text{port}}-b_i|}{\delta_{\max}},
$$

$$
\mathcal{L}_{\text{budget}}
=
\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}
(1-\xi_i^{\text{src}})\xi_i^{\text{port}}.
$$

当 source policy 几乎不修正基础检测器时，$\xi_i^{\text{src}}$ 较小，便携增强体
也被鼓励保持接近基础判断；当 source policy 使用较大残差时，预算约束随之放松。
最终目标为

$$
\mathcal{L}_{\text{transfer}}
=
\mathcal{L}_{\text{policy}}
+\beta\mathcal{L}_{\text{sup}}
+\lambda_{\text{budget}}\mathcal{L}_{\text{budget}}.
$$

因此，Transfer Circuit 不是一个外接后处理步骤，而是 CIRCUIT 的策略提取阶段：
它把高容量 residual policy 转换为轻量、同契约、可部署的增强体。这里的轻量性
来自结构性删减：便携增强体不再保留逐关系 expert ensemble、schema gate 和
relation-specific residual heads，而是用共享浅层 MLP 直接近似最终残差行为。
$\beta$ 控制任务监督对 policy extraction 的锚定强度，
$\lambda_{\text{budget}}$ 控制便携增强体继承 source policy 保守性的程度。这个
阶段的目标是忠实提取 source residual policy，而不是重新优化出一个必然超过
source policy 的检测器；因此，若 source 在某些样本上选择不干预，budget loss
会有意鼓励便携增强体保留这种保守行为。

---

## 3.5 Training Protocol

CIRCUIT 采用三阶段训练协议。该协议的关键不是训练顺序本身，而是每一阶段冻结什么、
学习什么、输出什么，从而保证残差增益可以被归因到关系证据干预。

```text
Algorithm 1: CIRCUIT training and residual-policy extraction

Input:
  Multi-relation graph G, node features X, train/val/test split,
  training labels y_train, residual budget delta_max,
  transfer weights beta and lambda_budget, gate temperature tau.

Stage 0: Frozen base construction
  Train the base detector with the standard supervised objective.
  Cache base logits b_i, frozen embeddings z_i, checkpoint, split, and seed.
  Freeze all base-detector parameters.

Stage 1: Evidence-conditioned intervention
  Build score-blind relation evidence using X, relation adjacency, and train labels only.
  Train the Intervention Circuit on b_i + Delta_i^src with bounded residual output.
  Select checkpoints and thresholds on the validation split.
  Export source logits, residuals, gates, and diagnostic statistics.

Stage 2: Portable residual-policy extraction
  Freeze the source residual policy.
  Train the Transfer Circuit with policy, supervised, and budget losses.
  Output a lightweight enhancer that preserves the same final-logit contract.
```

几个超参数承担不同角色。$\delta_{\max}$ 决定最大 logit 干预预算；$\tau$ 控制
schema gate 的选择尖锐度；$\beta$ 控制便携增强体对真实标签的任务锚定；
$\lambda_{\text{budget}}$ 控制它在 source policy 保守区域内的残差收缩强度。
这些超参数不应被解释为额外理论假设，而应在验证集上选择，并通过消融或敏感性
分析说明其影响。

复杂度上，Evidence Circuit 和 Intervention Circuit 主要随节点数与关系数线性
增长。若以 $d$ 表示输入特征维度、$d_h$ 表示隐藏维度，一个全图证据-干预前向的
主导开销可粗略写为
$O(\sum_{r=1}^{R}|\mathcal{E}_r|d + NRd_h)$，其中第一项来自逐关系邻域聚合，
第二项来自逐关系证据变换和残差策略计算。中小规模图可以直接全图计算关系证据；
大规模图则使用 mini-batch base training、缓存的 score-blind relation-basis
features、可扩展 Evidence Circuit 和对应的 Transfer Circuit。Transfer Circuit
部署时不再计算高容量 relation expert ensemble 和 schema gate，因此只需要轻量
残差头和同一个 $b_i+\Delta_i$ 接口。

---

## 3.6 Summary

CIRCUIT 将图欺诈检测中的关系证据增强问题重写为一条受约束的证据-干预-迁移回路。
Evidence Circuit 定义什么证据可以进入系统；Intervention Circuit 在冻结基础
检测器上学习何时进行有界修正；Transfer Circuit 再把这种残差策略提取为轻量
增强体。

这一叙事避免了“再训练一个检测器”或“附加一个后处理压缩步骤”的误解。CIRCUIT 的核心
不是模块堆叠，而是统一的 residual contract：冻结基础预测、外部化关系证据、用
架构约束干预幅度，并将最终 residual policy 压缩到可部署的增强接口中。
