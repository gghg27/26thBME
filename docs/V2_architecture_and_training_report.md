# V2 模型架构与训练策略详细报告

> 基于 `V2/expert_ssas_emotion_model.py` 和 `V2/train_expert_ssas_emotion.py` 的完整分析
>
> 生成日期：2026-06-11

---

## 目录

1. [总体框架概述](#1-总体框架概述)
2. [共享编码器 (Shared Encoder)](#2-共享编码器-shared-encoder)
3. [第一阶段模型：源域选择与适配 (Stage 1: SSAS)](#3-第一阶段模型源域选择与适配-stage-1-ssas)
4. [第二阶段模型：专家情绪适配 (Stage 2: Expert Emotion Adaptation)](#4-第二阶段模型专家情绪适配-stage-2-expert-emotion-adaptation)
5. [核心模块详解](#5-核心模块详解)
6. [损失函数设计](#6-损失函数设计)
7. [两阶段训练策略](#7-两阶段训练策略)
8. [数据流与域划分](#8-数据流与域划分)
9. [评估指标体系](#9-评估指标体系)
10. [超参数配置](#10-超参数配置)
11. [关键创新点总结](#11-关键创新点总结)

---

## 1. 总体框架概述

V2 采用**两阶段源选择与适配（SSAS: Source Selection and Adaptation）** 框架，针对**跨被试情绪识别中的个体差异与抑郁状态混杂**问题设计。核心思想是：
- **阶段一**：学习域不变（domain-invariant）表征，并通过域分类投票筛选与目标域最相关的源域被试；
- **阶段二**：基于抑郁诊断路由，自适应融合 HC（健康对照）和 DEP（抑郁）两类情绪专家，实现对不同亚群情绪的精准识别。

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: SSAS Source Selection               │
│                                                                 │
│   Source (x_s) ──┐                                              │
│                  ├──> Shared Encoder ──> z [B,185]              │
│   Target (x_t) ──┘         │                                    │
│                             │                                    │
│          ┌──────────────────┼──────────────────┐               │
│          ▼                  ▼                  ▼                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│   │ Domain Head  │  │ Emotion Head │  │ Diagnosis    │        │
│   │ (multi-class)│  │ (GRL adv.)   │  │ Head (GRL)   │        │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│          ▼                  ▼                  ▼                │
│     L_domain           L_emo_GRL         L_diag_GRL            │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘               │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │   MMD Head      │                          │
│                    │ L_mmd(z_s, z_t) │                          │
│                    └─────────────────┘                          │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              │  Source Weight Voting       │                    │
│              │  (domain classifier →       │                    │
│              │   per-source weights)       │                    │
│              └──────────────┬──────────────┘                    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Stage 2: Expert Emotion Adaptation               │
│                                                                 │
│   Source (x_s) ──> Shared Encoder ──> z [B,185]                │
│   Target (x_t) ──> (Stage1 init)                                │
│                             │                                    │
│          ┌──────────────────┼──────────────────────────┐       │
│          ▼                  ▼                          ▼        │
│   ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│   │ Diagnosis    │  │  Shared Emotion  │  │ Subject Domain   │ │
│   │ Router       │  │  Head            │  │ Head (GRL adv.)  │ │
│   │ → diag_prob  │  │  → prob_shared   │  │ → L_subject      │ │
│   └──────┬───────┘  └──────┬───────────┘  └──────────────────┘ │
│          │                  │                                    │
│          │         ┌────────┴────────┐                          │
│          │         ▼                 ▼                          │
│          │  ┌──────────────┐  ┌──────────────┐                 │
│          │  │ HC Emotion   │  │ DEP Emotion  │                 │
│          │  │ Expert       │  │ Expert       │                 │
│          │  │ → prob_hc    │  │ → prob_dep   │                 │
│          │  └──────┬───────┘  └──────┬───────┘                 │
│          │         │                 │                          │
│          │         └────────┬────────┘                          │
│          │                  ▼                                    │
│          │         expert_mix_prob =                            │
│          │           p_dep·prob_dep + p_hc·prob_hc              │
│          │                  │                                    │
│          │                  ▼                                    │
│          │         ┌──────────────────┐                         │
│          └────────>│ Final Mix Prob   │                         │
│                    │ α·shared + (1-α)·│                         │
│                    │ expert_mix        │                         │
│                    └────────┬─────────┘                         │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │   MMD Head      │                          │
│                    │ L_mmd(z_s, z_t) │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 共享编码器 (Shared Encoder)

### 2.1 编码器封装

V2 通过 `ThreeBranchEncoderWrapper` 封装 `BrainGraphBackbone`（来自 `models/dep_contrast_bio.py`），提供统一的 `forward(x, de_feat)` 接口：

```python
class ThreeBranchEncoderWrapper(nn.Module):
    """Unifies the existing three-branch backbone behind forward(x, de_feat)."""
    def __init__(self, sfreq=250.0, topk=8, dropout=0.2, ...):
        self.backbone = BrainGraphBackbone(...)
        self.out_dim = self.backbone.out_dim  # 128 + biomarker_dim
```

### 2.2 BrainGraphBackbone 三分支结构

共享编码器继承 V1 的三分支架构，实际包含（可选）三路特征：

```
BrainGraphBackbone
    │
    ├─ 分支 1: 多尺度时间卷积分支 (z_conv)
    │   EEG x → NodeFeatureEncoder → node_features [B,30,64]
    │        → FlattenGraphReadout → z_conv [B,64]
    │
    ├─ 分支 2: PLV 脑网络图分支 (z_plv)
    │   EEG x → RawSignalPLVGraphConstructor → adj [B,30,30]
    │   node_features + adj → WeightedGCNEncoder → plv_nodes [B,30,64]
    │                      → FlattenGraphReadout → z_plv [B,64]
    │
    └─ 分支 3 (可选): 显式 EEG 生物标记物分支 (z_bio)
        EEG x + de_feat + plv_matrix → BiologicalMarkerExtractor → z_bio [B,57]

最终输出 z = [z_conv; z_plv; z_bio] = [B, 185] (含 biomarker)
               或 [z_conv; z_plv] = [B, 128] (不含 biomarker)
```

**关键设计决策：**
- `node_dim=64, attn_dim=16, topk=8`
- 支持被试相对 DE 和 biomarker 归一化
- DE 频带数可配置（默认 `de_num_bands=5`，含 SampEn）

### 2.3 编码器输出维度

| 输出键 | 形状 (含biomarker) | 形状 (不含biomarker) | 含义 |
|--------|-------------------|---------------------|------|
| `z` | `[B, 185]` | `[B, 128]` | 全量拼接特征 |
| `z_conv` | `[B, 64]` | `[B, 64]` | 多尺度卷积分支 |
| `z_plv` | `[B, 64]` | `[B, 64]` | PLV图分支 |
| `z_bio` | `[B, 57]` | `[B, 0]` | 生物标记物分支 |
| `bio_raw` | `[B, 57]` | `None` | 原始生物标记物 |
| `node_features` | `[B, 30, 64]` | `[B, 30, 64]` | 节点特征 |

---

## 3. 第一阶段模型：源域选择与适配 (Stage 1: SSAS)

### 3.1 模型结构

```python
class Stage1SSASSourceSelectionModel(nn.Module):
    def __init__(self, num_domains, ...):
        self.shared_encoder = ThreeBranchEncoderWrapper(...)
        in_dim = self.shared_encoder.out_dim  # 185 or 128

        self.domain_head = MultiDomainHead(in_dim, num_domains)  # 多域分类
        self.emotion_head = MLPHead(in_dim, emotion_classes)     # 情绪分类 (含GRL)
        self.diagnosis_head = MLPHead(in_dim, diagnosis_classes) # 诊断分类 (含GRL)
        self.mmd_head = MMDHead(in_dim, out_dim=64)              # MMD投影
```

### 3.2 前向传播

**输入：** `x [B,30,T]`, `de_feat [B,30,K]`, `lambda_emo`, `lambda_diag`

**处理流程：**

1. 共享编码器提取特征：`z = enc(x, de_feat) [B, 185]`
2. 情绪/诊断特征经 GRL 梯度反转：
   - `z_emo = grad_reverse(z, lambda_emo)` — 对抗训练使编码器遗忘情绪信息
   - `z_diag = grad_reverse(z, lambda_diag)` — 对抗训练使编码器遗忘诊断信息
3. 多任务输出：

| 输出 | 形状 | 用途 |
|------|------|------|
| `z` | `[B, 185]` | 共享特征表示 |
| `z_mmd` | `[B, 64]` | MMD 域对齐投影 |
| `domain_logits` | `[B, N_domains]` | 域分类 logits |
| `emotion_logits_grl` | `[B, 2]` | 经 GRL 的情绪分类 |
| `diagnosis_logits_grl` | `[B, 2]` | 经 GRL 的诊断分类 |

### 3.3 设计动机

阶段一的核心目标是学习**域不变但保留诊断/情绪判别性的特征空间**：
- **域分类器**：多类分类器直接预测样本来自哪个源域/目标域，迫使编码器混淆域差异
- **GRL 对抗**：通过梯度反转层对抗训练情绪和诊断头，但不是完全消除这些信息（λ 很小），而是实现软约束
- **MMD 对齐**：在投影子空间中最小化源域与目标域的分布差异

---

## 4. 第二阶段模型：专家情绪适配 (Stage 2: Expert Emotion Adaptation)

### 4.1 模型结构

```python
class Stage2ExpertEmotionAdaptationModel(nn.Module):
    def __init__(self, num_domains, shared_mix_alpha=0.5, ...):
        self.shared_encoder = ThreeBranchEncoderWrapper(...)  # 从Stage1初始化
        in_dim = self.shared_encoder.out_dim

        # 核心组件
        self.diagnosis_router = MLPHead(in_dim, 2)   # 诊断路由器 (DEP/HC)
        self.shared_emotion_head = MLPHead(in_dim, 2) # 共享情绪头
        self.hc_emotion_expert = MLPHead(in_dim, 2)   # HC情绪专家
        self.dep_emotion_expert = MLPHead(in_dim, 2)  # DEP情绪专家
        self.mmd_head = MMDHead(in_dim, out_dim=64)   # MMD投影
        self.subject_domain_head = MultiDomainHead(in_dim, num_domains) # 域对抗
```

### 4.2 专家混合推理 (核心创新)

阶段二的核心是**基于诊断路由的自适应专家混合（Mixture of Diagnosis-Conditioned Experts）**：

```
输入: z [B, 185]

步骤1 — 诊断路由:
    diag_logits = diagnosis_router(z)         # [B, 2]
    diag_prob = softmax(diag_logits)           # [B, 2]
    p_dep = diag_prob[:, 0]                    # P(DEP | x)
    p_hc  = diag_prob[:, 1]                    # P(HC  | x)

步骤2 — 多专家预测:
    prob_shared = softmax(shared_emotion_head(z))   # [B, 2]  共享专家
    prob_hc     = softmax(hc_emotion_expert(z))     # [B, 2]  HC专家
    prob_dep    = softmax(dep_emotion_expert(z))    # [B, 2]  DEP专家

步骤3 — 诊断加权专家混合:
    expert_mix = p_dep · prob_dep + p_hc · prob_hc  # [B, 2]
    expert_mix = expert_mix / Σ expert_mix          # 归一化

步骤4 — 共享-专家最终融合:
    mix_prob = α · prob_shared + (1-α) · expert_mix  # α = shared_mix_alpha
    mix_prob = mix_prob / Σ mix_prob                 # 归一化
```

**公式表示：**

$$P_{\text{expert}}(y|x) = P(\text{DEP}|x) \cdot P_{\text{DEP}}(y|x) + P(\text{HC}|x) \cdot P_{\text{HC}}(y|x)$$

$$P_{\text{final}}(y|x) = \alpha \cdot P_{\text{shared}}(y|x) + (1-\alpha) \cdot P_{\text{expert}}(y|x)$$

其中 $\alpha \in [0, 1]$ 为共享专家与诊断专家的混合系数（默认 0.5）。

### 4.3 完整输出

| 输出 | 形状 | 含义 |
|------|------|------|
| `z` | `[B, 185]` | 共享特征 |
| `z_mmd` | `[B, 64]` | MMD 投影特征 |
| `diag_logits` | `[B, 2]` | 诊断分类 logits |
| `shared_logits` | `[B, 2]` | 共享情绪 logits |
| `hc_logits` | `[B, 2]` | HC 专家情绪 logits |
| `dep_logits` | `[B, 2]` | DEP 专家情绪 logits |
| `mix_prob` | `[B, 2]` | **最终混合预测概率** |
| `expert_mix_prob` | `[B, 2]` | 纯专家混合概率 |
| `subject_domain_logits` | `[B, N_domains]` | 域对抗分类 logits |
| `diag_prob` | `[B, 2]` | 诊断概率 (路由权重) |
| `logits` | `[B, 2]` | = mix_prob (兼容接口) |

---

## 5. 核心模块详解

### 5.1 MLPHead — 通用分类头

```python
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, dropout=0.2, use_layer_norm=False):
        layers = [Linear(in_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(LayerNorm(hidden_dim))
        layers.extend([GELU(), Dropout(dropout), Linear(hidden_dim, out_dim)])
```

| 参数 | 用于情绪头 | 用于诊断头 |
|------|----------|----------|
| `hidden_dim` | 64 | 64 |
| `dropout` | 0.2 (Stage1) / 0.35 (Stage2) | 同左 |
| `use_layer_norm` | False | False |

### 5.2 MultiDomainHead — 多域分类器（含 GRL）

```python
class MultiDomainHead(nn.Module):
    def __init__(self, in_dim, num_domains, hidden_dim=128, dropout=0.2):
        self.net = Sequential(
            Linear(in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, num_domains),
        )

    def forward(self, z, lambda_grl=0.0):
        if lambda_grl > 0:
            z = grad_reverse(z, lambda_grl)   # 梯度反转
        return self.net(z)
```

- 带 LayerNorm 稳定训练
- 支持可选的 GRL 对抗训练
- 域数 = `len(source_subjects) + len(val_subjects) + len(test_users)`

### 5.3 MMDHead — MMD 投影头

```python
class MMDHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.2):
        self.net = Sequential(
            Linear(in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, out_dim),
        )
```

将高维特征投影到 64 维子空间，在该空间中计算 MMD，提升域对齐效果。

### 5.4 GradReverse — 梯度反转层

```python
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)         # 前向：恒等映射

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None  # 反向：梯度取反
```

GRL 在前向传播时不做任何变换，反向传播时将梯度乘以 $-\lambda_{\text{GRL}}$，实现对抗训练。

---

## 6. 损失函数设计

### 6.1 阶段一损失函数

$$\mathcal{L}_{\text{Stage1}} = \lambda_{\text{domain}} \mathcal{L}_{\text{domain}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}} + \lambda_{\text{emo\_grl}} \mathcal{L}_{\text{emo}}^{\text{GRL}} + \lambda_{\text{diag\_grl}} \mathcal{L}_{\text{diag}}^{\text{GRL}}$$

| 损失项 | 公式 | 权重 (默认值) | 作用 |
|--------|------|-------------|------|
| $\mathcal{L}_{\text{domain}}$ | CrossEntropy(domain_logits, domain_labels) | $\lambda_{\text{domain}}=1.0$ | 源域判别 |
| $\mathcal{L}_{\text{MMD}}$ | Weighted RBF MMD(z_s, z_t) | $\lambda_{\text{mmd}}=0.03$ | 分布对齐 |
| $\mathcal{L}_{\text{emo}}^{\text{GRL}}$ | CrossEntropy(GRL(z), emo_label) | $\lambda_{\text{emo\_grl}}=0.001$ | 情绪对抗 |
| $\mathcal{L}_{\text{diag}}^{\text{GRL}}$ | CrossEntropy(GRL(z), diag_label) | $\lambda_{\text{diag\_grl}}=0.001$ | 诊断对抗 |

**GRL 系数：** `grl_emo=0.01`, `grl_diag=0.01`

### 6.2 阶段二损失函数

$$\mathcal{L}_{\text{Stage2}} = \lambda_{\text{expert}} \mathcal{L}_{\text{expert}} + \lambda_{\text{mix}} \mathcal{L}_{\text{mix}} + \lambda_{\text{diag}} \mathcal{L}_{\text{diag}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}} + \lambda_{\text{subject}} \mathcal{L}_{\text{subject}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$$

| 损失项 | 公式 | 权重 (默认值) | 作用 |
|--------|------|-------------|------|
| $\mathcal{L}_{\text{expert}}$ | hard_expert_emotion_loss | $\lambda_{\text{expert}}=0.5$ | HC/DEP专家独立训练 |
| $\mathcal{L}_{\text{mix}}$ | mixture_emotion_nll_loss | $\lambda_{\text{mix}}=1.0$ | 混合概率NLL |
| $\mathcal{L}_{\text{diag}}$ | Weighted CE(diag_logits, y_diag) | $\lambda_{\text{diag}}=0.02$ | 诊断路由训练 |
| $\mathcal{L}_{\text{MMD}}$ | Weighted RBF MMD | $\lambda_{\text{mmd}}=3\times10^{-4}$ | 分布对齐 |
| $\mathcal{L}_{\text{subject}}$ | CE(domain_logits, domain_labels) | $\lambda_{\text{subject}}=3\times10^{-4}$ | 被试域对抗 |
| $\mathcal{L}_{\text{ent}}$ | target_entropy_loss | $\lambda_{\text{ent}}=0.0$ | 目标域熵正则（可选） |

#### 6.2.1 专家硬损失 (hard_expert_emotion_loss)

对 HC 样本（`y_diag=1`）使用 HC 专家输出，对 DEP 样本（`y_diag=0`）使用 DEP 专家输出：

$$\mathcal{L}_{\text{expert}} = \begin{cases} \text{CE}(\text{hc\_logits}, y_{\text{emo}}) & \text{if } y_{\text{diag}} = 1 \\ \text{CE}(\text{dep\_logits}, y_{\text{emo}}) & \text{if } y_{\text{diag}} = 0 \end{cases}$$

#### 6.2.2 混合负对数似然损失 (mixture_emotion_nll_loss)

$$\mathcal{L}_{\text{mix}} = -\frac{1}{B}\sum_{i=1}^{B} w_i \cdot \log P_{\text{final}}(y_i^{\text{emo}}|x_i)$$

其中 $w_i$ 为基于阶段一源域投票的样本权重。

#### 6.2.3 加权 MMD 损失 (weighted_mmd_rbf)

$$\mathcal{L}_{\text{MMD}} = \mathbb{E}_{x,x'\sim P_s}[k(z_s, z_s')] + \mathbb{E}_{y,y'\sim P_t}[k(z_t, z_t')] - 2\mathbb{E}_{x\sim P_s, y\sim P_t}[k(z_s, z_t)]$$

使用多核 RBF（5个核，`kernel_mul=2.0`），支持源域样本加权：

$$k(x, y) = \sum_{i=0}^{4} \exp\left(-\frac{\|x-y\|^2}{\sigma \cdot 2^i}\right)$$

其中 $\sigma = \frac{\sum_{i\neq j} \|h_i - h_j\|^2}{n(n-1)}$ 为自适应带宽。

#### 6.2.4 目标域熵损失 (target_entropy_loss)

$$\mathcal{L}_{\text{ent}} = -\frac{1}{B_t}\sum_{i=1}^{B_t} \sum_{c} P_{\text{final}}(c|x_i^t) \log P_{\text{final}}(c|x_i^t)$$

鼓励目标域预测更加确定（低熵）。

---

## 7. 两阶段训练策略

### 7.1 训练流程总览

```
┌──────────────────────────────────────────────────────────────┐
│                    V2 训练流程                                │
│                                                              │
│  1. 数据准备                                                  │
│     ├─ 5折/10折交叉验证划分 (get_unified_subject_split)       │
│     ├─ 构造域ID映射 (source/val/test → domain_id)            │
│     └─ 创建 source/target dataloader (target = val + test)   │
│                                                              │
│  2. 被试基线统计 (Subject-Relative Baselines)                 │
│     ├─ DE baseline: 每个被试的 DE 均值/标准差                │
│     └─ Bio baseline: 每个被试的 biomarker 均值/标准差        │
│                                                              │
│  3. 阶段一训练 (Stage 1)                                      │
│     ├─ 域分类 + MMD + GRL情绪/诊断对抗                        │
│     ├─ 优化器: AdamW, lr=1e-4, CosineAnnealing               │
│     ├─ 类别权重平衡 (emotion + diagnosis)                    │
│     └─ 保存 best domain classifier checkpoint                │
│                                                              │
│  4. 源域权重投票 (Source Weight Voting)                       │
│     ├─ 目标域样本通过 domain classifier → 软投票              │
│     ├─ 统计每个源域被试被选中的频率                          │
│     └─ 归一化为样本权重 (mean-one / sum-one)                  │
│                                                              │
│  5. 阶段二训练 (Stage 2)                                      │
│     ├─ 从阶段一最佳模型初始化编码器                           │
│     ├─ HC/DEP 专家 + 诊断路由 + 共享头联合训练               │
│     ├─ 加权损失 (源域投票权重)                                │
│     ├─ 多标准 best checkpoint 追踪                           │
│     ├─ 优化器: AdamW, lr=1e-4, CosineAnnealing               │
│     └─ 保存 8 个不同标准的 best 模型                          │
│                                                              │
│  6. 测试预测                                                  │
│     ├─ 单模型: best checkpoint → test prediction             │
│     └─ 集成: 多折/多次重复 best checkpoints → ensemble       │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 训练超参数一览

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `stage1_epochs` | 5 | 阶段一训练轮数 |
| `stage2_epochs` | 25 | 阶段二训练轮数 |
| `batch_size` | 200 | 批量大小 |
| `lr_stage1` / `lr_stage2` | 1e-4 | 学习率 |
| `weight_decay` | 5e-4 | 权重衰减 |
| `dropout` | 0.35 | 通用 Dropout 率 |
| `shared_mix_alpha` | 0.5 | 共享/专家混合系数 |
| `save_warmup_epochs` | 1 | 前N轮不触发best更新 |

### 7.3 类别权重平衡

训练过程中对情绪和诊断损失使用类别权重：

$$w_c = \frac{N_{\text{total}}}{N_c} / \text{mean}\left(\frac{N_{\text{total}}}{N_c}\right)$$

确保少数类样本获得更高的损失权重。

### 7.4 学习率调度

两阶段均使用 **CosineAnnealingLR**：
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

其中 $T_{\max}$ = `stage1_epochs` 或 `stage2_epochs`。

---

## 8. 数据流与域划分

### 8.1 域ID映射 (Domain ID Mapping)

```
所有被试 → 全局域ID
    ├─ source:subject_1  → domain 0
    ├─ source:subject_2  → domain 1
    ├─ ...
    ├─ val:subject_k     → domain k
    ├─ ...
    └─ test:user_m       → domain N-1
```

- **源域 (source):** 训练集中的被试
- **验证域 (val):** 验证集中的被试
- **目标域 (test):** 测试集中的用户

### 8.2 目标域数据加载器

阶段一和阶段二的 `target_loader` 由 **val + test** 拼接而成：
- 训练时：`shuffle=True, drop_last=True`（与 source batch 等长）
- 投票时：`shuffle=False, drop_last=False`（全覆盖）

### 8.3 被试相对特征 (Subject-Relative Features)

对被试 $s$ 的 DE 特征和 biomarker 特征进行标准化：

$$\tilde{x}_{\text{DE}} = \frac{x_{\text{DE}} - \mu_s^{\text{DE}}}{\sigma_s^{\text{DE}} + \epsilon}$$

$$\tilde{x}_{\text{bio}} = \frac{x_{\text{bio}} - \mu_s^{\text{bio}}}{\sigma_s^{\text{bio}} + \epsilon}$$

其中 $\mu_s, \sigma_s$ 为被试 $s$ 在所有样本上的均值和标准差。

---

## 9. 评估指标体系

### 9.1 多层级评估

| 层级 | 指标 | 计算方式 |
|------|------|---------|
| **Segment 级** | segment_acc, segment_macro_f1 | 窗口级预测 vs 标签 |
| **Trial 级** | trial_acc, trial_macro_f1 | 窗口汇聚到 trial（硬投票） |
| **Top-K Trial 级** | topk_trial_acc, topk_trial_macro_f1 | V1风格 soft top-K（每被试选K=4个最高分trial） |
| **诊断级** | diag_acc, diag_macro_f1 | 诊断路由准确率 |
| **Per-Class** | precision, recall, f1, support | 每类详细指标 |

### 9.2 多标准 Best Checkpoint 追踪

阶段二同时追踪 8 个 best checkpoint 标准：

| 标准名 | 主要指标 | 用途 |
|--------|---------|------|
| `combined` | topk_trial_f1 → acc → trial_f1 → ... → loss | **综合最优（默认）** |
| `topk_trial_f1` | topk_trial_macro_f1 → acc → trial_f1 → ... | Top-K 最优 |
| `topk_trial_acc` | topk_trial_acc → f1 → trial_acc → ... | Top-K 准确率最优 |
| `trial_f1` | trial_macro_f1 → acc → emo_f1 → loss | Trial F1 最优 |
| `trial_acc` | trial_acc → f1 → emo_acc → loss | Trial 准确率最优 |
| `segment_emo_f1` | emotion_macro_f1 → acc → trial_f1 → loss | Segment 情绪最优 |
| `diag_f1` | diag_macro_f1 → acc → trial_f1 → loss | 诊断最优 |
| `loss` | loss → trial_f1 → emo_f1 | 损失最小 |

### 9.3 测试预测策略

| 策略 | 说明 |
|------|------|
| `soft_topk` (默认) | 每被试选 score 最高的 K=4 个 trial 为正类 |
| `soft_threshold` | prob ≥ 0.5 为正类 |
| `hard` | trial 内多数窗口投票 |

### 9.4 集成预测

多折/多重复的 best checkpoints 做 **概率平均融合**：

$$P_{\text{ensemble}}(y|x) = \frac{1}{M}\sum_{m=1}^{M} P_m(y|x)$$

---

## 10. 超参数配置

### 10.1 完整命令行参数

| 参数组 | 参数 | 默认值 | 说明 |
|--------|------|--------|------|
| **数据** | `index_csv` | `com_index_sub_2s.csv` | 训练索引 |
| | `test_csv` | `com_test_trial_index_2s.csv` | 测试索引 |
| | `n_splits` | 10 | 交叉验证折数 |
| **训练** | `stage1_epochs` | 5 | 阶段一轮数 |
| | `stage2_epochs` | 25 | 阶段二轮数 |
| | `batch_size` | 200 | 批量大小 |
| | `lr_stage1/2` | 1e-4 | 学习率 |
| | `weight_decay` | 5e-4 | 权重衰减 |
| **模型** | `sfreq` | 250.0 | 采样率 |
| | `topk` | 8 | 图稀疏化K |
| | `dropout` | 0.35 | Dropout率 |
| | `biomarker_dim` | 57 | 生物标记物维度 |
| | `de_num_bands` | 5 | DE频带数 |
| | `shared_mix_alpha` | 0.5 | 共享/专家混合比 |
| **阶段一损失** | `lambda_domain` | 1.0 | 域分类权重 |
| | `stage1_lambda_mmd` | 0.03 | MMD权重 |
| | `lambda_emo_grl` | 0.001 | 情绪对抗权重 |
| | `grl_emo` | 0.01 | 情绪GRL系数 |
| | `lambda_diag_grl` | 0.001 | 诊断对抗权重 |
| | `grl_diag` | 0.01 | 诊断GRL系数 |
| **投票** | `vote_smooth` | 5.0 | Laplace平滑系数 |
| | `vote_level` | window | 投票粒度 |
| **阶段二损失** | `lambda_expert` | 0.5 | 专家损失权重 |
| | `lambda_mix` | 1.0 | 混合NLL权重 |
| | `lambda_diag` | 0.02 | 诊断CE权重 |
| | `stage2_lambda_mmd` | 3e-4 | MMD权重 |
| | `lambda_subject` | 3e-4 | 域对抗权重 |
| | `grl_subject` | 0.001 | 域GRL系数 |
| | `lambda_ent` | 0.0 | 熵正则权重 |
| **预测** | `k_pos` | 4 | Soft Top-K 正类数 |
| | `threshold` | 0.5 | 概率阈值 |
| | `test_vote_method` | soft_topk | 测试投票策略 |

### 10.2 典型运行命令

```bash
# 快速测试
python V2/train_expert_ssas_emotion.py --fold 0 --stage1_epochs 2 --stage2_epochs 2

# 完整5折训练 (默认10折)
python V2/train_expert_ssas_emotion.py --all_folds --stage1_epochs 20 --stage2_epochs 100

# 多重复训练
python V2/train_expert_ssas_emotion.py --all_folds --all_repeats --batch_size 200 --test_vote_method soft_topk --k_pos 4
```

---

## 11. 关键创新点总结

### 11.1 模型架构创新

1. **诊断条件专家混合 (Diagnosis-Conditioned Expert Mixture)**
   - 将抑郁诊断信息显式融入情绪识别：HC 和 DEP 群体使用独立的情绪专家
   - 诊断路由器提供软权重，实现平滑的专家切换
   - 共享专家与诊断专家的混合系数 α 可调节

2. **两阶段源域选择与适配 (SSAS)**
   - 阶段一：通过域分类投票识别与目标域分布最接近的源域被试
   - 阶段二：利用源域权重加权训练，减少分布差异大的源样本影响

3. **多层次域对齐**
   - 域分类器（显式监督）+ MMD（分布匹配）+ GRL 对抗（隐式对齐）
   - 三重机制互补，从不同角度最小化域差异

### 11.2 训练策略创新

1. **阶段一编码器初始化 → 阶段二**
   - 保证阶段二的起点已具有域不变特性
   - 避免专家模型在域偏置的表示上训练

2. **源域样本加权**
   - 阶段一域分类器对目标域投票 → 量化每个源域被试与目标域的相关性
   - Laplace 平滑避免权重过于极端
   - 支持 mean-one（保持原始损失尺度）和 sum-one 两种归一化模式

3. **多标准 Best Model 追踪**
   - 同时追踪 8 个不同评估标准的 best checkpoint
   - 可针对不同应用场景选择最优模型

4. **跨被试集成预测**
   - 多折/多重复模型做概率平均
   - 支持多种投票策略（soft_topk / soft_threshold / hard）

### 11.3 与 V1 的关键差异

| 方面 | V1 (two_branch_subject_relative) | V2 (expert_ssas_emotion) |
|------|----------------------------------|--------------------------|
| 训练阶段 | 单阶段端到端 | 两阶段 (SSAS + Expert) |
| 情绪分类 | 单头 | 共享头 + HC/DEP 双专家 |
| 诊断利用 | 作为独立分类任务 | 作为情绪专家的路由器 |
| 域对齐 | 单头 GRL | GRL + MMD + 域分类器 |
| 源域权重 | 无 | 阶段一投票产生 |
| 编码器设计 | 三头并行 | 包装 V1 编码器，专注上层策略 |
| 评估 | 简单 best model | 8标准并行追踪 |

---

## 附录 A：模块依赖关系

```
V2/expert_ssas_emotion_model.py
    ├── models/dep_contrast_bio.py
    │   ├── BrainGraphBackbone (三分支编码器)
    │   ├── NodeFeatureEncoder (多尺度时间编码)
    │   ├── RawSignalPLVGraphConstructor (PLV构图)
    │   ├── WeightedGCNEncoder (图编码)
    │   ├── FlattenGraphReadout (图读出)
    │   ├── BiologicalMarkerExtractor (生物标记物)
    │   └── LearnableDEDiagonalFusion (可学习DE对角线)
    └── torch (GradReverse, nn.Module, etc.)

V2/train_expert_ssas_emotion.py
    ├── V2/expert_ssas_emotion_model.py (模型定义)
    ├── config (全局配置, V2_seed)
    ├── dataloader (Competition4ClassDataset)
    ├── utils/data (数据路径解析, 窗口展开)
    ├── utils/folds (统一被试划分)
    └── sklearn.metrics (评估指标)
```

## 附录 B：论文引用建议

> 本工作提出了两阶段源选择与适配（SSAS）框架用于跨被试 EEG 情绪识别。第一阶段通过域分类投票和分布匹配学习域不变表征，并自动筛选与目标域最相关的源域被试；第二阶段引入诊断条件专家混合机制，通过诊断路由器自适应融合健康对照（HC）和抑郁（DEP）两类情绪专家，有效缓解了抑郁状态对情绪识别造成的干扰。实验采用 10 折交叉验证，在 segment 级和 trial 级两个粒度上评估模型性能，并通过多模型集成进一步提高预测鲁棒性。

---

> **参考文献建议：**
> [1] Ganin Y, et al. "Domain-Adversarial Training of Neural Networks." JMLR, 2016.
> [2] Long M, et al. "Learning Transferable Features with Deep Adaptation Networks." ICML, 2015.
> [3] Gretton A, et al. "A Kernel Two-Sample Test." JMLR, 2012.
> [4] Ding Y, et al. "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition." IEEE TAC, 2022.
> [5] Jacobs RA, et al. "Adaptive Mixtures of Local Experts." Neural Computation, 1991.
