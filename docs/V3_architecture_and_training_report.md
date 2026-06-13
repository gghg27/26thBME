# V3 模型架构与训练策略详细报告

> 基于 `V3/expert_ssas_emotion_model.py` 和 `V3/train_expert_ssas_emotion.py` 的完整分析
>
> 生成日期：2026-06-13

---

## 目录

1. [总体框架概述](#1-总体框架概述)
2. [共享编码器 (Shared Encoder)](#2-共享编码器-shared-encoder)
3. [第一阶段模型：SSAS 源域选择](#3-第一阶段模型ssas-源域选择)
4. [第二阶段模型：专家情绪适配](#4-第二阶段模型专家情绪适配)
5. [损失函数设计](#5-损失函数设计)
6. [V3 核心创新：条件域对齐](#6-v3-核心创新条件域对齐)
7. [V3 核心创新：被试内 Trial 排序损失](#7-v3-核心创新被试内-trial-排序损失)
8. [V3 核心创新：动态源权重](#8-v3-核心创新动态源权重)
9. [两阶段训练流程](#9-两阶段训练流程)
10. [V3 相对 V2 的变化总结](#10-v3-相对-v2-的变化总结)
11. [超参数配置](#11-超参数配置)

---

## 1. 总体框架概述

V3 在 V2 的**两阶段源选择与适配（SSAS）**框架基础上，增加了三项核心创新：

- **条件域对齐 (Conditional MMD)**：按类别分别对齐源域和目标域的分布
- **被试内 Trial 排序损失 (Subject Ranking Loss)**：使训练目标与 soft top-K 推理逻辑一致
- **动态源权重 (Dynamic Source Weight)**：训练过程中根据当前模型的 MMD 距离动态调整源受试者权重

整体框架仍为两阶段：

```
阶段一 (Stage 1): SSAS 源域选择
    ├── 共享编码器 (BrainGraphBackbone, 与 V2 相同)
    ├── 域分类器 (多类) → 目标域投票 → 源权重
    ├── MMD 对齐 (边缘分布)
    └── GRL 对抗 (情绪 + 诊断)

阶段二 (Stage 2): 专家情绪适配
    ├── 从 Stage1 最优模型初始化编码器
    ├── 诊断路由器 + HC/DEP 双专家 + 共享头
    ├── 边缘 MMD + 条件 MMD（新增）
    ├── Trial 排序损失（新增）
    ├── 动态源权重更新（新增）
    └── 8 标准 best checkpoint 追踪
```

---

## 2. 共享编码器 (Shared Encoder)

V3 的共享编码器与 V2 完全相同，使用 `BrainGraphBackbone`（PMG 版），由 `ThreeBranchEncoderWrapper` 封装。

### 2.1 三分支结构

```
BrainGraphBackbone
    │
    ├── 分支 1: 多尺度时间卷积分支 (z_conv)
    │   EEG x → NodeFeatureEncoder → node_features [B,30,64]
    │        → FlattenGraphReadout → z_conv [B,64]
    │
    ├── 分支 2: PLV 脑网络图分支 (z_plv)
    │   EEG x → RawSignalPLVGraphConstructor → adj [B,30,30]
    │   node_features + adj → WeightedGCNEncoder → plv_nodes [B,30,64]
    │                      → FlattenGraphReadout → z_plv [B,64]
    │
    ├── PMG 分支 (z_pmg)
    │   node_features + adj → RegionPLVPMGEncoder
    │       ├── Local GCN (通道级)
    │       ├── Region GCN (脑区级, 5 个脑区)
    │       └── Fusion → z_pmg [B,64]
    │
    └── 分支 3 (可选): 生物标记物分支 (z_bio)
        EEG x + de_feat + plv_matrix → BiologicalMarkerExtractor → z_bio [B,57]

最终输出 z = [z_conv; z_pmg; z_bio] = [B, 185] (含 biomarker)
```

### 2.2 编码器输出

| 输出键 | 形状 | 含义 |
|--------|------|------|
| `z` | `[B, 185]` | 全量拼接特征 |
| `z_conv` | `[B, 64]` | 多尺度卷积分支 |
| `z_plv` | `[B, 64]` | PLV 图分支 |
| `z_pmg` | `[B, 64]` | PMG 融合分支 |
| `z_bio` | `[B, 57]` | 生物标记物分支 |
| `node_features` | `[B, 30, 64]` | 节点特征 |

---

## 3. 第一阶段模型：SSAS 源域选择

### 3.1 模型结构

```python
class Stage1SSASSourceSelectionModel(nn.Module):
    def __init__(self, num_domains, ...):
        self.shared_encoder = ThreeBranchEncoderWrapper(...)
        in_dim = self.shared_encoder.out_dim  # 185

        self.domain_head    = MultiDomainHead(in_dim, num_domains)  # 多域分类器
        self.emotion_head   = MLPHead(in_dim, 2)                   # 情绪分类 (GRL)
        self.diagnosis_head = MLPHead(in_dim, 2)                   # 诊断分类 (GRL)
        self.mmd_head       = MMDHead(in_dim, out_dim=64)          # MMD 投影
```

### 3.2 前向传播

```
输入: x [B,30,T], de_feat [B,30,K], lambda_emo, lambda_diag

1. z = encoder(x, de_feat)                            # [B, 185]
2. z_emo  = grad_reverse(z, lambda_emo)               # 情绪对抗
3. z_diag = grad_reverse(z, lambda_diag)              # 诊断对抗

输出:
  z                    [B, 185]  共享特征
  z_mmd                [B, 64]   MMD 投影
  domain_logits        [B, N]    域分类 logits
  emotion_logits_grl   [B, 2]    情绪分类 (经 GRL)
  diagnosis_logits_grl [B, 2]    诊断分类 (经 GRL)
```

### 3.3 阶段一损失函数

$$\mathcal{L}_{\text{Stage1}} = \lambda_{\text{domain}} \mathcal{L}_{\text{domain}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}} + \lambda_{\text{emo\_grl}} \mathcal{L}_{\text{emo}}^{\text{GRL}} + \lambda_{\text{diag\_grl}} \mathcal{L}_{\text{diag}}^{\text{GRL}}$$

| 损失项 | 公式 | 权重 (默认值) | 作用 |
|--------|------|-------------|------|
| $\mathcal{L}_{\text{domain}}$ | CrossEntropy(domain_logits, domain_labels) | 1.0 | 源域多类判别 |
| $\mathcal{L}_{\text{MMD}}$ | RBF MMD(z_s, z_t) | 0.03 | 分布对齐 |
| $\mathcal{L}_{\text{emo}}^{\text{GRL}}$ | CrossEntropy(GRL(z), emo_label) | 0.001 | 情绪对抗 (GRL=0.01) |
| $\mathcal{L}_{\text{diag}}^{\text{GRL}}$ | CrossEntropy(GRL(z), diag_label) | 0.001 | 诊断对抗 (GRL=0.01) |

### 3.4 最优模型选择

Stage1 以**验证集域分类损失 `val loss_domain`** 为唯一标准，选择域分类能力最强的模型用于后续投票。

---

## 4. 第二阶段模型：专家情绪适配

### 4.1 模型结构

```python
class Stage2ExpertEmotionAdaptationModel(nn.Module):
    def __init__(self, num_domains, shared_mix_alpha=0.5, ...):
        self.shared_encoder = ThreeBranchEncoderWrapper(...)  # 从 Stage1 初始化
        in_dim = self.shared_encoder.out_dim

        self.diagnosis_router     = MLPHead(in_dim, 2)   # 诊断路由器
        self.shared_emotion_head  = MLPHead(in_dim, 2)   # 共享情绪头
        self.hc_emotion_expert    = MLPHead(in_dim, 2)   # HC 情绪专家
        self.dep_emotion_expert   = MLPHead(in_dim, 2)   # DEP 情绪专家
        self.mmd_head             = MMDHead(in_dim, 64)  # MMD 投影
        self.subject_domain_head  = MultiDomainHead(...) # 被试域对抗
```

### 4.2 专家混合推理（核心机制）

```
输入: z [B, 185]

步骤1 — 诊断路由:
    diag_prob = softmax(diagnosis_router(z))    # [B, 2]
    p_dep = diag_prob[:, 0]                     # P(DEP | x)
    p_hc  = diag_prob[:, 1]                     # P(HC  | x)

步骤2 — 多专家预测:
    prob_shared = softmax(shared_emotion_head(z))  # [B, 2]
    prob_hc     = softmax(hc_emotion_expert(z))    # [B, 2]
    prob_dep    = softmax(dep_emotion_expert(z))   # [B, 2]

步骤3 — 诊断加权专家混合:
    expert_mix = p_dep · prob_dep + p_hc · prob_hc  # [B, 2]

步骤4 — 共享-专家最终融合:
    mix_prob = α · prob_shared + (1-α) · expert_mix  # α = 0.5
```

数学表达：

$$P_{\text{expert}}(y|x) = P(\text{DEP}|x) \cdot P_{\text{DEP}}(y|x) + P(\text{HC}|x) \cdot P_{\text{HC}}(y|x)$$

$$P_{\text{final}}(y|x) = \alpha \cdot P_{\text{shared}}(y|x) + (1-\alpha) \cdot P_{\text{expert}}(y|x)$$

### 4.3 完整输出

| 输出 | 形状 | 含义 |
|------|------|------|
| `z` | `[B, 185]` | 共享特征 |
| `z_mmd` | `[B, 64]` | MMD 投影特征 |
| `diag_logits` | `[B, 2]` | 诊断分类 logits |
| `shared_logits` | `[B, 2]` | 共享情绪 logits |
| `hc_logits` | `[B, 2]` | HC 专家情绪 logits |
| `dep_logits` | `[B, 2]` | DEP 专家情绪 logits |
| `mix_prob` | `[B, 2]` | 最终混合预测概率 |
| `expert_mix_prob` | `[B, 2]` | 纯专家混合概率 |
| `subject_domain_logits` | `[B, N_domains]` | 域对抗 logits |
| `diag_prob` | `[B, 2]` | 诊断路由权重 |

---

## 5. 损失函数设计

### 5.1 阶段二总损失

$$\begin{aligned}
\mathcal{L}_{\text{Stage2}} &= \lambda_{\text{expert}} \mathcal{L}_{\text{expert}}
+ \lambda_{\text{mix}} \mathcal{L}_{\text{mix}}
+ \lambda_{\text{diag}} \mathcal{L}_{\text{diag}} \\
&+ \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}}
+ \lambda_{\text{cond\_mmd}} \mathcal{L}_{\text{CondMMD}} \quad \textcolor{red}{\leftarrow \text{V3 新增}} \\
&+ \lambda_{\text{subject}} \mathcal{L}_{\text{subject}}
+ \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}
+ \lambda_{\text{rank}} \mathcal{L}_{\text{rank}} \quad\;\; \textcolor{red}{\leftarrow \text{V3 新增}}
\end{aligned}$$

### 5.2 各损失项详解

#### (1) 专家硬损失 `hard_expert_emotion_loss`

对 HC 样本使用 HC 专家输出，对 DEP 样本使用 DEP 专家输出：

```python
selected_logits[diag==HC] = hc_logits[diag==HC]
selected_logits[diag==DEP] = dep_logits[diag==DEP]
loss_expert = cross_entropy(selected_logits, y_emo, weight=sample_weight)
```

#### (2) 混合 NLL 损失 `mixture_emotion_nll_loss`

$$L_{\text{mix}} = -\frac{1}{B}\sum_{i=1}^{B} w_i \cdot \log P_{\text{final}}(y_i^{\text{emo}}|x_i)$$

#### (3) 诊断路由损失

$$L_{\text{diag}} = -\frac{1}{B}\sum_i w_i \cdot \log P(\text{diag}_i | x_i)$$

#### (4) 加权 MMD 损失 `weighted_mmd_rbf`（V3 增强版）

V3 的 MMD 相对于 V2 有两处增强：

**(a) 同时支持源端和目标端权重：**

$$L_{\text{MMD}} = \sum_{i,j} w_i^s w_j^s k(z_i^s, z_j^s) + \sum_{i,j} w_i^t w_j^t k(z_i^t, z_j^t) - 2\sum_{i,j} w_i^s w_j^t k(z_i^s, z_j^t)$$

**(b) 数值稳定性增强：**
- `torch.clamp(loss, min=0.0)` 确保 MMD 非负
- 权重归一化带 `eps` 防止除零
- 权重为零时返回 0 而非崩溃

#### (5) 被试域对抗损失

$$L_{\text{subject}} = \text{CrossEntropy}(\text{GRL}(z), \text{domain\_label})$$

GRL 系数 `lambda_subject = 3e-4`，**编码器对抗、域分类器正常训练**。

#### (6) 目标域熵损失

$$L_{\text{ent}} = -\frac{1}{B_t}\sum_{i=1}^{B_t} \sum_{c} P_{\text{final}}(c|x_i^t) \log P_{\text{final}}(c|x_i^t)$$

鼓励目标域预测更加确定（低熵），默认权重为 0（不启用）。

---

## 6. V3 核心创新：条件域对齐

### 6.1 设计动机

V2 的普通 MMD 只做整体边缘分布对齐 $P_s(z) \approx P_t(z)$。问题在于源域和目标域整体可能重合了，但正类对正类、中性对中性的对齐可能并不好——源域的正类样本可能被拉向了目标域的中性样本。

V3 引入 **类别条件 MMD**：分别对齐每个类别的条件分布 $P_s(z|y=c) \approx P_t(z|y=c)$。

### 6.2 算法实现

```python
def soft_conditional_mmd_rbf(z_source, z_target, y_source, target_prob,
                              num_classes=2, conf_threshold=0.7,
                              detach_target_prob=True):
    # 目标域用软伪标签而非硬标签
    for cls in range(num_classes):
        # 源域：hard label mask
        source_class_weight = (y_source == cls) * source_sample_weight

        # 目标域：软概率 + 置信度过滤
        target_class_weight = target_prob[:, cls] * conf_mask
        # conf_mask = (max_prob >= conf_threshold)

        # 跳过权重不足的类别
        if source_weight_sum < min_weight or target_weight_sum < min_weight:
            continue

        # 对该类计算加权 MMD
        class_losses.append(
            weighted_mmd_rbf(z_source, z_target,
                source_weight=source_class_weight,
                target_weight=target_class_weight)
        )

    return mean(class_losses)
```

### 6.3 关键设计点

#### 软伪标签

目标域使用**软概率** `target_prob[:, cls]` 作为类别权重，而非先 argmax 再 one-hot：

```python
# 例如一个样本 mix_prob = [0.7, 0.3]
# 它以 0.7 的权重参与正类 MMD，0.3 的权重参与中性 MMD
target_class_weight_cls0 = target_prob[:, 0] * conf_mask  # 正类
target_class_weight_cls1 = target_prob[:, 1] * conf_mask  # 中性
```

#### 置信度过滤

```python
conf = target_prob.max(dim=1).values         # 每样本最大概率
conf_mask = (conf >= conf_threshold)          # 默认 ≥ 0.7
```

只让模型足够确信的目标样本参与条件对齐，避免低置信度的噪声样本。

#### 梯度隔离

```python
if detach_target_prob:
    target_prob = target_prob.detach()
```

目标域伪标签**不参与梯度回传**，防止模型为了降低条件 MMD 而故意让目标伪标签退化（cheating）。

#### Warmup 机制

```python
if current_epoch > cond_mmd_warmup_epochs:  # 默认 5
    loss_cond_mmd = soft_conditional_mmd_rbf(...)
else:
    loss_cond_mmd = 0.0
```

等待模型预测质量稳定后再开启。

### 6.4 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_cond_mmd` | 0.0003 | 条件 MMD 损失权重 |
| `--cond_mmd_warmup_epochs` | 5 | Warmup epoch 数 |
| `--cond_mmd_conf_threshold` | 0.7 | 目标域置信度阈值 |
| `--cond_mmd_detach_target_prob` | True | 是否 detach 目标伪标签 |

---

## 7. V3 核心创新：被试内 Trial 排序损失

### 7.1 设计动机

最终推理使用 **soft top-K**：每个被试内按 `score_pos` 排序，选 top K 个 trial 为正类。但训练时用的是逐窗口独立的 NLL loss，与推理时的排序逻辑脱节。

`stage2_subject_ranking_loss` 弥补了这种脱节：**直接在被试内部，要求正类 trial 的 score 高于中性 trial 的 score**。

### 7.2 算法流程

```
对每个被试独立执行:

Step 1: 窗口级 score
    score_pos = log(P_pos / P_neutral)    # 与推理使用完全相同的公式

Step 2: 窗口聚合到 trial
    trial_score = mean(同 trial 内窗口的 score_pos)
    trial_label = 该 trial 的真实标签

Step 3: 构造正-负 pair
    pos_scores = trial_scores[label == 1]
    neg_scores = trial_scores[label == 0]

    对每一对 (正trial i, 中性trial j):
        loss_ij = softplus(margin - (score_i - score_j))

Step 4: Top-K hard pair 截断 (默认保留 loss 最大的 128 对)

Step 5: 先被试内平均，再跨被试平均
```

### 7.3 核心公式

$$\mathcal{L}_{\text{rank}} = \frac{1}{S}\sum_{s=1}^{S} \frac{1}{|P_s|}\sum_{(i^+, i^-) \in P_s} \text{softplus}\left(m - (s_{i^+} - s_{i^-})\right)$$

其中 $m=0.2$ 为 margin，`softplus(x) = ln(1 + e^x)`。

**直观理解：**

| $s_{\text{pos}} - s_{\text{neg}}$ | Loss 状态 |
|---|---|
| > 0.2 | 已满足 margin，loss ≈ 0 |
| ≈ 0 | 有损失，差距不够 |
| < 0（正 < 中，错序） | loss 很大，严重惩罚 |

### 7.4 与推理的对应

```
训练 (ranking loss):                  推理 (soft topk):

  pos score > neg score + margin         按 score 排序 → top K = positive
  使用相同 score 定义: log(P_pos/P_neutral)
  使用相同聚合方式: window mean → trial
```

### 7.5 与 V2 的区别

`stage2_subject_ranking_loss` 函数在 V2 代码中**已存在**，V3 并非新增此函数本身。区别在于 V3 将其**正式启用**并加入了训练总损失（通过 `lambda_rank` 权重和 `rank_warmup_epochs` 机制），而 V2 中该函数虽已定义但未参与实际训练。

### 7.6 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_rank` | 0.03 | 排序损失权重 |
| `--rank_warmup_epochs` | 3 | Warmup epoch 数 |
| `--rank_margin` | 0.2 | 正-负 trial score 最小差距 |
| `--rank_max_pairs_per_subject` | 128 | 每被试最大 pair 数 |

---

## 8. V3 核心创新：动态源权重

### 8.1 设计动机

V2 的源权重是 Stage1 投票后**静态不变**的。但 Stage1 的域分类器只是一个粗糙估计，随着 Stage2 训练的进行，模型在 MMD 空间中能更准确地衡量源受试者与目标域的相似度。

V3 引入**周期性动态更新**：每隔若干 epoch，用当前模型提取源域和目标域的 `z_mmd` 特征，计算每个源受试者与目标域之间的 MMD 距离，距离越近权重越高，然后通过 EMA 平滑更新。

### 8.2 更新流程

```
每个 update_every 个 epoch (从 start_epoch 开始):

1. 模型 → eval 模式
2. 采样目标域 (最多 4096 条) → 提取 z_mmd
3. 逐源受试者采样 (每人最多 512 条) → 提取 z_mmd
4. 计算每源受试者 vs 目标的 MMD 距离:
     distance[s] = weighted_mmd_rbf(z_source[s], z_target)

5. 距离 → score:
     score[s] = exp(-(distance[s] - min_distance) / tau)

6. score 归一化 → 裁剪:
     weight[s] = clip(score[s] / mean(score), min=0.5, max=2.0)

7. EMA 平滑:
     new = momentum × old + (1-momentum) × new_mmd

8. Stage1 保底混合:
     new = α_stage1 × stage1_weight + (1-α_stage1) × new

9. 保存权重历史 → JSON
```

### 8.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dynamic_source_weight` | False | 开关，需显式开启 |
| `--dynamic_weight_start_epoch` | 5 | 从第几个 epoch 开始更新 |
| `--source_weight_update_every` | 5 | 每隔几个 epoch 更新一次 |
| `--source_weight_ema` | 0.8 | EMA 动量 (偏保守) |
| `--dynamic_weight_tau` | 1.0 | 温度参数 (越小权重分布越尖锐) |
| `--dynamic_weight_alpha_stage1` | 0.3 | Stage1 权重保底混合比例 |
| `--dynamic_weight_min` | 0.5 | 权重下限 |
| `--dynamic_weight_max` | 2.0 | 权重上限 |
| `--dynamic_max_samples_per_subject` | 512 | 每源受试者最大采样数 |
| `--dynamic_max_target_samples` | 4096 | 目标域最大采样数 |

---

## 9. 两阶段训练流程

### 9.1 训练总览

```
1. 数据准备
   ├── 交叉验证划分 (get_unified_subject_split)
   ├── 域 ID 映射 (source/val/test → domain_id)
   └── Source / Target dataloader

2. 被试基线统计
   ├── DE 基线: 每被试 DE 均值/标准差
   └── Bio 基线: 每被试 biomarker 均值/标准差

3. 阶段一训练 (Stage 1)
   ├── 域分类 + MMD + GRL 情绪/诊断对抗
   ├── 优化器: AdamW, lr=1e-4, CosineAnnealing
   ├── 类别权重平衡
   └── 保存最优域分类器 checkpoint

4. 源域权重投票
   ├── 目标域样本通过 domain classifier → 软投票
   ├── 统计每源域被试被选中频率
   └── 归一化为样本权重

5. 阶段二训练 (Stage 2)  [V3 增强]
   ├── 从 Stage1 最优模型初始化编码器
   ├── HC/DEP 专家 + 诊断路由 + 共享头联合训练
   ├── 加权损失 (源域投票权重)
   ├── [V3 新增] 条件 MMD (类别级对齐)
   ├── [V3 新增] Trial 排序损失
   ├── [V3 新增] 动态源权重更新
   ├── 多标准 best checkpoint 追踪 (8 个标准)
   └── 早停机制

6. 测试预测
   ├── 单模型: best checkpoint → soft top-K 预测
   └── 集成: 多折/多重复概率平均
```

### 9.2 优化器和调度器

| 配置 | Stage 1 | Stage 2 |
|------|---------|---------|
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 |
| 权重衰减 | 5e-4 | 5e-4 |
| 调度器 | CosineAnnealingLR | CosineAnnealingLR |

---

## 10. V3 相对 V2 的变化总结

### 10.1 模型架构

| 组件 | V2 | V3 |
|------|-----|-----|
| 共享编码器 (Backbone) | BrainGraphBackbone (PMG + RegionGCN) | **相同** |
| Stage1 结构 | domain_head + emotion_head + diag_head + mmd_head | **相同** |
| Stage2 结构 | diagnosis_router + shared_head + hc/dep expert + mmd_head + subject_domain_head | **相同** |
| 专家混合机制 | 诊断加权 + α 融合 | **相同** |

> **结论：V3 的模型网络结构与 V2 完全一致，变化都在训练策略和损失函数层面。**

### 10.2 损失函数

| 损失项 | V2 | V3 |
|--------|-----|-----|
| $\mathcal{L}_{\text{domain}}$ (S1) | ✓ | ✓ |
| $\mathcal{L}_{\text{MMD}}$ (S1) | ✓ (仅源权重) | ✓ (源+目标双权重, 数值稳定增强) |
| $\mathcal{L}_{\text{emo}}^{\text{GRL}}$ (S1) | ✓ | ✓ |
| $\mathcal{L}_{\text{diag}}^{\text{GRL}}$ (S1) | ✓ | ✓ |
| $\mathcal{L}_{\text{expert}}$ (S2) | ✓ | ✓ |
| $\mathcal{L}_{\text{mix}}$ (S2) | ✓ | ✓ |
| $\mathcal{L}_{\text{diag}}$ (S2) | ✓ | ✓ |
| $\mathcal{L}_{\text{MMD}}$ (S2) | ✓ (仅源权重) | ✓ (源+目标双权重) |
| $\mathcal{L}_{\text{subject}}$ (S2) | ✓ | ✓ |
| $\mathcal{L}_{\text{ent}}$ (S2) | ✓ | ✓ |
| **$\mathcal{L}_{\text{cond\_mmd}}$ (S2)** | **✗** | **✓ 新增 — 类别条件 MMD** |
| **$\mathcal{L}_{\text{rank}}$ (S2)** | **已定义但未启用** | **✓ 正式启用** |

### 10.3 训练机制

| 机制 | V2 | V3 |
|------|-----|-----|
| 源权重 | Stage1 静态投票 | 静态 + **动态更新** (EMA) |
| 动态权重更新周期 | 无 | 每 N epoch 基于 MMD 距离重算 |
| Stage1 权重保底 | 无 | α 混合保留 Stage1 信息 |
| 条件对齐 Warmup | 无 | 前 5 epoch 仅边缘 MMD |
| 排序损失 Warmup | 无 | 前 3 epoch 不启用 |

### 10.4 新增 CLI 参数

V3 新增约 16 个命令行参数，涵盖条件 MMD、排序损失、动态权重三大模块（详见 [§11](#11-超参数配置)）。

---

## 11. 超参数配置

### 11.1 完整命令行参数

| 参数组 | 参数 | 默认值 | 说明 |
|--------|------|--------|------|
| **数据** | `index_csv` | `com_index_sub_2s.csv` | 训练索引 |
| | `test_csv` | `com_test_trial_index_2s.csv` | 测试索引 |
| | `n_splits` | 10 | 交叉验证折数 |
| | `n_repeats` | 3 | 重复次数 |
| **训练** | `stage1_epochs` | 5 | 阶段一轮数 |
| | `stage2_epochs` | 25 | 阶段二轮数 |
| | `batch_size` | 200 | 批量大小 |
| | `lr_stage1/2` | 1e-4 | 学习率 |
| | `weight_decay` | 5e-4 | 权重衰减 |
| | `save_warmup_epochs` | 1 | 前 N 轮不触发 best 更新 |
| **模型** | `dropout` | 0.35 | Dropout 率 |
| | `shared_mix_alpha` | 0.5 | 共享/专家混合系数 |
| | `biomarker_dim` | 57 | 生物标记物维度 |
| **阶段一损失** | `lambda_domain` | 1.0 | 域分类权重 |
| | `stage1_lambda_mmd` | 0.03 | MMD 权重 |
| | `lambda_emo_grl` | 0.001 | 情绪对抗权重 |
| | `lambda_diag_grl` | 0.001 | 诊断对抗权重 |
| **阶段二损失** | `lambda_expert` | 0.5 | 专家损失 |
| | `lambda_mix` | 1.0 | 混合 NLL |
| | `lambda_diag` | 0.02 | 诊断 CE |
| | `stage2_lambda_mmd` | 3e-4 | 边缘 MMD |
| | `lambda_subject` | 3e-4 | 域对抗 |
| | `lambda_ent` | 0.0 | 熵正则 |
| **V3 条件 MMD** | `--lambda_cond_mmd` | 0.0003 | 条件 MMD 权重 |
| | `--cond_mmd_warmup_epochs` | 5 | Warmup 轮数 |
| | `--cond_mmd_conf_threshold` | 0.7 | 置信度阈值 |
| | `--cond_mmd_detach_target_prob` | True | 是否 detach |
| **V3 排序损失** | `--lambda_rank` | 0.03 | 排序损失权重 |
| | `--rank_warmup_epochs` | 3 | Warmup 轮数 |
| | `--rank_margin` | 0.2 | Pair 间隔 |
| | `--rank_max_pairs_per_subject` | 128 | 最大 pair 数 |
| **V3 动态权重** | `--dynamic_source_weight` | False | 开关 |
| | `--dynamic_weight_start_epoch` | 5 | 起始 epoch |
| | `--source_weight_update_every` | 5 | 更新间隔 |
| | `--source_weight_ema` | 0.8 | EMA 动量 |
| | `--dynamic_weight_tau` | 1.0 | 温度参数 |
| | `--dynamic_weight_alpha_stage1` | 0.3 | Stage1 保底比例 |
| | `--dynamic_weight_min` | 0.5 | 权重下限 |
| | `--dynamic_weight_max` | 2.0 | 权重上限 |
| | `--dynamic_max_samples_per_subject` | 512 | 每受试者最大样本 |
| | `--dynamic_max_target_samples` | 4096 | 目标域最大样本 |
| **预测** | `k_pos` | 4 | Soft Top-K 正类数 |
| | `test_vote_method` | soft_topk | 测试投票策略 |
| | `predict_best_name` | topk_trial_f1 | 预测用 best checkpoint |

### 11.2 典型运行命令

```bash
# V3 基础训练 (5 折)
python V3/train_expert_ssas_emotion.py --all_folds \
    --stage1_epochs 20 --stage2_epochs 100 \
    --test_vote_method soft_topk --k_pos 4

# V3 全特性训练 (条件 MMD + 排序损失 + 动态权重)
python V3/train_expert_ssas_emotion.py --all_folds \
    --stage1_epochs 20 --stage2_epochs 100 \
    --lambda_cond_mmd 0.0003 --cond_mmd_warmup_epochs 5 \
    --lambda_rank 0.03 --rank_warmup_epochs 3 \
    --dynamic_source_weight --dynamic_weight_start_epoch 5 \
    --source_weight_update_every 5 --source_weight_ema 0.8 \
    --test_vote_method soft_topk --k_pos 4
```

---

## 附录 A：模块依赖关系

```
V3/expert_ssas_emotion_model.py
    ├── V3/pmg_backbone.py
    │   ├── BrainGraphBackbone (三分支编码器 + PMG)
    │   ├── RegionPLVPMGEncoder (RegionGCN 区域编码)
    │   ├── NodeFeatureEncoder
    │   ├── RawSignalPLVGraphConstructor
    │   ├── WeightedGCNEncoder
    │   ├── FlattenGraphReadout
    │   └── BiologicalMarkerExtractor
    └── torch (GradReverse, nn.Module)

V3/train_expert_ssas_emotion.py
    ├── V3/expert_ssas_emotion_model.py (模型定义)
    ├── config (全局配置, V3_seed)
    ├── dataloader (Competition4ClassDataset)
    ├── utils/data (数据路径)
    ├── utils/folds (被试划分)
    └── sklearn.metrics (评估指标)
```

## 附录 B：V3 关键创新点速查

| 创新 | 作用 | 所在文件 | 默认状态 |
|------|------|---------|---------|
| 条件 MMD | 按类别对齐源/目标域分布 | `expert_ssas_emotion_model.py` | 启用 (λ=0.0003) |
| Trial 排序损失 | 训练目标与 topk 推理对齐 | `train_expert_ssas_emotion.py` | 启用 (λ=0.03) |
| 动态源权重 | 根据当前模型自适应调整源权重 | `train_expert_ssas_emotion.py` | 关闭 (需 `--dynamic_source_weight`) |
| MMD 双端权重 | 目标域也支持加权 | `expert_ssas_emotion_model.py` | 默认等效于均匀权重 |
| MMD 非负裁剪 | 数值稳定性增强 | `expert_ssas_emotion_model.py` | 始终启用 |
