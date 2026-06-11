# Two-Branch Subject-Relative EEG Encoder: 模型架构详细报告

> 基于 `V1/two_branch_subject_relative.py` 的编码器架构分析
>
> 生成日期：2026-06-11

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [节点特征编码器 (NodeFeatureEncoder)](#2-节点特征编码器-nodefeatureencoder)
3. [多尺度时间编码器 (MultiScaleTemporalEncoder)](#3-多尺度时间编码器-multiscaletemporalencoder)
4. [谱特征提取模块](#4-谱特征提取模块)
5. [图构造模块](#5-图构造模块)
6. [图编码与读出模块](#6-图编码与读出模块)
7. [生物标记物提取分支 (Biomarker Extractor)](#7-生物标记物提取分支-biomarker-extractor)
8. [完整数据流与维度变换](#8-完整数据流与维度变换)
9. [参数量估算](#9-参数量估算)
10. [关键创新点总结](#10-关键创新点总结)

---

## 1. 总体架构概览

本模型 `BrainGraphBackbone` 是一个**三分支脑电信号编码器**，用于从 30 通道 EEG 信号中提取多粒度特征表示。模型输入为原始 EEG 信号 $\mathbf{x} \in \mathbb{R}^{B \times 30 \times T}$ 及其差分熵特征 $\mathbf{de} \in \mathbb{R}^{B \times 30 \times K}$，输出三个互补的特征表示。

### 1.1 整体结构图

```
                          ┌─────────────────────────────────────┐
                          │          EEG Input x                │
                          │         [B, 30, T]                  │
                          └──────────┬──────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────────┐
          │                          │                              │
          ▼                          ▼                              ▼
 ┌────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
 │  NodeFeature   │    │ LearnableDEDiagonalFusion│    │  BiologicalMarker   │
 │   Encoder      │    │  (DE → Diag Values)     │    │   Extractor         │
 │                │    │                         │    │                     │
 │ [B,30,64]      │    │ [B,30]                  │    │ [B,57]              │
 └───────┬────────┘    └───────────┬─────────────┘    └──────────┬──────────┘
         │                         │                             │
         │                ┌────────┴────────┐                    │
         │                ▼                 ▼                    │
         │    ┌──────────────────┐  ┌───────────────┐            │
         │    │ PLV Graph        │  │ z_conv (Branch│            │
         │    │ Constructor      │  │ 1) Flatten    │            │
         │    │ [B,30,30]        │  │ Readout       │            │
         │    └────────┬─────────┘  │ [B,64]        │            │
         │             │            └───────────────┘            │
         │    ┌────────┴─────────┐                               │
         │    │ WeightedGCN      │                               │
         │    │ Encoder (2-layer)│                               │
         │    │ [B,30,64]        │                               │
         │    └────────┬─────────┘                               │
         │             │                                         │
         │    ┌────────┴─────────┐                               │
         │    │ z_plv (Branch 2) │                               │
         │    │ Flatten Readout  │                               │
         │    │ [B,64]           │                               │
         └────┼──────────────────┘                               │
              │                                                  │
              ▼                    ▼                             ▼
        z_conv [B,64]      z_plv [B,64]                  z_bio [B,57]
              │                    │                             │
              └────────────────────┼─────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   z = concat(z_conv,        │
                    │        z_plv, z_bio)        │
                    │      [B, 185]               │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────┐
          ▼                        ▼                          ▼
   ┌──────────────┐      ┌──────────────────┐     ┌──────────────────┐
   │  cls4_head   │      │   cls2_head      │     │  domain_head     │
   │  4-class     │      │   2-class        │     │  subject-domain  │
   │  [B,4]       │      │   [B,2]          │     │  [B,N_subjects]  │
   └──────────────┘      └──────────────────┘     └──────────────────┘
```

### 1.2 三分支分工

| 分支 | 编码方式 | 输出维度 | 用途 |
|------|----------|----------|------|
| **分支 1** (z_conv) | 多尺度时间卷积 → 图读出 | `[B, 64]` | 四分类情绪识别 |
| **分支 2** (z_plv) | PLV 脑网络 + GCN → 图读出 | `[B, 64]` | 二分类抑郁诊断 |
| **分支 3** (z_bio) | 显式 EEG 生物标记物提取 | `[B, 57]` | 二分类抑郁诊断 |

---

## 2. 节点特征编码器 (NodeFeatureEncoder)

### 2.1 类定义

```python
class NodeFeatureEncoder(nn.Module):
    """Full node feature encoder:
    - temporal CNN branch (MultiScaleTemporalEncoder)
    - spectral band-power branch (DifferentialEntropyExtractor + SpectralMLP)
    - Hjorth parameters

    Input:  x [B, C, T]
    Output: dict {h_raw, h_spec, hjorth}
    """
```

### 2.2 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sfreq` | 250.0 Hz | EEG 采样率 |
| `temporal_dim` | 64 | 时间分支输出维度 |
| `spectral_dim` | 32 | 谱分支输出维度 |
| `node_dim` | 64 (= temporal_dim) | 最终节点特征维度 |
| `branch_dim` (TSception) | 64 | 多尺度分支的内部维度 |

### 2.3 子模块组成

```
NodeFeatureEncoder
    ├── MultiScaleTemporalEncoder (TSception-style)
    │   ├── 4 个多尺度卷积分支 (kernel_sizes=[17,33,65,125])
    │   ├── Temporal Block (depthwise-separable conv)
    │   └── Attention Pooling
    ├── DifferentialEntropyExtractor
    │   └── FFT → Band Power → DE = 0.5·log(2πeσ²)
    ├── Hjorth Parameters
    │   └── Activity, Mobility, Complexity (per channel)
    └── SpectralMLP
        └── Linear(30×4, 128) → GELU → Linear(128, 32)
```

### 2.4 前向传播

**输入：** $\mathbf{x} \in \mathbb{R}^{B \times C \times T}$（30通道原始EEG信号）

**输出字典：**

| 键 | 形状 | 含义 |
|----|------|------|
| `h_raw` | `[B, C, 64]` | 多尺度时间卷积特征（→ 图节点特征） |
| `h_spec` | `[B, 32]` | 全局谱特征 |
| `hjorth` | `[B, C×3]` = `[B, 90]` | Hjorth 参数（Activity, Mobility, Complexity） |

---

## 3. 多尺度时间编码器 (MultiScaleTemporalEncoder)

### 3.1 设计思想

该模块是 TSception 风格的时间分支，用于替换传统的单尺度卷积前端。核心设计原则：
- **通道独立编码**：每个 EEG 通道共享同一套时间分支参数，避免提前混合通道信息
- **多尺度感受野**：使用 4 个不同长度的卷积核捕获不同时间尺度的 EEG 节律
- **深度可分离卷积**：减少参数量，防止过拟合
- **注意力汇聚**：取代简单的平均池化，自适应关注关键时间片段

### 3.2 卷积核尺度设计

在 250 Hz 采样率下，4 个卷积核对应的时间尺度为：

| 卷积核大小 | 时间尺度 | 对应节律 | 该尺度下的有效感受野 |
|-----------|----------|---------|---------------------|
| $k_1 = 17$ | $\approx 68\text{ ms}$ | Gamma 波 (~30-45 Hz) | 约 2-3 个周期 |
| $k_2 = 33$ | $\approx 132\text{ ms}$ | Beta 波 (~13-30 Hz) | 约 1.7-4 个周期 |
| $k_3 = 65$ | $\approx 260\text{ ms}$ | Alpha 波 (~8-13 Hz) | 约 2-3 个周期 |
| $k_4 = 125$ | $\approx 500\text{ ms}$ | Theta 波 (~4-8 Hz) | 约 2-4 个周期 |

所有卷积核使用**奇数大小**，确保 `padding = kernel_size // 2` 后各分支输出时间长度一致。

### 3.3 网络结构详解

```
MultiScaleTemporalEncoder
    │
    ├─ 输入: x [B, C, T] → reshape → [B*C, 1, T]
    │
    ├─ 多尺度卷积分支 (4 branches)
    │   ├─ Branch 1: Conv1d(1→64, k=17, pad=8) → BN → GELU → Dropout(0.15)
    │   ├─ Branch 2: Conv1d(1→64, k=33, pad=16) → BN → GELU → Dropout(0.15)
    │   ├─ Branch 3: Conv1d(1→64, k=65, pad=32) → BN → GELU → Dropout(0.15)
    │   └─ Branch 4: Conv1d(1→64, k=125, pad=62) → BN → GELU → Dropout(0.15)
    │   └─ concat → [B*C, 256, T]
    │
    ├─ Temporal Block
    │   ├─ AvgPool1d(k=4, stride=4)              → [B*C, 256, T/4]
    │   ├─ DepthwiseConv1d(256→256, k=7, groups=256) → [B*C, 256, T/4]
    │   ├─ PointwiseConv1d(256→64, k=1)           → [B*C, 64, T/4]
    │   ├─ BN → GELU → Dropout(0.3)
    │   ├─ DilatedDepthwiseConv1d(64→64, k=7, d=2, groups=64) → [B*C, 64, T/4]
    │   ├─ PointwiseConv1d(64→64, k=1)            → [B*C, 64, T/4]
    │   └─ BN → GELU → Dropout(0.3)
    │
    ├─ Attention Pooling
    │   ├─ Conv1d(64→32, k=1) → GELU
    │   ├─ Conv1d(32→1, k=1) → Softmax (over time dim)
    │   └─ Weighted sum: h = Σ_t α_t · h_t
    │
    └─ 输出: h [B, C, 64]
```

### 3.4 关键公式

**多尺度特征拼接：**

$$\mathbf{h}_{\text{multi}} = \big[\mathbf{f}_{17}(\mathbf{x}); \mathbf{f}_{33}(\mathbf{x}); \mathbf{f}_{65}(\mathbf{x}); \mathbf{f}_{125}(\mathbf{x})\big] \in \mathbb{R}^{B\cdot C \times 256 \times T}$$

**深度可分离时间卷积（第一阶段）：**

$$\mathbf{h}_{\text{mid}} = \sigma\big(\text{BN}(\mathbf{W}_{\text{point}} * \text{BN}(\mathbf{W}_{\text{depth}} \circledast \text{Pool}(\mathbf{h}_{\text{multi}})))\big) \in \mathbb{R}^{B\cdot C \times 64 \times T/4}$$

其中 $*$ 表示 1×1 逐点卷积，$\circledast$ 表示分组的逐通道深度卷积。

**注意力池化：**

$$\alpha_t = \frac{\exp(\mathbf{w}_\alpha^\top \cdot \text{GELU}(\mathbf{W}_a \cdot \mathbf{h}_t))}{\sum_{t'} \exp(\mathbf{w}_\alpha^\top \cdot \text{GELU}(\mathbf{W}_a \cdot \mathbf{h}_{t'}))}$$

$$\mathbf{h} = \sum_{t} \alpha_t \cdot \mathbf{h}_t \in \mathbb{R}^{B\cdot C \times 64}$$

### 3.5 与原始 TSception 的差异

| 方面 | 原始 TSception | 本实现 |
|------|---------------|--------|
| 分支维度 | 通常 64-128 | 64 (TSception-lite) |
| 时间块结构 | 标准深度可分离 | 增加膨胀深度卷积扩大感受野 |
| 汇聚方式 | 通常平均池化 | 注意力池化 (Softmax加权) |
| 通道策略 | 通道混合 | 通道独立（reshape到batch维） |

---

## 4. 谱特征提取模块

### 4.1 差分熵提取器 (DifferentialEntropyExtractor)

**原理：** 对每个通道、每个频带计算差分熵（Differential Entropy, DE）：

$$DE = \frac{1}{2} \log(2\pi e \sigma^2)$$

其中 $\sigma^2$ 用该频带的 FFT 平均功率近似。

**频带划分（250 Hz 采样率）：**

| 频带 | 频率范围 | FFT 频点范围 |
|------|---------|-------------|
| Theta (θ) | 4–8 Hz | $[4T/250, 8T/250)$ |
| Alpha (α) | 8–13 Hz | $[8T/250, 13T/250)$ |
| Beta (β) | 13–30 Hz | $[13T/250, 30T/250)$ |
| Gamma (γ) | 30–45 Hz | $[30T/250, 45T/250)$ |

**计算流程：**

$$\mathbf{X}_f = \text{FFT}(\mathbf{x}) \in \mathbb{C}^{B \times C \times T_{\text{freq}}}$$

$$P(\omega) = \frac{1}{T}\big(\Re(\mathbf{X}_f)^2 + \Im(\mathbf{X}_f)^2\big) \in \mathbb{R}^{B \times C \times T_{\text{freq}}}$$

$$\text{DE}_k = \frac{1}{2} \log\Big(2\pi e \cdot \frac{1}{|\mathcal{B}_k|} \sum_{\omega \in \mathcal{B}_k} P(\omega) + \epsilon\Big) \in \mathbb{R}^{B \times C}$$

其中 $\mathcal{B}_k$ 为第 $k$ 个频带的频点集合。

**输出：** `de_features [B, C, 4]` — 每个通道 4 个频带的 DE 特征。

### 4.2 Hjorth 参数

Hjorth 参数提供时域统计特征，计算效率高，补充频率域信息。

| 参数 | 公式 | 物理意义 |
|------|------|---------|
| Activity | $\text{Var}(x(t))$ | 信号功率/能量 |
| Mobility | $\sqrt{\frac{\text{Var}(\dot{x})}{\text{Var}(x)}}$ | 平均频率的估计 |
| Complexity | $\frac{\sqrt{\text{Var}(\ddot{x})/\text{Var}(\dot{x})}}{\sqrt{\text{Var}(\dot{x})/\text{Var}(x)}}$ | 信号形态复杂度 |

对每个通道独立计算，输出 $\mathbf{h}_{\text{hjorth}} \in \mathbb{R}^{B \times (C \times 3)} = \mathbb{R}^{B \times 90}$。

### 4.3 谱 MLP (SpectralMLP)

将各通道频带 DE 特征映射为全局谱嵌入：

```python
SpectralMLP:
    Linear(30×4=120, 128) → GELU
    Linear(128, 32)
```

**输出：** `h_spec [B, 32]`

---

## 5. 图构造模块

### 5.1 DE 对角线融合 (LearnableDEDiagonalFusion)

将 4 个节律的 DE 特征通过可学习权重融合为脑网络主对角线值：

$$\mathbf{w}_{\text{band}} = \text{Softmax}(\mathbf{p}_{\text{logits}}) \in \mathbb{R}^{4}$$

$$d_i = \sigma\Big(\gamma \cdot \sum_{k=1}^{4} w_k \cdot \text{Norm}(\text{DE}_{i,k}) + \beta\Big) \in (0, 1)$$

其中 $\gamma$ 为可学习尺度参数，$\beta$ 为可学习偏置，$\sigma$ 为 sigmoid 函数。输出 $\mathbf{d} \in \mathbb{R}^{B \times C}$ 作为每个通道的自连接强度。

### 5.2 PLV 脑网络构造 (RawSignalPLVGraphConstructor)

基于原始 EEG 信号，使用相位锁定值（Phase Locking Value, PLV）构造功能连接矩阵：

**步骤：**

1. **Hilbert 变换：** 通过 FFT 实现解析信号，提取瞬时相位

   $$\mathbf{x}_a(t) = \mathcal{H}\{\mathbf{x}(t)\} = \text{IFFT}\{\mathbf{H}(\omega) \cdot \text{FFT}\{\mathbf{x}(t)\}\}$$

   其中 $\mathbf{H}(\omega)$ 为频域 Hilbert 滤波器。

2. **相位提取：** $\phi_c(t) = \angle \mathbf{x}_{a,c}(t) \in \mathbb{R}^{B \times C \times T}$

3. **PLV 计算：** 使用复数矩阵乘法避免显式构造 $[B, C, C, T]$ 张量

   $$\Phi_c(t) = e^{j\phi_c(t)} \in \mathbb{C}^{B \times C \times T}$$

   $$\text{PLV}_{ij} = \frac{1}{T}\Big|\sum_{t} \Phi_i(t) \cdot \Phi_j^*(t)\Big| \in [0, 1]^{B \times C \times C}$$

4. **图后处理流程：**

   ```
   PLV Matrix [B,C,C]
       → 去自连接 (zero diagonal)
       → Top-K 稀疏化 (K=8, 保留每节点最强的8条边)
       → 对称化
       → 加 DE 对角线自环
       → 对称归一化 D^{-1/2} A D^{-1/2}
       → 归一化邻接矩阵 [B,C,C]
   ```

---

## 6. 图编码与读出模块

### 6.1 加权图卷积层 (WeightedGCNLayer)

$$\mathbf{H}' = \text{Dropout}\big(\text{GELU}(\text{LayerNorm}(\mathbf{A} \mathbf{H} \mathbf{W}))\big)$$

- 使用 LayerNorm 而非 BatchNorm，适用于小批量训练
- 激活函数采用 GELU 而非 ReLU

### 6.2 双层 GCN 编码器 (WeightedGCNEncoder)

```
WeightedGCNEncoder (node_dim=64, hidden_dim=64):
    GCN Layer 1: 64 → 64 (dropout=0.2)
    GCN Layer 2: 64 → 64 (dropout=0.2)
    Residual: h_out = h_out + h_in  (当维度匹配时)
```

输出节点嵌入 $\mathbf{H}_{\text{gcn}} \in \mathbb{R}^{B \times 30 \times 64}$。

### 6.3 展平图读出 (FlattenGraphReadout)

```python
FlattenGraphReadout:
    Flatten [B, 30, 64] → [B, 1920]
    Linear(1920, 256) → GELU → Dropout(0.2)
    Linear(256, 64)
```

输出图级表示 $\mathbf{z} \in \mathbb{R}^{B \times 64}$。

---

## 7. 生物标记物提取分支 (Biomarker Extractor)

分支 3 (`BiologicalMarkerExtractor`) 提取 **57 维显式 EEG 生物标记物**，无需可学习参数（或仅有少量归一化层），包括：

| 类别 | 维度 | 具体特征 |
|------|------|---------|
| 频带能量统计 | 33 | 各频带 (θ,α,β,γ) 功率 + DE 的均值/标准差/偏度/峰度 + SampEn |
| 左右半球不对称性 | 3 | 额叶、颞叶、顶枕叶区域的不对称比 |
| Hjorth 参数统计 | 10 | Activity/Mobility/Complexity 的全局统计量 |
| PLV 图指标 | 5 | 平均 PLV、聚类系数、全局效率、特征路径长度等 |
| 非线性复杂度 | 6 | 样本熵 (SampEn)、Lempel-Ziv 复杂度等 |

**输出：** `z_bio [B, 57]`

---

## 8. 完整数据流与维度变换

### 8.1 维度变换表

| 步骤 | 模块 | 输入形状 | 输出形状 |
|------|------|---------|---------|
| 1 | `NodeFeatureEncoder` | `x [B, 30, T]` | `node_features [B, 30, 64]` |
| 2 | `DifferentialEntropyExtractor` | `x [B, 30, T]` | `de_band [B, 30, 4]` |
| 3 | `hjorth_parameters` | `x [B, 30, T]` | `hjorth [B, 90]` |
| 4 | `LearnableDEDiagonalFusion` | `de_band [B, 30, 4]` | `diag_values [B, 30]` |
| 5 | `PLV Graph Constructor` | `x [B, 30, T]` + `diag [B, 30]` | `adj_norm [B, 30, 30]` |
| 6 | `WeightedGCNEncoder` | `node_features [B, 30, 64]` + `adj [B, 30, 30]` | `plv_nodes [B, 30, 64]` |
| 7a | `FlattenGraphReadout (conv)` | `node_features [B, 30, 64]` | `z_conv [B, 64]` |
| 7b | `FlattenGraphReadout (plv)` | `plv_nodes [B, 30, 64]` | `z_plv [B, 64]` |
| 8 | `BiologicalMarkerExtractor` | `x, de, plv_matrix` | `z_bio [B, 57]` |
| 9 | Concat | `z_conv, z_plv, z_bio` | `z [B, 185]` |

### 8.2 最终输出汇总

```python
output = {
    # 主输出
    "z_conv":  [B, 64],   # 分支1: 多尺度卷积特征 (→ 四分类情绪)
    "z_plv":   [B, 64],   # 分支2: PLV图特征
    "z_bio":   [B, 57],   # 分支3: 显式EEG生物标记物
    "z_diag":  [B, 121],  # z_plv ⊕ z_bio (→ 二分类抑郁诊断)
    "z":       [B, 185],  # 三路全拼接 (兼容旧接口)

    # 中间表示
    "node_features":       [B, 30, 64],   # 节点特征
    "node_contrast_feat":  [B, 64],       # 对比学习特征
    "node_embeddings":     [B, 30, 64],   # GCN后的节点嵌入

    # 图结构
    "adj_norm":      [B, 30, 30],   # 归一化邻接矩阵
    "plv_matrix":    [B, 30, 30],   # 原始PLV矩阵
    "plv_adj_norm":  [B, 30, 30],   # PLV归一化邻接矩阵

    # 辅助信息
    "power_diag_values":  [B, 30],    # DE对角线自连接值
    "de_band_weight":     [4],        # 可学习频带权重
    "hjorth":             [B, 90],    # Hjorth参数
    "bio_raw":            [B, 57],    # 原始生物标记物
}
```

---

## 9. 参数量估算

### 9.1 MultiScaleTemporalEncoder

| 组件 | 参数量 | 计算 |
|------|--------|------|
| 4 个卷积分支 | $4 \times (1 \times 64 \times k)$ | 约 $4 \times 64 \times 60 \approx 15,360$ |
| 每个分支 BN | $4 \times 128$ | 512 |
| Depthwise Conv (7) | $256 \times 7 / 256 \approx 7$ | ~1,792 |
| Pointwise Conv (1) | $256 \times 64$ | 16,384 |
| Dilated Depthwise (7) | $64 \times 7$ | 448 |
| Pointwise Conv (1) | $64 \times 64$ | 4,096 |
| Attention Pooling | $64 \times 32 + 32 \times 1$ | 2,080 |
| **小计** | | **≈ 40,672** |

### 9.2 总参数量估算

| 子模块 | 估算参数量 |
|--------|-----------|
| MultiScaleTemporalEncoder | ~40K |
| SpectralMLP | 120×128 + 128×32 ≈ 19K |
| LearnableDEDiagonalFusion | 极少 (~25) |
| WeightedGCNEncoder (2层) | 64×64×2 ≈ 8K |
| FlattenGraphReadout × 2 | 2 × (1920×256 + 256×64) ≈ 1,016K |
| BiologicalMarkerExtractor | 极少 (~可忽略) |
| Classification Heads (3个) | 3 × (185×64 + 64×C) ≈ 36K |
| **总计 (不含分类头)** | **≈ 1,083K (~1.1M)** |

---

## 10. 关键创新点总结

### 10.1 模块级创新

1. **TSception 风格多尺度时间编码器**
   - 4 个不同感受野的卷积分支（对应 θ/α/β/γ 节律时间尺度）
   - 深度可分离膨胀卷积扩大感受野
   - 注意力池化替代平均池化

2. **PLV 功能连接脑网络**
   - 基于 Hilbert 解析信号的相位锁定值
   - 复数矩阵乘法避免显式 4D 张量构造
   - 可学习 DE 对角线替换固定自环

3. **可学习频带自适应融合**
   - Softmax 加权融合 4 频带 DE 特征
   - 学习到的权重反映各频带在不同任务中的重要性

4. **显式 EEG 生物标记物分支**
   - 57 维手工设计特征（频带统计+不对称性+Hjorth+非线性）
   - 与深度学习特征互补，增强可解释性

### 10.2 架构级创新

1. **三分支异构编码**
   - 时间/谱/图三个视角独立建模
   - 分支间无信息泄露（PLV 使用原始信号而非学习特征）
   - 任务级特征分离：情绪识别 (z_conv) vs 抑郁诊断 (z_plv+z_bio)

2. **被试相对特征归一化 (Subject-Relative)**
   - DE 特征：`de_rel = de - de_mu`, `de_z = de_rel / de_std`
   - Biomarker 特征：类似的被试内标准化
   - 降低个体差异对模型训练的影响

3. **多任务学习框架**
   - 四分类情绪头 + 二分类情绪头 + 被试域对抗头
   - 共享编码器、任务专属分类器
   - GRL 梯度反转实现域对抗

---

## 附录 A：超参数配置

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `sfreq` | 250 Hz | EEG 采样率 |
| `node_dim` | 64 | 图节点特征维度 |
| `attn_dim` | 16 | 注意力维度 |
| `topk` | 8 | Top-K 稀疏化保留边数 |
| `dropout` | 0.2 | 通用 Dropout 率 |
| `branch_dim` | 64 | TSception 分支内部维度 |
| `kernel_sizes` | [17, 33, 65, 125] | 多尺度卷积核 |
| `bands` | [(4,8),(8,13),(13,30),(30,45)] | 频带划分 |
| `bio_abs_scale` | 0.3 | 被试相对 biomarker 中的绝对值缩放 |
| `relative_eps` | 1e-6 | 防止除零的微小常数 |

## 附录 B：依赖关系

```
two_branch_subject_relative.py
    ├── models/diag_model_ssas.py  (DepressionBiomarkerExtractor, _safe_std)
    └── models/dep_contrast_bio.py (BiologicalMarkerExtractor)
```

---

> **引用格式建议：**
>
> 本编码器架构基于 TSception 的多尺度时间编码思想 [1]，结合相位锁定值 (PLV) 功能连接图 [2] 和显式 EEG 生物标记物特征 [3]，构建了三分支异构脑网络编码器。模型通过可学习频带权重融合、注意力时间池化、被试相对归一化等设计，实现了对 EEG 信号的多粒度特征提取。
>
> **参考文献：**
> [1] Ding Y, et al. "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition." IEEE TAC, 2022.
> [2] Lachaux JP, et al. "Measuring phase synchrony in brain signals." Human Brain Mapping, 1999.
> [3] Hjorth B. "EEG analysis based on time domain properties." Electroencephalography and Clinical Neurophysiology, 1970.
