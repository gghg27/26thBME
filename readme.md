# EEG 抑郁症辅助诊断与情绪识别技术报告

> 项目路径：`B:\26thbme\stage`  
> 编写日期：2026-06-08  
> 代码形态：PyTorch 深度学习训练、验证、测试推理与多折集成流水线

## 1. 项目概述

本项目面向 EEG 脑电信号，构建了一套“抑郁症辅助诊断 + 情绪识别”的深度学习系统。系统不是只训练一个单一分类器，而是将问题拆成三个模型和一个两阶段推理管线：

| 模块 | 输入窗口 | 训练对象 | 输出 |
|------|----------|----------|------|
| 诊断模型 | 10 秒 EEG 窗口 | 全部训练被试 | 被试属于 DEP 或 HC 的概率 |
| HC 情绪模型 | 2 秒 EEG 窗口 | HC 正常组被试 | neutral / positive 情绪概率 |
| DEP 情绪模型 | 2 秒 EEG 窗口 | DEP 抑郁组被试 | neutral / positive 情绪概率 |
| 两阶段融合 | 诊断概率 + 两个情绪概率 | 测试 trial | 最终情绪标签 |

整体思路可以概括为：

1. 先用诊断模型判断一个被试更像 HC 还是 DEP。
2. 再分别用 HC 情绪模型和 DEP 情绪模型预测该 trial 的情绪。
3. 最后用诊断概率做软路由，把两个情绪模型的结果融合成最终情绪预测。

这样做的动机是：正常人和抑郁患者的情绪相关 EEG 表征可能存在差异。将 HC 和 DEP 分开建模，再由诊断模型动态决定更信任哪一路情绪模型，比直接训练一个统一情绪分类器更符合这个任务的业务逻辑。

## 2. 数据与标签设计

### 2.1 数据规模

训练索引文件中共有 60 个训练被试，其中：

| 诊断类别 | 被试数 | 10s 窗口数 | 2s 窗口数 |
|----------|--------|------------|-----------|
| DEP | 20 | 800 | 7840 |
| HC | 40 | 1600 | 15680 |
| 合计 | 60 | 2400 | 23520 |

测试索引文件包含 10 个测试用户、80 个 trial。10 秒测试索引中每个 trial 通常对应 1 个诊断窗口；2 秒测试索引中每个 trial 通常对应 9 个情绪窗口。

### 2.2 四分类标签 `label4`

项目中最稳定、最清晰的原始组合标签是 `label4`：

| `label4` | 诊断组别 | 情绪类别 | 含义 |
|----------|----------|----------|------|
| 0 | DEP | neutral | 抑郁组中性情绪 |
| 1 | DEP | positive | 抑郁组正性情绪 |
| 2 | HC | neutral | 正常组中性情绪 |
| 3 | HC | positive | 正常组正性情绪 |

由 `label4` 可以稳定推出两个二分类任务标签：

```text
emotion_label = label4 % 2
0 = neutral, 1 = positive

group_binary = label4 >= 2
0 = DEP, 1 = HC
```

需要特别注意：当前 CSV 中的 `diagnosis_label` 字段与部分代码注释并不完全一致。训练索引实际显示 `DEP=1, HC=0`，而部分情绪训练脚本为了避免编码不统一，会优先用 `label4 >= 2` 推出 `0=DEP, 1=HC`。因此报告和后续使用中应把 `label4` 当作诊断组别的更可靠来源；诊断模型推理中 `class1_prob` 被命名为 `p_dep`，对应 checkpoint/meta 中的诊断标签映射 `{"HC": 0, "DEP": 1}`。

### 2.3 数据索引文件

| 文件 | 用途 | 关键字段 |
|------|------|----------|
| `com_index_sub_10s.csv` | 诊断模型训练和验证 | `subject_id`, `trial_path`, `start`, `end`, `de_path`, `de_win_id`, `diagnosis_label`, `label4` |
| `com_index_sub_2s.csv` | HC/DEP 情绪模型训练和验证 | `subject_id`, `trial_path`, `start`, `end`, `de_path`, `de_win_id`, `emotion_label`, `label4` |
| `com_test_trial_index_10s.csv` | 测试集诊断推理 | `user_id`, `trial_id`, `trial_path`, `de_path`, `n_windows` |
| `com_test_trial_index_2s.csv` | 测试集情绪推理 | `user_id`, `trial_id`, `trial_path`, `de_path`, `n_windows` |

### 2.4 EEG 与特征输入

模型同时使用两类输入：

1. 原始 EEG 窗口 `x`，形状一般为 `[B, 30, T]`。
2. 差分熵特征 `de_feat`，由 `de_path` 加载，窗口级特征通过 `de_win_id` 选择。

窗口设置如下：

| 任务 | 窗口长度 | 采样点 | 作用 |
|------|----------|--------|------|
| 诊断 | 10 秒 | 2500 点，采样率 250 Hz | 更适合被试级稳定诊断 |
| 情绪 | 2 秒 | 500 点，50% 重叠 | 更适合捕捉 trial 内短时情绪变化 |

`dataloader.py` 用于训练阶段，会返回字典格式 batch，包括 `x`、`de_feat`、`label4`、`emotion_label`、`diagnosis_label`、`subject_id`、`domain_id`、`trial_id`。`utils/data.py` 用于推理阶段，负责从 trial 级测试索引展开到窗口级样本，并加载原始 EEG 与 DE 特征。

## 3. 交叉验证与数据防泄露

### 3.1 统一跨被试划分

项目使用 `StratifiedGroupKFold` 做跨被试五折交叉验证。这里有两个重点：

1. 分层依据是诊断组别，使每一折中 DEP 和 HC 的比例尽量稳定。
2. 分组依据是 `subject_id`，确保同一个被试不会同时出现在训练集和验证集中。

`utils/folds.py` 中的 `get_unified_subject_split()` 统一生成一套 train/val 被试划分，供诊断、HC 情绪、DEP 情绪三个模型共用。这样可以避免三个模型各自随机划分，导致级联推理时验证集被试不一致。

五折下，总共有 60 个训练被试，所以每折大致是：

```text
60 个训练被试
  -> 5 折交叉验证
  -> 每折约 48 个训练被试、12 个验证被试
```

### 3.2 三个模型如何使用同一折

以某一个 fold 为例：

```text
本 fold 的 train_all
  -> 诊断模型使用全部训练被试，学习 HC/DEP 二分类
  -> HC 情绪模型只取 train_all 中的 HC 被试，学习 neutral/positive
  -> DEP 情绪模型只取 train_all 中的 DEP 被试，学习 neutral/positive

本 fold 的 val_all
  -> 诊断、HC 情绪、DEP 情绪推理时都只在这些验证被试上评估
```

这种设计保证验证阶段模拟真正的未知被试场景：模型从未见过验证被试的任何窗口。

### 3.3 多随机种子重复

全局随机种子配置在 `config.py`：

```python
DIAG_REPEAT_SEEDS = [20, 42, 123]
HC_REPEAT_SEEDS = [20, 42, 123]
DEP_REPEAT_SEEDS = [20, 42, 123]
N_SPLITS = 5
```

每个 `repeat` 对应一个基础 seed，每个 fold 再生成独立 `run_seed`：

```python
run_seed = base_seed * 1000 + fold
```

理论上完整训练会得到：

```text
3 个 repeat × 5 个 fold × 3 个任务模型 = 45 个 checkpoint
```

测试集最终可以对 15 套 `(repeat, fold)` 的两阶段结果做概率平均集成。

## 4. 模型总体设计

三个模型的核心思想相同：先从原始 EEG 中抽取时间特征和频域/图特征，再通过图神经网络建模通道之间的脑连接关系，最后输出分类结果。

### 4.1 共同 Backbone

模型文件包括：

| 文件 | 模型角色 |
|------|----------|
| `models/diagnosis_model.py` | 诊断模型 |
| `models/hc_contrast_bio.py` | HC 情绪模型 |
| `models/dep_contrast_bio.py` | DEP 情绪模型 |

共同核心组件如下：

| 组件 | 技术作用 |
|------|----------|
| `MultiScaleTemporalEncoder` | 多尺度时间卷积编码器，类似 TSception 思路，用不同卷积核捕捉不同时间尺度的 EEG 模式 |
| `DifferentialEntropyExtractor` | 从 EEG 频带功率计算差分熵特征，补充传统 EEG 频域信息 |
| `LearnableDEDiagonalFusion` | 将 DE 频带信息转成每个通道的自连接强度，让 self-loop 不再是固定值 |
| `PairSpecificGraphConstructor` | 可学习的通道对注意力构图，为不同通道对生成不同连接权重 |
| `RawSignalPLVGraphConstructor` | 基于相位锁定值 PLV 构造功能连接图，提供脑网络先验 |
| `GatedWeightedGCNEncoder` | 门控加权图卷积，在通道图上传播和融合节点特征 |
| `FlattenGraphReadout` / `GraphReadout` | 将节点级图特征汇聚为样本级表示 |
| `GradReverse` / `GRLClassificationHead` | 梯度反转层，用于诊断或被试域对抗 |

### 4.2 诊断模型结构

诊断模型位于 `models/diagnosis_model.py`，入口类为 `EmotionPretrainModel`。虽然类名中有 `EmotionPretrainModel`，但在本项目当前使用中，它承担的是诊断二分类主任务。

诊断模型的数据流如下：

```text
原始 EEG x [B, 30, T]
  -> 多尺度时间编码 MultiScaleTemporalEncoder
  -> 得到每个通道的时间表征 h_raw

h_raw + de_feat
  -> 可学习注意力构图 adj_attn
  -> PLV 功能连接构图 adj_plv
  -> DE 自适应对角线 self-loop

adj_attn 分支
  -> GatedWeightedGCNEncoder
  -> FlattenGraphReadout
  -> z_conv

adj_plv 分支
  -> GatedWeightedGCNEncoder
  -> FlattenGraphReadout
  -> z_plv

生物标志物分支
  -> DepressionBiomarkerExtractor
  -> z_bio

concat(z_conv, z_plv, z_bio)
  -> diagnosis_logits
  -> subject/domain adversarial logits
```

这个模型的技术重点有三点：

1. 双图建模：一条图来自模型学习出的注意力连接，一条图来自 PLV 功能连接。
2. 生物标志物增强：显式加入抑郁相关 EEG 指标，例如频带不对称、Hjorth 参数、PLV 图统计、非线性复杂度等。
3. 被试域对抗接口：模型保留 subject/domain head，可以通过 GRL 减少被试个体差异，但当前训练配置中诊断域对抗权重为 0。

### 4.3 HC/DEP 情绪模型结构

HC 情绪模型和 DEP 情绪模型结构基本一致，分别位于：

```text
models/hc_contrast_bio.py
models/dep_contrast_bio.py
```

入口类同样为 `EmotionPretrainModel`。输出包括：

| 输出字段 | 作用 |
|----------|------|
| `emo_logits` | 情绪二分类主输出，0=neutral, 1=positive |
| `diagnosis_logits` | 诊断对抗辅助头，经过 GRL |
| `subject_logits` | 被试 ID 对抗辅助头，经过 GRL |
| `contrast_feat` | 监督对比学习使用的投影特征 |
| `domain_logits` | 兼容旧代码的别名，等价于 `subject_logits` |

情绪模型的数据流如下：

```text
原始 EEG x [B, 30, T] + de_feat
  -> 多尺度时间编码
  -> 生物标志物提取
  -> 可学习图构建 + 图卷积
  -> 图读出特征 z

z
  -> EmotionHead -> emo_logits
  -> GRL DiagnosisHead -> diagnosis_logits
  -> GRL SubjectHead -> subject_logits
  -> ContrastProjectionHead -> contrast_feat
```

HC 模型只在 HC 被试上训练，DEP 模型只在 DEP 被试上训练。这样两个模型分别学习各自群体内部的情绪区分模式。

## 5. 训练策略

### 5.1 诊断模型训练

训练脚本为：

```bash
python trainers/train_diag.py
```

核心配置：

| 参数 | 值 |
|------|----|
| 输入索引 | `com_index_sub_10s.csv` |
| 输入窗口 | 10 秒 |
| 模型 | `models.diagnosis_model.EmotionPretrainModel` |
| 任务类别数 | `nclass=2` |
| batch size | 32 |
| epoch | 100 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| scheduler | CosineAnnealingLR |
| early stopping | patience=25, warmup=15 |
| checkpoint 目录 | `model_params/diag_reapt{repeat}_fold{fold}` |

诊断模型当前主要损失为带类别权重的交叉熵：

```text
loss = CE(diagnosis_logits, diagnosis_label)
     + lambda_center * center_contrast_loss
     + lambda_dom * domain_loss
```

当前主入口中配置为：

| 损失项 | 当前权重 |
|--------|----------|
| 诊断交叉熵 | 1.0 |
| center contrast | 0.0 |
| domain loss | 0.0 |
| graph loss | 0.0 |
| contrast loss | 0.0 |

因此当前诊断训练的核心是监督式诊断二分类，其他项保留为可扩展接口。

### 5.2 HC 情绪模型训练

训练脚本为：

```bash
python trainers/train_hc.py
```

核心配置：

| 参数 | 值 |
|------|----|
| 输入索引 | `com_index_sub_2s.csv` |
| 输入窗口 | 2 秒 |
| 训练组 | `train_group="hc"` |
| batch size | 256 |
| epoch | 100 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| scheduler | CosineAnnealingLR |
| checkpoint 目录 | `model_params/hc_reapt{repeat}_fold{fold}` |

损失函数为：

```text
loss = lambda_emo * CE(emo_logits, emotion_label)
     + lambda_con * SupCon(contrast_feat, emotion_label)
     + lambda_diag * CE(diagnosis_logits, diagnosis_label)
     + lambda_subject * CE(subject_logits, domain_id)
     + lambda_graph * intra_class_graph_loss
```

当前主入口配置：

| 损失项 | 权重 |
|--------|------|
| 情绪交叉熵 `lambda_emo` | 1.0 |
| 监督对比学习 `lambda_con` | 0.05 |
| 诊断对抗 `lambda_diag` | 0.0 |
| 被试对抗 `lambda_subject` | 0.01 |
| 图一致性正则 `lambda_graph` | 0.0 |
| SupCon temperature | 0.1 |
| subject GRL 强度 | 0.01 |

### 5.3 DEP 情绪模型训练

训练脚本为：

```bash
python trainers/train_dep.py
```

它与 HC 情绪模型几乎完全一致，区别是：

```text
train_group = "dep"
checkpoint 目录 = model_params/dep_reapt{repeat}_fold{fold}
模型文件 = models/dep_contrast_bio.py
```

也就是说，DEP 模型只在抑郁组被试上学习 neutral/positive 情绪分类。

### 5.4 最优模型选择与早停

训练过程中不是只保存一个 best，而是并行维护多套 best tracker。例如情绪模型包含：

| best 名称 | 主要选择标准 |
|-----------|--------------|
| `combined` | trial_macro_f1 -> trial_acc -> emotion_macro_f1 -> emotion_acc -> loss |
| `trial_f1` | trial_macro_f1 -> trial_acc -> loss |
| `trial_acc` | trial_acc -> trial_macro_f1 -> loss |
| `segment_emo_f1` | emotion_macro_f1 -> emotion_acc -> trial_macro_f1 -> loss |

后续 `combine_fold_models.py` 默认组合每个任务目录中的 `*_best.pt`，实际代表训练脚本保存出的主要 best checkpoint。

## 6. 推理与融合流程

### 6.1 checkpoint 组合

单个 fold 的三个模型训练完成后，先运行：

```bash
python scripts/combine_fold_models.py --repeat 0 --fold 0
```

该脚本会查找：

```text
model_params/diag_reapt{repeat}_fold{fold}/diag_best.pt
model_params/hc_reapt{repeat}_fold{fold}/hc_best.pt
model_params/dep_reapt{repeat}_fold{fold}/dep_best.pt
```

然后生成：

```text
model_params/combined_reapt{repeat}_fold{fold}/combined_meta.json
```

`combined_meta.json` 保存三类模型的 checkpoint 路径、meta 路径，以及软路由公式：

```text
p_final = (1 - p_dep_subject) * p_hc + p_dep_subject * p_dep
```

### 6.2 验证集两阶段推理

验证脚本为：

```bash
python inference/infer_val_fold.py --repeat 0 --fold 0
```

流程如下：

1. 读取 `combined_meta.json`。
2. 加载诊断、HC 情绪、DEP 情绪三个模型。
3. 优先从 checkpoint/meta 中读取训练时保存的 `val_subjects`。
4. 检查验证被试是否与每个模型的训练被试重叠。
5. 在 10 秒窗口上运行诊断模型，得到每个窗口的 `p_dep`。
6. 按被试聚合诊断概率，得到 `p_dep_subject`。
7. 在 2 秒窗口上分别运行 HC 和 DEP 情绪模型。
8. 按 trial 聚合窗口概率，得到 `p_pos_hc` 和 `p_pos_dep`。
9. 用软路由公式融合，得到 `p_final`。
10. 以 0.5 为阈值得到 `pred_emotion`。

验证输出：

```text
predictions/reapt{repeat}_fold{fold}/val_two_stage_preds.csv
results/reapt{repeat}_fold{fold}_metrics.json
```

### 6.3 测试集两阶段推理

测试脚本为：

```bash
python inference/infer_test_fold.py --repeat 0 --fold 0
```

测试集没有真实标签，因此只输出预测概率和标签。流程和验证类似，但会先用 `expand_window_index()` 将 trial 级测试索引展开为窗口级样本。

输出：

```text
predictions/reapt{repeat}_fold{fold}/test_two_stage_preds.csv
```

核心字段包括：

| 字段 | 含义 |
|------|------|
| `p_dep_subject` | 诊断模型聚合出的被试级 DEP 概率 |
| `p_pos_hc` | HC 情绪模型对当前 trial 的 positive 概率 |
| `p_pos_dep` | DEP 情绪模型对当前 trial 的 positive 概率 |
| `p_final` | 软路由融合后的 positive 概率 |
| `pred_emotion` | 单 fold 下的情绪预测标签 |

### 6.4 多 fold 集成

多折测试预测完成后，运行：

```bash
python inference/ensemble_test.py
```

它会扫描：

```text
predictions/reapt*_fold*/test_two_stage_preds.csv
```

然后对同一 `user_id + trial_id` 的 `p_final` 求均值：

```text
p_final_ensemble = mean(p_final over all available folds)
Emotion_label = 1 if p_final_ensemble >= 0.5 else 0
```

最终输出：

```text
predictions/ensemble/test_10fold_probs.csv
predictions/ensemble/submission.xlsx
```

## 7. 一键流水线

项目提供 `run_pipeline.py` 作为统一入口：

```bash
python run_pipeline.py status
python run_pipeline.py combine
python run_pipeline.py val
python run_pipeline.py test
python run_pipeline.py all
```

各命令含义：

| 命令 | 作用 |
|------|------|
| `status` | 扫描 `model_params`，查看哪些 `(repeat, fold)` 的三模型 checkpoint 已齐全 |
| `combine` | 对就绪 fold 生成 `combined_meta.json` |
| `val` | 对就绪 fold 运行验证集两阶段推理，并汇总平均指标 |
| `test` | 对就绪 fold 运行测试集推理，然后执行多 fold 集成 |
| `all` | 依次执行 `combine -> val -> test` |

可以限制范围：

```bash
python run_pipeline.py combine --repeat 0
python run_pipeline.py val --repeat 0 --fold 2
python run_pipeline.py test --repeat 1 --fold 4
```

## 8. 评估指标

### 8.1 诊断任务指标

| 指标 | 级别 | 说明 |
|------|------|------|
| `diag_subject_acc` | subject | 被试级诊断准确率 |
| `trial_diagnosis_acc` | trial | trial 聚合后的诊断准确率 |
| `trial_diagnosis_macro_f1` | trial | trial 级宏平均 F1 |
| `subject_diagnosis_macro_f1` | subject | 被试级宏平均 F1 |

### 8.2 情绪任务指标

| 指标 | 级别 | 说明 |
|------|------|------|
| `emotion_trial_acc_soft` | trial | 两阶段软融合后的情绪准确率 |
| `emotion_macro_f1_soft` | trial | 两阶段软融合后的情绪宏平均 F1 |
| `hc_emotion_acc` | trial subset | HC 子集情绪准确率 |
| `hc_emotion_f1` | trial subset | HC 子集情绪宏平均 F1 |
| `dep_emotion_acc` | trial subset | DEP 子集情绪准确率 |
| `dep_emotion_f1` | trial subset | DEP 子集情绪宏平均 F1 |

窗口到 trial、trial 到 subject 的聚合逻辑主要在训练脚本和 `utils/predict.py` 中实现。诊断被试级概率采用窗口或 trial 概率均值；情绪 trial 级概率采用同一 trial 内多个 2 秒窗口的均值。

## 9. 技术亮点

### 9.1 原始 EEG 与 DE 特征联合建模

模型不是只吃手工特征，也不是只吃原始信号，而是同时利用原始 EEG 的时序模式和 DE 频域统计。原始信号提供端到端可学习能力，DE 特征提供更稳定的 EEG 频带先验。

### 9.2 多尺度时间编码

EEG 信号的有效模式可能出现在不同时间尺度上。`MultiScaleTemporalEncoder` 使用多分支卷积核提取短时和较长时程的节律变化，再通过注意力/池化得到通道级表征。

### 9.3 双图脑网络建模

项目同时使用：

1. 可学习注意力图：让模型从数据中学习通道间关系。
2. PLV 功能连接图：用相位同步关系表达更有脑网络意义的连接先验。

两个分支互补，一个偏数据驱动，一个偏神经生理先验。

### 9.4 生物标志物显式分支

诊断模型中的 `DepressionBiomarkerExtractor`、情绪模型中的 `BiologicalMarkerExtractor` 会显式抽取 EEG 生物标志物，例如：

| 特征类别 | 说明 |
|----------|------|
| 频带统计 | theta、alpha、beta、gamma 等频带能量或 DE 表征 |
| 半球不对称 | 左右半球频带差异，尤其与抑郁研究相关的 alpha 不对称 |
| Hjorth 参数 | Activity、Mobility、Complexity |
| PLV 图指标 | 节点强度、连接模式、半球间/半球内连接 |
| 非线性复杂度 | 分形维数、排列熵、线长、过零率等 |

这使模型不仅依赖黑盒深度特征，也显式吸收传统 EEG 分析中的有效特征。

### 9.5 监督对比学习

HC/DEP 情绪模型引入 `contrast_feat` 和监督对比损失。它的目标是让同一情绪类别的窗口特征更接近，不同情绪类别的窗口特征更远，从而提升特征空间的可分性。

### 9.6 被试域对抗

情绪模型通过 subject GRL head 让 backbone 学到更少的被试身份信息，降低模型对个体差异的过拟合。当前权重较小，属于温和正则化：

```text
lambda_subject = 0.01
grl_subject = 0.01
```

### 9.7 严格防止跨被试泄露

项目在训练、验证和推理上都围绕被试级划分设计：

1. `StratifiedGroupKFold` 保证被试不跨 train/val。
2. checkpoint 保存 `train_subjects` 和 `val_subjects`。
3. 验证推理时优先读取 checkpoint 中的验证被试，而不是重新随机划分。
4. `infer_val_fold.py` 会检查评估被试是否出现在模型训练被试中。

## 10. 项目文件结构

```text
stage/
├── config.py
├── dataloader.py
├── run_pipeline.py
├── TECHNICAL_REPORT.md
│
├── models/
│   ├── diagnosis_model.py
│   ├── hc_contrast_bio.py
│   └── dep_contrast_bio.py
│
├── trainers/
│   ├── train_diag.py
│   ├── train_hc.py
│   └── train_dep.py
│
├── inference/
│   ├── infer_val_fold.py
│   ├── infer_test_fold.py
│   └── ensemble_test.py
│
├── scripts/
│   └── combine_fold_models.py
│
├── utils/
│   ├── checkpoint.py
│   ├── data.py
│   ├── folds.py
│   ├── metrics.py
│   ├── predict.py
│   └── training.py
│
├── com_index_sub_10s.csv
├── com_index_sub_2s.csv
├── com_test_trial_index_10s.csv
├── com_test_trial_index_2s.csv
│
├── data/
├── testdata/
├── model_params/
├── predictions/
└── results/
```

## 11. 推荐运行顺序

完整训练和推理建议按以下顺序：

```bash
# 1. 训练诊断模型
python trainers/train_diag.py

# 2. 训练 HC 情绪模型
python trainers/train_hc.py

# 3. 训练 DEP 情绪模型
python trainers/train_dep.py

# 4. 查看哪些 fold 已经三模型齐全
python run_pipeline.py status

# 5. 生成 combined_meta、跑验证、跑测试并集成
python run_pipeline.py all
```

如果只想调试某一个 fold：

```bash
python scripts/combine_fold_models.py --repeat 0 --fold 0
python inference/infer_val_fold.py --repeat 0 --fold 0
python inference/infer_test_fold.py --repeat 0 --fold 0
python inference/ensemble_test.py
```

## 12. 现阶段需要注意的问题

1. `diagnosis_label` 的编码方向需要统一。当前 CSV 实际是 `DEP=1, HC=0`，但部分注释和部分情绪验证分组逻辑写成 `0=DEP, 1=HC`。建议后续统一以 `label4` 推导诊断组别，或明确全项目 `diagnosis_label` 的含义并同步修改注释和评估分组。
2. `utils/folds.py` 中函数名 `reapt_name()` 拼写保留了历史写法，目录名也是 `reapt`。这不影响运行，但报告和命令中要保持一致。
3. 当前诊断模型的图正则、域对抗、中心对比损失权重为 0，属于预留能力；真正生效的是诊断交叉熵和模型结构本身。
4. 当前 `model_params` 目录中只看到 `repeat0_fold0` 旧式目录时，`run_pipeline.py status` 可能只能发现已经齐全的历史 checkpoint。新训练脚本默认输出的是 `diag_reapt*`、`hc_reapt*`、`dep_reapt*` 三类目录。
5. 由于模型较大，完整 `3 seed × 5 fold × 3 task` 训练成本较高。实际实验时可以先跑单个 `repeat/fold` 验证流程，再扩展到完整集成。

## 13. 总结

本项目的技术路线可以总结为：用跨被试五折验证保证泛化评估，用多尺度 EEG 编码、双图脑网络和生物标志物分支增强模型表达能力，用 HC/DEP 分组情绪模型处理不同诊断群体的情绪差异，最后用诊断模型输出的被试级概率对两个情绪模型做软路由融合。

从工程角度看，项目已经具备完整闭环：

```text
数据索引
  -> 训练三类模型
  -> 保存 checkpoint 和 meta
  -> 组合 fold 模型
  -> 验证集两阶段评估
  -> 测试集两阶段预测
  -> 多 fold 概率集成
  -> 输出 submission.xlsx
```

这使它不仅是一个模型文件集合，而是一套可以复现实验、可以验证防泄露、可以产出测试提交结果的完整 EEG 深度学习系统。
