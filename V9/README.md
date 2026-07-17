# V9 BF-GCN EEG 情绪识别

## 目标与版本边界

V9 在不修改 V2/V5/V6/V7/V8 的前提下，新增独立的 BF-GCN 二分类流程：`0=neutral`、`1=positive`。诊断标签遵循当前索引文件的真实编码：`0=HC`、`1=DEP`，只用于被试级分层、分人群评估与统计，不进入模型损失。

V9 第一版只验证 BF-GCN，不包含诊断软路由、HC/DEP 专家、SSAS、GRL、MMD、SupCon、四分类联合头或时间注意力。`utils/trial_aggregation.py` 只做评估/提交所需的概率平均，不是时间聚合网络。

## 已确认的现有数据

项目根目录下实际使用：

- 训练窗口索引：`com_index_sub_2s.csv`
- 测试窗口索引：`com_test_window_index_2s.csv`
- 清洗后训练 trial：`data/com_split_data_subject_2s/*.npy`
- 训练 DE：`data/com_de_features_2s/*_de.npy`
- 清洗后测试 trial：`data/com_test_split_trial_2s/*.npy`
- 测试 DE：`data/com_test_de_features_2s/*_de.npy`

训练集为 60 名被试（40 HC、20 DEP），每人 8 个原始 50 秒 trial。每个原始 trial 是 `[30,12500]`，对应 DE `[49,30,5]`。V9 将每个原始 trial 映射为 5 个连续、不重叠的 10 秒 pseudo-trial；每段按 2 秒窗、1 秒步长保留 9 个内部窗口，排除跨越 10 秒边界的全局窗口 9/19/29/39。因此训练集共 2,400 个 10 秒 trial、21,600 个窗口。

测试集有 10 名被试、每人 8 个 10 秒 trial。每个 trial 是 `[30,2500]`，DE 为 `[9,30,5]`，共 80 个 trial、720 个窗口。

采样率 250 Hz，窗口长度 500 点（2 秒），步长 250 点（1 秒）。频段从现有 `preprcocess/preprocess_test.py` 核对后集中定义在 `configs/bfgcn_config.py`：Delta 1–4、Theta 4–8、Alpha 8–13、Beta 13–30、Gamma 30–45 Hz。

现有文件没有 PLV。V9 不重新处理原始 `.mat` EEG，而是复用清洗后 `.npy` trial，仅对当前 2 秒窗口补算缺失的五频段 PLV，并默认缓存在 `V9/cache/plv/`。每个样本为：

```text
de                 [30, 5]
plv                [30, 30, 5]
emotion_label      scalar
diagnosis_label    scalar
subject_id         str
trial_id           scalar（10 秒 trial 的被试内唯一编号）
original_trial_id  scalar
pseudo_trial_id    scalar（训练为 1..5，测试为 0）
window_id          scalar（0..8）
```

## 模型结构与张量

输入 DE `[B,30,5]` 同时进入：

1. learnable-specific：全局可学习图上的图卷积；
2. functional-specific：五频段 PLV 注意力融合图上的图卷积；
3. common：同一组图卷积参数分别处理两张图，再做平均。

可学习图参数为 `[30,30]`，前向时对称化、softplus 非负化、加自环并做对称度归一化。PLV 注意力是归一化的 `[B,5]`，融合功能图为 `[B,30,30]`，同样执行对称化、非负约束、自环与图归一化。

图层默认构造 `I/A/A²` 并学习各阶结果的组合。它是多阶邻接传播，不是严格意义上的 Chebyshev Laplacian 递推；配置名为 `graph_conv_type="multi_order"`，为以后增加真正的 `chebyshev` 实现保留接口。

分支输出堆叠为 `[B,R,30,H]`，`R` 是实际启用分支数而非固定 3。分支注意力为 `[B,R,30]`，融合节点表示为 `[B,30,H]`。图级读出明确使用节点维 global mean pooling 与 global max pooling，拼接为 `[B,2H]`；默认 `H=64`，图特征为 `[B,128]`。分类头为 Linear → LayerNorm → GELU → Dropout → Linear，输出 `[B,2]`。

模型字典持续返回 `emotion_logits`、`graph_feature`、`node_feature`、`learnable_adj`、`functional_adj`、`band_attention` 和 `branch_attention`。关闭的图对应值为 `None`。`extract_interpretability_outputs(...)` 可将图、注意力和概率保存为 `.npz`。

## 数据检查与 smoke test

正式训练会自动先执行真实数据 smoke test：一个 batch 的前向、CE loss、backward、optimizer.step 和 trial 概率聚合；任一形状不符都会停止训练。

```bash
python V9/inspect_bfgcn_data.py --max-plv-samples 0
python V9/inspect_bfgcn_data.py --index com_test_window_index_2s.csv --max-plv-samples 0
python V9/train_bfgcn_cv.py --smoke-only --amp 0
```

`--max-plv-samples 0` 检查全部 PLV；首次会计算缓存，耗时明显，调试时可改为 16。检查覆盖 DE/PLV 维度、NaN/Inf、PLV 对称性和对角线、DE 极值、元数据对应关系、trial 窗口数、每人情绪样本及可选 train/val 被试交集。严重问题会抛出异常。

## 训练和十折验证

不传 `--fold` 时运行完整十折；传入的是 1-based 折号。划分先生成每名被试一行的诊断表，再优先尝试 `StratifiedGroupKFold(group=subject_id, stratify=diagnosis_label)`。代码会核对实际折分；sklearn 当前对本数据会产生少数 3/3 或 5/1 折，因此自动切换到确定性后备方案：HC、DEP 各自按统一 seed 打乱并分成十份。最终每折验证集严格为 4 HC + 2 DEP，同一被试的所有原始 trial、pseudo-trial 和窗口只会出现在训练或验证一侧。

```bash
# 完整十折（默认 100 epoch）
python V9/train_bfgcn_cv.py --experiment full_bfgcn

# 单折调试
python V9/train_bfgcn_cv.py --experiment full_bfgcn --fold 1 --epochs 2 --batch-size 16
```

默认参数：batch 64、epoch 100、AdamW lr `1e-3`、weight decay `5e-4`、dropout 0.3、label smoothing 0.05、cosine scheduler、梯度裁剪 5.0、patience 15、CUDA 时 AMP。Windows 默认 `num_workers=0`；有 CUDA 自动用 GPU，无 CUDA 可在 CPU 完成前向和训练。默认不启用 class weight；`--auto-class-weight` 只从当前训练折计算。

每 epoch 输出训练 loss/窗口 accuracy/macro-F1，以及验证窗口和 trial 指标、HC/DEP trial accuracy/macro-F1 与学习率。每折固定按验证 `trial_macro_f1` 选优，相同时比较 `trial_accuracy`。每折仅保存一个 state-dict checkpoint：

```text
V9/checkpoints/fold_01/best_model.pt
...
V9/checkpoints/fold_10/best_model.pt
```

验证结果包含总体、HC、DEP 的 accuracy、macro-F1、neutral recall、positive recall 和 confusion matrix：

```text
V9/results/fold_01_metrics.json
V9/results/fold_01_predictions.csv
V9/results/cv_summary.csv
V9/results/cv_summary.json
```

非 `full_bfgcn` 实验写入以实验名命名的子目录，避免覆盖完整模型。

## 消融命令

```bash
# A: DE + learnable graph
python V9/train_bfgcn_cv.py --experiment ablation_a --use-learnable-graph 1 --use-functional-graph 0 --use-common-branch 0

# B: DE + PLV functional graph
python V9/train_bfgcn_cv.py --experiment ablation_b --use-learnable-graph 0 --use-functional-graph 1 --use-common-branch 0

# C: two specific branches, no common branch
python V9/train_bfgcn_cv.py --experiment ablation_c --use-learnable-graph 1 --use-functional-graph 1 --use-common-branch 0

# D: full BF-GCN
python V9/train_bfgcn_cv.py --experiment full_bfgcn --use-learnable-graph 1 --use-functional-graph 1 --use-common-branch 1
```

频段和分支注意力可通过 `--use-band-attention 0`、`--use-branch-attention 0` 关闭；关闭后使用均匀权重。

## 测试预测

```bash
python V9/predict_bfgcn_test.py --experiment full_bfgcn
```

每个 fold 先对同一 trial 的 9 个窗口 softmax 概率求平均，再对十个 fold 的 trial 概率求平均，最后 argmax。排序采用自然数顺序，避免 `P_test1,P_test10,P_test2`。输出：

```text
V9/outputs/bfgcn_test_probabilities.csv
V9/outputs/bfgcn_submission.xlsx
```

CSV 包含 `user_id,trial_id,Emotion_label,prob_neutral,prob_positive`；Excel 严格只保留 `user_id,trial_id,Emotion_label`。增加 `--save-interpretability` 会保存首折首 batch 的 `.npz`。

## 常见错误排查

- `DE shape`：应读取 `[W,30,5]` 后按 `de_index` 取 `[30,5]`，不能直接 reshape。
- `PLV shape`：轴顺序必须是 `[30,30,5]`，batch 后为 `[B,30,30,5]`。
- trial 变成 49 窗：说明没有拆 50 秒训练 trial；每个 pseudo-trial 应为 9 窗并排除跨边界窗。
- 被试泄漏：划分单位必须是 `subject_id`，不能按窗口随机切分。
- CUDA/worker 问题：使用 `--amp 0 --num-workers 0` 做 CPU/Windows 调试；模型内部没有 `.cuda()`。
- 首轮较慢：缺失 PLV 正从清洗后 trial 计算并缓存，后续 epoch 直接加载。

## 相对原 BF-GCN demo 的明确修正

- 删除硬编码 CUDA，device/dtype 来自输入或模型参数；
- PLV 频段数从配置/输入维度动态适配，不写死为 4；
- 可学习邻接矩阵在前向时显式对称化；
- 频段注意力和分支注意力使用不同变量并分别返回；
- 使用定义清楚的节点维 mean-max 图级 pooling；
- Dropout 模块真实参与图分支和分类头前向；
- trial 评估与比赛提交统一为“trial 内窗口概率平均”；
- 50 秒训练 trial 的五个 10 秒片段具有独立聚合键，绝不会被错误合回 50 秒。
