# 决赛多版本模型推理与投票框架

该目录提供统一的决赛推理入口，当前支持：

- V7 Experiment A：`stage2_best.pt`，支持 `adaptive_threshold_best.pt`；
- V8 Experiment A：`stage2_best.pt`，支持自适应阈值与 PLV 缓存；
- V9 BF-GCN：`best_model.pt`。

每个版本可以选择任意 fold、单独设置 checkpoint 权重和版本权重。框架先在版本内部对所选 checkpoint 加权平均，再在版本之间进行软投票或硬投票。

## 直接使用工程内现有模型

查看已配置模型：

```bash
python finals_prediction/run.py --list-models
```

只验证配置和文件，不加载模型：

```bash
python finals_prediction/run.py --dry-run
```

只运行某一个版本：

```bash
python finals_prediction/run.py --models v7_non_res
python finals_prediction/run.py --models v8_experiment_a
python finals_prediction/run.py --models v9_bfgcn
```

选择多个版本进行投票：

```bash
python finals_prediction/run.py --models v7_non_res,v8_experiment_a,v9_bfgcn
```

默认配置是 `configs/current_models.json`，默认决赛索引是项目根目录的 `com_juesai_window_index_2s.csv`。

## 使用 model_zoo 放置模型

推荐目录结构：

```text
finals_prediction/model_zoo/
  my_v7/
    fold0/
      stage2_best.pt
      adaptive_threshold_best.pt
    fold1/
      ...
  my_v8/
    fold0/
      stage2_best.pt
      adaptive_threshold_best.pt
  my_v9/
    fold_01/
      best_model.pt
    fold_02/
      ...
```

复制并修改 `configs/model_zoo_example.json`，然后运行：

```bash
python finals_prediction/run.py \
  --config finals_prediction/configs/model_zoo_example.json \
  --models my_v7,my_v9
```

配置中的相对路径全部以工程根目录为基准。

## 选择 fold 和权重

在模型配置中加入：

```json
{
  "folds": [0, 1, 4, 7],
  "checkpoint_weights": {
    "0": 1.2,
    "1": 1.0,
    "4": 1.3,
    "7": 0.8
  },
  "weight": 1.5
}
```

- `folds`：只使用指定的最优 fold checkpoint；
- `checkpoint_weights`：版本内部各 fold 权重；
- `weight`：该版本参与最终投票的权重；
- V7/V8 的 `probability` 可选 `adaptive` 或 `raw`；
- V9 使用 `raw` softmax 概率。

如果只想指定几个确切的最优参数文件，可以不用 `checkpoint_glob`，直接写：

```json
"checkpoints": [
  "finals_prediction/model_zoo/my_v8/fold0/stage2_best.pt",
  "finals_prediction/model_zoo/my_v8/fold4/stage2_best.pt"
]
```

软投票：

```json
"vote": {"method": "soft", "threshold": 0.5}
```

硬投票：

```json
"vote": {"method": "hard", "threshold": 0.5}
```

## 输出文件

默认写入 `finals_prediction/outputs/current_vote/`：

- `checkpoint_predictions.csv`：每个 checkpoint 的概率；
- `model_probabilities.csv`：每个版本内部集成概率；
- `final_vote_probabilities.csv`：所有版本概率和最终投票；
- `submission.csv`、`submission.xlsx`：最终提交文件；
- `run_manifest.json`：本次使用的模型、参数路径、设备和结果审计信息。

## 注意事项

- V7/V8 checkpoint 内的 `domain_mapping` 必须包含决赛用户 ID；框架会在推理前检查。
- `stage2_best.pt` 已经是每折训练阶段选出的最优参数；是否使用全部折或部分折由 `folds` 决定。
- 初次运行 V8/V9 会生成 PLV 缓存，之后运行会明显加快。
- 如果配置 CUDA 但 `torch.cuda.is_available()` 为 `False`，框架会直接报错，不会静默假装使用 GPU。
