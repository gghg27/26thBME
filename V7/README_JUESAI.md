# V7 决赛数据训练

V7 现在默认使用：

- 有标签源域：`com_index_sub_2s.csv`
- 无标签目标域：`com_juesai_window_index_2s.csv`
- 模型输出：`model_params/V7_juesai`

决赛数据没有真实情绪标签，它只作为 target domain 参与域自适应，不会进入有监督分类损失。

## 1. 单折小规模验证

```bash
python V7/train_experiment_a.py \
  --fold 0 \
  --stage1_epochs 1 \
  --stage2_epochs 1 \
  --batch_size 2 \
  --max_batches 1 \
  --trial_num_windows 9 \
  --predict_test
```

## 2. 完整十折训练

```bash
python V7/train_experiment_a.py \
  --all_folds \
  --all_repeats \
  --trial_num_windows 0 \
  --batch_size 4 \
  --num_workers 0 \
  --predict_test
```

`--trial_num_windows 0` 表示训练 trial 使用全部 49 个窗口；决赛 trial 自然使用其 9 个窗口。如果内存不足，先降低 `--batch_size`。

## 3. 显式指定路径

即使以后修改了默认值，也可以用下面的命令确保使用决赛数据：

```bash
python V7/train_experiment_a.py \
  --all_folds \
  --all_repeats \
  --index_csv com_index_sub_2s.csv \
  --test_csv com_juesai_window_index_2s.csv \
  --save_root model_params/V7_juesai \
  --trial_num_windows 0 \
  --batch_size 4 \
  --predict_test
```

十折完成后会生成：

```text
model_params/V7_juesai/experiment_a_repeat0_fold*/stage2_best.pt
model_params/V7_juesai/test_ensemble/test_ensemble_probs.csv
model_params/V7_juesai/test_ensemble/submission_test_ensemble.csv
```
