# Model Zoo

把用于决赛推理的 checkpoint 放在此目录。建议每个版本或实验一个子目录、每个 fold 一个子目录。

V7/V8 的每个 fold 通常需要：

```text
stage2_best.pt
adaptive_threshold_best.pt   # probability=adaptive 时必需
```

V9 的每个 fold 需要：

```text
best_model.pt
```

也可以不复制模型，在配置的 `checkpoint_glob` 中直接引用项目现有的 `model_params/` 或 `V9/checkpoints/`。
