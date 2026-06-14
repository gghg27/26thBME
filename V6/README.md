# V6_NoSSAS_DualFeature_SoftRouter

V6 is a clean one-stage version based on the V2 backbone and expert-router idea.
It removes SSAS source selection and source-target domain alignment, then keeps
only supervised training on labeled source windows.

## Core Design

1. Remove SSAS source selection and source-target domain alignment from V2.
2. Keep the backbone dual feature paths:
   - `z_emotion`: emotion-relative feature stream.
   - `z_diag`: diagnosis-absolute feature stream.
3. Use `z_diag` to predict DEP/HC diagnosis probability.
4. Use `z_emotion` for the shared emotion head, HC expert, and DEP expert.
5. Use diagnosis probability to softly route HC/DEP experts.
6. Fuse final emotion probability from shared emotion probability and expert mixture:
   `mix_prob = alpha * prob_shared + (1 - alpha) * expert_mix_prob`.

Diagnosis label convention is unchanged: `0=DEP`, `1=HC`.

## Training Objective

```text
loss = lambda_mix * L_mix
     + lambda_expert * L_expert
     + lambda_shared * L_shared
     + lambda_diag * L_diag
```

Where:

- `L_mix`: NLL on the final fused emotion probability.
- `L_expert`: hard expert emotion loss using the true diagnosis label.
- `L_shared`: shared emotion head cross entropy.
- `L_diag`: diagnosis router cross entropy.

## What Is Disabled

- SSAS source selection.
- Source selection voting and source subject weights.
- MMD loss.
- Subject-domain GRL and subject-domain head.
- Target entropy loss.
- Target loader participation during training.
- Trial SupCon.
- Encoder initialization from a previous stage.

## Commands

Quick check:

```bash
python V6/train_dual_feature_soft_router.py --fold 0 --epochs 2
```

Full folds:

```bash
python V6/train_dual_feature_soft_router.py --all_folds --epochs 100
```

Test prediction:

```bash
python V6/train_dual_feature_soft_router.py --fold 0 --epochs 100 --predict_test
```

Full folds with automatic test ensemble:

```bash
python V6/train_dual_feature_soft_router.py --all_folds --epochs 100 --predict_test
```

Ensemble existing checkpoints without retraining:

```bash
python V6/ensemble_predict.py --model_dir model_params/V6_NoSSAS_DualFeature_SoftRouter
```

Or pass checkpoint paths explicitly:

```bash
python V6/ensemble_predict.py --model_paths path/to/fold0/v6_best.pt path/to/fold1/v6_best.pt
```

The default test voting method is `soft_topk`, matching the expected 4 positive
trials per 8-trial test subject.

## Outputs

Each run saves to:

```text
model_params/V6_NoSSAS_DualFeature_SoftRouter/v6_dual_router_repeat{repeat}_fold{fold}/
```

Important files include:

- `config.json`
- `domain_id_mapping.json`
- `v6_history.csv`
- `v6_best_summary_fold{fold}.csv`
- `v6_best.pt`
- `v6_dual_router_final.pt`
- `submission_v6_dual_router.csv` when `--predict_test` is enabled.

When `--all_folds --predict_test` is used, V6 also averages all fold-level
`test_trial_probs.csv` files and writes ensemble outputs under:

```text
model_params/V6_NoSSAS_DualFeature_SoftRouter/v6_test_ensemble/
```

The final ensemble submission is `submission_v6_ensemble.csv`.

For `V6/ensemble_predict.py`, outputs are written by default to:

```text
model_params/V6_NoSSAS_DualFeature_SoftRouter/v6_checkpoint_ensemble/
```

This version is meant to test whether relative/absolute feature disentanglement
plus diagnosis-guided emotion expert routing is stable without complex domain
adaptation machinery.
