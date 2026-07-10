# V2 dual-signal SSAS

This directory is a minimally invasive branch of V2. The cross-subject split,
Stage-1 SSAS voting/source weights, Stage-2 experts and router, MMD, GRL,
ranking loss, checkpoint/early stopping, trial metrics, soft top-k, ensemble,
and optional trial SupCon strategy are retained.

## Data definition

- `x_abs`: filtered, RANSAC/interpolated and ICA/ICLabel-cleaned EEG. It is not
  z-scored per trial or per window.
- `x_rel`: `(x_abs - subject_mean) / max(subject_std, 1e-6)`, with `[30,1]`
  channel statistics computed jointly over all eight trials of that subject.
- `de_abs`: five-band DE recomputed from each `x_abs` window.
- `de_rel`: five-band DE independently recomputed from each `x_rel` window.
  It is never derived by normalizing `de_abs`.

The effective V2 2-second configuration was verified from the current CSV:
`WIN_LEN=500` samples and `STEP=250` samples. Training trials have 49 windows;
test trials have 9 windows. Directory suffixes are not used to infer these
values; training prints the index/DE check before each fold.

## Output layout

```text
data/
  com_split_data_subject_abs_2s/
  com_split_data_subject_rel_2s/
  com_de_features_abs_2s/
  com_de_features_rel_2s/
  com_subject_stats_dual/
  com_index_dual_2s.csv
  com_test_split_data_subject_abs_2s/
  com_test_split_data_subject_rel_2s/
  com_test_de_features_abs_2s/
  com_test_de_features_rel_2s/
  com_test_subject_stats_dual/
  com_test_index_dual_2s.csv
```

The unified CSV contains identity/labels, `trial_path_abs`, `trial_path_rel`,
`de_path_abs`, `de_path_rel`, `start`, `end`, and `de_win_id`. Diagnosis labels
are fixed as `0=DEP`, `1=HC`.

## Commands

Raw training data is expected under `data/com_rawdata` (`DEP/HC*timedata.mat`)
and raw test data under `testdata` (`P_test*.mat`).

```powershell
# Build training absolute/relative signals, DE, stats and unified index
B:\anaconda\envs\pytorch\python.exe preprcocess\dual_signal_preprocess.py train --raw_root data\com_rawdata --ch_name_path ch_name.mat

# Build test data/index; relative statistics use all 8 trials, never labels
B:\anaconda\envs\pytorch\python.exe preprcocess\dual_signal_preprocess.py test --raw_root testdata --ch_name_path ch_name.mat

# Rebuild only a unified index from already generated paired directories
B:\anaconda\envs\pytorch\python.exe preprcocess\build_dual_index.py --metadata_csv com_index_sub_2s.csv --trial_abs_root data\com_split_data_subject_abs_2s --trial_rel_root data\com_split_data_subject_rel_2s --de_abs_root data\com_de_features_abs_2s --de_rel_root data\com_de_features_rel_2s --out_csv data\com_index_dual_2s.csv

# Synthetic forward/backward/optimizer/validation/test smoke test
B:\anaconda\envs\pytorch\python.exe -m V2_dual_signal.smoke_test --with_biomarkers

# Quick single-fold run
B:\anaconda\envs\pytorch\python.exe -m V2_dual_signal.train_expert_ssas_emotion --fold 0 --stage1_epochs 1 --stage2_epochs 1 --batch_size 2 --num_workers 0 --no_test_ensemble

# Full current 10-fold run and test ensemble
B:\anaconda\envs\pytorch\python.exe -m V2_dual_signal.train_expert_ssas_emotion --all_folds --all_repeats --batch_size 200 --test_vote_method soft_topk --k_pos 4
```

## Shared encoder routing

Both stages own exactly one `shared_encoder`. It is called once with
`x_abs/de_abs` and once with `x_rel/de_rel`; no encoder state is copied.
Stage 1 uses `z_rel` for source-domain classification, MMD and emotion GRL,
and `z_abs` for diagnosis GRL. Stage 2 uses `z_abs` for the diagnosis router,
and `z_rel` for the shared emotion head, HC/DEP experts, MMD and subject GRL.
With `0=DEP, 1=HC`, routing remains
`p_dep=diag_prob[:,0:1]`, `p_hc=diag_prob[:,1:2]`.
