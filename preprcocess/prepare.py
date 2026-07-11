from com_preprocess import build_competition_4class_index

build_competition_4class_index(
    mat_root="data/com_clean_domain",
    save_trial_root="data/com_split_data_subject_2s",
    save_de_root="data/com_de_features_2s",
    out_csv="data/com_index_sub_2s.csv",
    smooth_kernel=3,
)