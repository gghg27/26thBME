from com_preprocess import build_competition_4class_index

build_competition_4class_index(
    mat_root="clean_data",
    save_trial_root="data/test_split_data_subject_2s",
    save_de_root="data/test_de_features_2s",
    out_csv="data/test_index_sub_2s.csv",
    smooth_kernel=3,
)