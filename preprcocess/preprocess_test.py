import os
import re
import glob
import json
import math
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat, savemat

import mne
from autoreject import Ransac
from mne.preprocessing import ICA
from mne_icalabel import label_components


# ============================================================
# 基本参数
# ============================================================

SFREQ = 250

# 测试集：每个被试 8 个 trial
N_TEST_TRIALS = 8

# 测试集每个 trial: 2500 = 250 Hz * 10 s
TEST_TRIAL_LEN = 2500

# 和你训练集保持一致：2 秒窗口，1 秒步长
WIN_LEN = 2500
STEP = 2500

DE_BANDS = [
    (1.0, 4.0),     # delta
    (4.0, 8.0),     # theta
    (8.0, 13.0),    # alpha
    (13.0, 30.0),   # beta
    (30.0, 45.0),   # gamma
]


# ============================================================
# 读取 mat：兼容普通 mat 和 v7.3 hdf5 mat
# ============================================================

def load_mat_auto(mat_path: str):
    """
    自动读取 scipy .mat 和 MATLAB v7.3 .mat。
    """
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        data = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])

                # MATLAB v7.3 有时会变成 [T, C]，这里转成 [C, T]
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T

                data[key] = arr
        return data


def find_test_eeg_array(mat_dict):
    """
    从测试集 mat 文件中找到 EEG 数据。
    优先读取 test_eeg_c。
    如果没有该 key，则自动寻找形状中包含 30 通道的二维数组。
    """
    # 优先使用官方测试集变量名
    if "test_eeg_c" in mat_dict:
        arr = np.asarray(mat_dict["test_eeg_c"])

        if arr.ndim != 2:
            raise ValueError(f"test_eeg_c 应该是二维数组，但得到 shape={arr.shape}")

        # 保证输出为 [30, T]
        if arr.shape[0] == 30:
            return arr.astype(np.float32)
        elif arr.shape[1] == 30:
            return arr.T.astype(np.float32)
        else:
            raise ValueError(f"test_eeg_c 形状异常，应包含 30 通道，但得到 shape={arr.shape}")

    # 如果变量名不是 test_eeg_c，则自动寻找
    candidates = []
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue

        arr = np.asarray(value)
        if arr.ndim == 2 and (arr.shape[0] == 30 or arr.shape[1] == 30):
            candidates.append((key, arr))

    if len(candidates) == 0:
        raise ValueError("没有在 mat 文件中找到形状包含 30 通道的 EEG 二维数组")

    if len(candidates) > 1:
        print("发现多个可能的 EEG 变量：", [x[0] for x in candidates])
        print("默认使用第一个：", candidates[0][0])

    key, arr = candidates[0]

    if arr.shape[0] == 30:
        return arr.astype(np.float32)
    else:
        return arr.T.astype(np.float32)


# ============================================================
# 通道名
# ============================================================

def get_default_channel_names():
    """
    根据数据集说明中的 30 通道顺序。
    """
    return [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
        "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
        "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"
    ]


def standardize_channel_names(ch_names):
    """
    将比赛数据中的通道名标准化为 MNE standard_1020 可识别的名字。
    """
    name_map = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "OZ": "Oz",
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }

    return [name_map.get(ch, ch) for ch in ch_names]


def load_channel_names(ch_name_path=None):
    """
    优先从 ch_name.mat 读取通道名。
    如果路径不存在，则使用数据集说明中的默认 30 通道顺序。
    """
    if ch_name_path is not None and os.path.exists(ch_name_path):
        name = loadmat(ch_name_path)
        ch_names = name["labels"].reshape(-1).tolist()
        ch_names = [arr.item() for arr in ch_names]
    else:
        print("未找到 ch_name.mat，使用数据集说明中的默认 30 通道顺序")
        ch_names = get_default_channel_names()

    return standardize_channel_names(ch_names)


# ============================================================
# 标准化
# ============================================================

def subject_wise_zscore(clean_all, eps=1e-6):
    """
    对一个测试被试的完整 EEG 数据做 subject-wise + channel-wise z-score。

    clean_all: [C, T]

    return:
        clean_all_norm: [C, T]
        mean: [C, 1]
        std: [C, 1]
    """
    mean = clean_all.mean(axis=1, keepdims=True)
    std = clean_all.std(axis=1, keepdims=True)

    std = np.where(std < eps, eps, std)

    clean_all_norm = (clean_all - mean) / std

    return clean_all_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


# ============================================================
# 测试集单被试预处理
# ============================================================

def preprocess_test_subject(
    test_data,
    sfreq=250,
    ch_name_path=None,
    ransac_duration=2.0,
):
    """
    对一个测试被试的完整 EEG 数据统一预处理。

    输入:
        test_data: [30, 20000]

    流程:
        1. 创建 MNE Raw
        2. 主数据滤波 0.1–45 Hz
        3. ICA 拟合数据滤波 1–45 Hz
        4. RANSAC 检测坏道
        5. 坏道插值
        6. ICA + ICLabel 去伪迹
        7. subject-wise z-score

    输出:
        clean_norm: [30, 20000]
        info_dict
    """
    test_data = np.asarray(test_data, dtype=np.float64)

    assert test_data.ndim == 2, f"test_data 应为 [C, T]，但得到 {test_data.shape}"
    assert test_data.shape[0] == 30, f"测试集应为 30 通道，但得到 {test_data.shape}"

    ch_names_std = load_channel_names(ch_name_path)

    if len(ch_names_std) != test_data.shape[0]:
        raise ValueError(
            f"通道名数量和数据通道数不一致: len(ch_names_std)={len(ch_names_std)}, "
            f"data channels={test_data.shape[0]}"
        )

    info = mne.create_info(
        ch_names=ch_names_std,
        sfreq=sfreq,
        ch_types="eeg"
    )

    montage = mne.channels.make_standard_montage("standard_1020")

    raw = mne.io.RawArray(test_data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")

    # 主数据：用于最终输出
    raw_main = raw.copy().filter(
        l_freq=0.1,
        h_freq=45,
        verbose=False
    )

    # ICA 数据：用于拟合 ICA
    raw_ica = raw.copy().filter(
        l_freq=1.0,
        h_freq=45,
        verbose=False
    )

    # RANSAC 坏道检测
    epochs = mne.make_fixed_length_epochs(
        raw_main,
        duration=ransac_duration,
        overlap=0.0,
        preload=True,
        verbose=False
    )

    ransac = Ransac(
        n_resample=100,
        min_channels=0.5,
        min_corr=0.75,
        verbose=False
    )

    try:
        ransac.fit(epochs)
        bads = ransac.bad_chs_
    except Exception as e:
        print(f"RANSAC 失败，跳过坏道检测。错误信息: {repr(e)}")
        bads = []

    print(f"RANSAC bad channels: {bads}")

    raw_main.info["bads"] = bads
    raw_ica.info["bads"] = bads

    if len(bads) > 0:
        raw_main.interpolate_bads(reset_bads=True, verbose=False)
        raw_ica.interpolate_bads(reset_bads=True, verbose=False)

    # ICA 去伪迹
    ica = ICA(
        n_components=None,
        random_state=97,
        method="infomax",
        verbose=False
    )

    try:
        ica.fit(raw_ica, verbose=False)

        ic_labels = label_components(
            raw_ica,
            ica,
            method="iclabel"
        )

        labels = ic_labels["labels"]
        print("IC labels:", labels)

        exclude_idx = [
            idx for idx, label in enumerate(labels)
            if label not in ["brain", "other"]
        ]

        print(f"Excluding ICA components: {exclude_idx}")

        ica.exclude = exclude_idx

        raw_clean = ica.apply(raw_main.copy(), verbose=False)
        clean_all = raw_clean.get_data()

    except Exception as e:
        print(f"ICA 或 ICLabel 失败，使用滤波+坏道插值后的数据。错误信息: {repr(e)}")
        labels = []
        exclude_idx = []
        clean_all = raw_main.get_data()

    # subject-wise 标准化
    clean_norm, subj_mean, subj_std = subject_wise_zscore(clean_all)

    info_dict = {
        "bad_channels": list(bads),
        "ica_labels": list(labels),
        "ica_exclude_idx": list(map(int, exclude_idx)),
        "subject_mean_shape": list(subj_mean.shape),
        "subject_std_shape": list(subj_std.shape),
    }

    return clean_norm, subj_mean, subj_std, info_dict


# ============================================================
# DE 特征
# ============================================================

def compute_de_one_window(x, sfreq=250, bands=None, eps=1e-6):
    """
    对一个 EEG 窗口计算 5 频段 DE 特征。

    输入:
        x: [C, T]

    输出:
        de_feat: [C, 5]
    """
    if bands is None:
        bands = DE_BANDS

    x = np.asarray(x, dtype=np.float32)
    C, T = x.shape

    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(x, axis=-1)

    power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(T, 1)

    de_list = []

    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            band_var = np.zeros((C,), dtype=np.float32)
        else:
            band_var = power[:, mask].mean(axis=-1)

        de = 0.5 * np.log(2.0 * math.pi * math.e * band_var + eps)
        de_list.append(de)

    de_feat = np.stack(de_list, axis=-1)

    return de_feat.astype(np.float32)


def smooth_de_sequence(de_seq, smooth_kernel=3):
    """
    对 trial 内 DE 序列做滑动平均。

    输入:
        de_seq: [W, C, 5]

    输出:
        smooth_de_seq: [W, C, 5]
    """
    de_seq = np.asarray(de_seq, dtype=np.float32)

    if smooth_kernel is None or smooth_kernel <= 1:
        return de_seq

    W, C, K = de_seq.shape
    pad = smooth_kernel // 2

    padded = np.pad(
        de_seq,
        pad_width=((pad, pad), (0, 0), (0, 0)),
        mode="edge"
    )

    smooth = np.zeros_like(de_seq, dtype=np.float32)

    for i in range(W):
        smooth[i] = padded[i:i + smooth_kernel].mean(axis=0)

    return smooth.astype(np.float32)


def extract_trial_de_sequence(
    trial,
    sfreq=250,
    win_len=500,
    step=250,
    smooth_kernel=3,
):
    """
    对一个测试 trial 提取 DE 序列。

    输入:
        trial: [30, 2500]

    输出:
        de_seq: [W, 30, 5]

    对测试集来说:
        trial 长度 2500
        win_len 500
        step 250
        所以 W = 9
    """
    trial = np.asarray(trial, dtype=np.float32)
    C, T = trial.shape

    de_list = []

    for start in range(0, T - win_len + 1, step):
        end = start + win_len
        x_win = trial[:, start:end]

        de_feat = compute_de_one_window(
            x_win,
            sfreq=sfreq,
            bands=DE_BANDS
        )

        de_list.append(de_feat)

    if len(de_list) == 0:
        raise ValueError(f"trial 太短，无法提取 DE: trial shape={trial.shape}")

    de_seq = np.stack(de_list, axis=0)

    de_seq = smooth_de_sequence(
        de_seq,
        smooth_kernel=smooth_kernel
    )

    return de_seq.astype(np.float32)


# ============================================================
# 文件名解析
# ============================================================

def parse_test_user_id(mat_path):
    """
    P_test1.mat -> P_test1
    P_test10.mat -> P_test10
    """
    base = os.path.basename(mat_path)
    user_id = os.path.splitext(base)[0]
    return user_id


def parse_test_subject_number(user_id):
    """
    P_test1 -> 1
    P_test10 -> 10
    如果无法解析，则返回 -1。
    """
    m = re.search(r"(\d+)$", user_id)
    if m is None:
        return -1
    return int(m.group(1))


# ============================================================
# 构建测试集 trial 和 window 索引
# ============================================================

def build_test_index(
    test_mat_root,
    save_clean_root,
    save_trial_root,
    save_de_root,
    out_window_csv,
    out_trial_csv,
    ch_name_path=None,
    smooth_kernel=3,
):
    """
    处理整个测试集。

    输出:
        1. clean mat:
            save_clean_root/P_test1_clean.mat

        2. trial npy:
            save_trial_root/P_test1_trial01.npy
            ...
            save_trial_root/P_test1_trial08.npy

        3. DE npy:
            save_de_root/P_test1_trial01_de.npy
            ...
            save_de_root/P_test1_trial08_de.npy

        4. window 级 CSV:
            每一行对应一个 2 秒窗口，用于模型逐窗口预测。

        5. trial 级 CSV:
            每一行对应一个 trial，用于最后聚合提交。
    """
    os.makedirs(save_clean_root, exist_ok=True)
    os.makedirs(save_trial_root, exist_ok=True)
    os.makedirs(save_de_root, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(test_mat_root, "*.mat")))

    if len(mat_files) == 0:
        raise FileNotFoundError(f"没有在 {test_mat_root} 找到 .mat 文件")

    print(f"找到测试集 mat 文件数量: {len(mat_files)}")
    for p in mat_files:
        print("  ", p)

    window_records = []
    trial_records = []
    preprocess_infos = {}

    for mat_path in mat_files:
        print("\n" + "=" * 80)
        print(f"Processing test file: {mat_path}")

        user_id = parse_test_user_id(mat_path)
        subject_number = parse_test_subject_number(user_id)

        mat = load_mat_auto(mat_path)
        eeg = find_test_eeg_array(mat)

        print(f"user_id: {user_id}")
        print(f"raw eeg shape: {eeg.shape}")

        if eeg.shape[0] != 30:
            raise ValueError(f"{user_id} 通道数异常，应为 30，但得到 {eeg.shape}")

        expected_len = N_TEST_TRIALS * TEST_TRIAL_LEN

        if eeg.shape[1] != expected_len:
            raise ValueError(
                f"{user_id} 采样点数异常。期望 {expected_len}，但得到 {eeg.shape[1]}。"
                f"如果你的测试集 trial 长度不是 2500，请修改 TEST_TRIAL_LEN。"
            )

        # 1. 对整个测试被试统一预处理
        clean_eeg, subj_mean, subj_std, info_dict = preprocess_test_subject(
            test_data=eeg,
            sfreq=SFREQ,
            ch_name_path=ch_name_path,
        )

        print(f"clean eeg shape: {clean_eeg.shape}")

        preprocess_infos[user_id] = info_dict

        # 2. 保存清洗后的完整被试数据
        clean_save_path = os.path.join(save_clean_root, f"{user_id}_clean.mat")
        savemat(
            clean_save_path,
            {
                "test_eeg_c_clean": clean_eeg,
                "subject_mean": subj_mean,
                "subject_std": subj_std,
            }
        )

        print(f"Saved clean mat: {clean_save_path}")

        # 3. 按 8 个视频切成 8 个 trial
        for i in range(N_TEST_TRIALS):
            trial_id = i + 1

            s = i * TEST_TRIAL_LEN
            e = (i + 1) * TEST_TRIAL_LEN

            trial = clean_eeg[:, s:e]

            if trial.shape != (30, TEST_TRIAL_LEN):
                raise ValueError(
                    f"{user_id} trial {trial_id} shape 异常: {trial.shape}"
                )

            trial_name = f"{user_id}_trial{trial_id:02d}.npy"
            trial_path = os.path.join(save_trial_root, trial_name)

            np.save(trial_path, trial.astype(np.float32))

            # 4. 提取并保存该 trial 的 DE 序列
            de_seq = extract_trial_de_sequence(
                trial,
                sfreq=SFREQ,
                win_len=WIN_LEN,
                step=STEP,
                smooth_kernel=smooth_kernel,
            )

            de_name = f"{user_id}_trial{trial_id:02d}_de.npy"
            de_path = os.path.join(save_de_root, de_name)

            np.save(de_path, de_seq.astype(np.float32))

            starts = list(range(0, TEST_TRIAL_LEN - WIN_LEN + 1, STEP))

            if len(starts) != de_seq.shape[0]:
                raise ValueError(
                    f"{user_id} trial {trial_id}: window 数和 DE 数不一致，"
                    f"windows={len(starts)}, de_seq={de_seq.shape[0]}"
                )

            trial_records.append({
                "user_id": user_id,
                "subject_number": subject_number,
                "file_name": os.path.basename(mat_path),
                "trial_id": trial_id,
                "trial_path": trial_path,
                "de_path": de_path,
                "trial_start": s,
                "trial_end": e,
                "n_windows": len(starts),
            })

            # 5. window 级索引
            for win_id, start in enumerate(starts):
                end = start + WIN_LEN

                window_records.append({
                    "user_id": user_id,
                    "subject_number": subject_number,
                    "file_name": os.path.basename(mat_path),

                    # 测试集没有真实标签，全部用占位符
                    "diagnosis": "TEST",
                    "diagnosis_label": -1,
                    "emotion": "unknown",
                    "emotion_label": -1,
                    "label4": -1,

                    "trial_id": trial_id,
                    "trial_path": trial_path,
                    "start": start,
                    "end": end,

                    "de_path": de_path,
                    "de_win_id": win_id,
                })

            print(
                f"Saved trial {trial_id}: {trial_path}, "
                f"trial shape={trial.shape}, DE shape={de_seq.shape}"
            )

    # 6. 保存 CSV
    df_window = pd.DataFrame(window_records)
    df_trial = pd.DataFrame(trial_records)

    df_window.to_csv(out_window_csv, index=False, encoding="utf-8-sig")
    df_trial.to_csv(out_trial_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"Window index saved to: {out_window_csv}")
    print(f"Trial index saved to:  {out_trial_csv}")

    print("\nwindow index head:")
    print(df_window.head())

    print("\ntrial index head:")
    print(df_trial.head())

    print("\n统计信息:")
    print("测试被试数:", df_trial["user_id"].nunique())
    print("trial 总数:", len(df_trial))
    print("window 总数:", len(df_window))
    print("每个 trial 的窗口数:")
    print(df_trial["n_windows"].value_counts().sort_index())

    # 7. 保存预处理信息
    info_json_path = os.path.join(save_clean_root, "test_preprocess_info.json")
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_infos, f, ensure_ascii=False, indent=2)

    print(f"\nPreprocess info saved to: {info_json_path}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    # 修改成你的测试集 mat 文件夹
    # 里面应该是:
    #   P_test1.mat
    #   P_test2.mat
    #   ...
    #   P_test10.mat
    TEST_MAT_ROOT = "../testdata"

    # 如果你有 ch_name.mat，就填真实路径
    # 如果没有，程序会自动使用数据集说明中的 30 通道顺序
    CH_NAME_PATH = "../ch_name.mat"

    SAVE_CLEAN_ROOT = "../data/com_test_clean_10s"
    SAVE_TRIAL_ROOT = "../data/com_test_split_trial_10s"
    SAVE_DE_ROOT = "../data/com_test_de_features_10s"

    OUT_WINDOW_CSV = "../data/com_test_window_index_10s.csv"
    OUT_TRIAL_CSV = "../data/com_test_trial_index_10s.csv"

    build_test_index(
        test_mat_root=TEST_MAT_ROOT,
        save_clean_root=SAVE_CLEAN_ROOT,
        save_trial_root=SAVE_TRIAL_ROOT,
        save_de_root=SAVE_DE_ROOT,
        out_window_csv=OUT_WINDOW_CSV,
        out_trial_csv=OUT_TRIAL_CSV,
        ch_name_path=CH_NAME_PATH,
        smooth_kernel=3,
    )