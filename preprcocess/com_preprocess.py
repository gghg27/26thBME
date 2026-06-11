import os
import re
import glob
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
import os
import re
import glob
import h5py
from scipy.io import loadmat,savemat
import mne
from autoreject import Ransac
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne_icalabel import label_components


SFREQ = 250
TRIAL_LEN = 12500
WIN_LEN = 2500
STEP = 2500


def load_mat_auto(mat_path: str):
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        data = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])

                # 保证最终是 [30, 50000]
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T

                data[key] = arr
        return data


def parse_subject_info(file_name: str):
    base = os.path.basename(file_name)
    m = re.match(r"^(DEP|HC)(\d+)timedata\.mat$", base, re.IGNORECASE)
    if m is None:
        raise ValueError(f"文件名格式不符合预期: {base}")
    diagnosis = m.group(1).upper()
    subject_id = int(m.group(2))
    return diagnosis, subject_id


def get_label4(diagnosis: str, emotion: str):
    label_map = {
        ("DEP", "neu"): 0,
        ("DEP", "pos"): 1,
        ("HC",  "neu"): 2,
        ("HC",  "pos"): 3,
    }
    return label_map[(diagnosis, emotion)]

def parse_subject_info_build(file_name: str):
    base = os.path.basename(file_name)
    m = re.match(r"^(DEP|HC)_(\d+)\.mat$", base, re.IGNORECASE)
    if m is None:
        raise ValueError(f"文件名格式不符合预期: {base}")
    diagnosis = m.group(1).upper()
    subject_id = int(m.group(2))
    return diagnosis, subject_id

import math
import numpy as np


DE_BANDS = [
    (1.0, 4.0),     # delta
    (4.0, 8.0),     # theta
    (8.0, 13.0),    # alpha
    (13.0, 30.0),   # beta
    (30.0, 45.0),   # gamma
]


def compute_de_one_window(x, sfreq=250, bands=None, eps=1e-6):
    """
    对一个 EEG 窗口计算 5 频段 DE 特征。

    x: [C, T]
        例如 [30, 1000]

    return:
        de_feat: [C, 5]
    """

    if bands is None:
        bands = DE_BANDS

    x = np.asarray(x, dtype=np.float32)
    C, T = x.shape

    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)  # [F]
    fft_vals = np.fft.rfft(x, axis=-1)         # [C, F]

    power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(T, 1)

    de_list = []

    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            band_var = np.zeros((C,), dtype=np.float32)
        else:
            band_var = power[:, mask].mean(axis=-1)

        # DE = 0.5 * log(2*pi*e*sigma^2)
        de = 0.5 * np.log(2.0 * math.pi * math.e * band_var + eps)
        de_list.append(de)

    de_feat = np.stack(de_list, axis=-1)  # [C, 5]
    return de_feat.astype(np.float32)


def smooth_de_sequence(de_seq, smooth_kernel=3):
    """
    对 trial 内的 DE 序列做滑动平均。

    de_seq: [W, C, 5]

    return:
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
    win_len=1000,
    step=1000,
    smooth_kernel=3,
):
    """
    对一个 trial 提取 DE 序列。

    trial: [30, 12500]

    return:
        de_seq: [W, 30, 5]
    """

    trial = np.asarray(trial, dtype=np.float32)
    C, T = trial.shape

    de_list = []

    for start in range(0, T - win_len + 1, step):
        end = start + win_len
        x_win = trial[:, start:end]  # [30, 1000]

        de_feat = compute_de_one_window(
            x_win,
            sfreq=sfreq,
            bands=DE_BANDS
        )  # [30, 5]

        de_list.append(de_feat)

    if len(de_list) == 0:
        raise ValueError(f"trial 太短，无法提取 DE: trial shape = {trial.shape}")

    de_seq = np.stack(de_list, axis=0)  # [W, 30, 5]

    de_seq = smooth_de_sequence(
        de_seq,
        smooth_kernel=smooth_kernel
    )

    return de_seq.astype(np.float32)


def build_competition_4class_index(
    mat_root: str,
    save_trial_root: str,
    save_de_root: str,
    out_csv: str,
    smooth_kernel: int = 3,
):
    """
    构建比赛 4 分类索引 CSV。

    新增内容：
    1. 保存每个 trial 的 EEG 到 trial_path
    2. 保存每个 trial 的 DE 序列到 de_path
    3. CSV 中每个样本增加 de_path 和 de_win_id

    CSV 每一行对应一个 4 秒 EEG 窗口：
        x = trial[:, start:end]
        de_feat = de_seq[de_win_id]
    """

    os.makedirs(save_trial_root, exist_ok=True)
    os.makedirs(save_de_root, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(mat_root, "*.mat")))
    records = []

    for mat_path in mat_files:
        file_name = os.path.basename(mat_path)
        diagnosis, subject_id = parse_subject_info_build(file_name)
        diagnosis_label = 1 if diagnosis == "DEP" else 0

        mat = load_mat_auto(mat_path)

        print(f"\n正在处理: {file_name}")
        print("mat keys:", list(mat.keys()))

        eeg_neu = np.asarray(mat["EEG_data_neu"], dtype=np.float32)
        eeg_pos = np.asarray(mat["EEG_data_pos"], dtype=np.float32)

        print("EEG_data_neu shape:", eeg_neu.shape)
        print("EEG_data_pos shape:", eeg_pos.shape)

        if eeg_neu.shape != (30, 50000):
            raise ValueError(f"{file_name} 的 EEG_data_neu shape 异常: {eeg_neu.shape}")
        if eeg_pos.shape != (30, 50000):
            raise ValueError(f"{file_name} 的 EEG_data_pos shape 异常: {eeg_pos.shape}")

        # =====================================================
        # 1. 处理中性 neu 的 4 个 trial
        # =====================================================
        for i in range(4):
            trial_id = i + 1

            s = i * TRIAL_LEN
            e = (i + 1) * TRIAL_LEN

            trial = eeg_neu[:, s:e]  # [30, 12500]

            # 保存原始 trial
            trial_name = f"{diagnosis}{subject_id}_trial{trial_id:02d}_neu.npy"
            trial_path = os.path.join(save_trial_root, trial_name)
            np.save(trial_path, trial)

            # 保存该 trial 的 DE 序列
            de_name = f"{diagnosis}{subject_id}_trial{trial_id:02d}_neu_de.npy"
            de_path = os.path.join(save_de_root, de_name)

            de_seq = extract_trial_de_sequence(
                trial,
                sfreq=SFREQ,
                win_len=WIN_LEN,
                step=STEP,
                smooth_kernel=smooth_kernel,
            )  # [W, 30, 5]

            np.save(de_path, de_seq)

            print(f"Saved trial: {trial_path}")
            print(f"Saved DE: {de_path}, shape={de_seq.shape}")

            # 每一个 EEG 窗口对应一个 de_win_id
            starts = list(range(0, TRIAL_LEN - WIN_LEN + 1, STEP))

            for win_id, start in enumerate(starts):
                end = start + WIN_LEN

                records.append({
                    "subject_id": subject_id,
                    "file_name": file_name,
                    "diagnosis": diagnosis,
                    "diagnosis_label": diagnosis_label,
                    "emotion": "neu",
                    "emotion_label": 0,
                    "label4": get_label4(diagnosis, "neu"),

                    "trial_id": trial_id,
                    "trial_path": trial_path,
                    "start": start,
                    "end": end,

                    # 新增
                    "de_path": de_path,
                    "de_win_id": win_id,
                })

        # =====================================================
        # 2. 处理积极 pos 的 4 个 trial
        # =====================================================
        for i in range(4):
            trial_id = i + 5

            s = i * TRIAL_LEN
            e = (i + 1) * TRIAL_LEN

            trial = eeg_pos[:, s:e]  # [30, 12500]

            # 保存原始 trial
            trial_name = f"{diagnosis}{subject_id}_trial{trial_id:02d}_pos.npy"
            trial_path = os.path.join(save_trial_root, trial_name)
            np.save(trial_path, trial)

            # 保存该 trial 的 DE 序列
            de_name = f"{diagnosis}{subject_id}_trial{trial_id:02d}_pos_de.npy"
            de_path = os.path.join(save_de_root, de_name)

            de_seq = extract_trial_de_sequence(
                trial,
                sfreq=SFREQ,
                win_len=WIN_LEN,
                step=STEP,
                smooth_kernel=smooth_kernel,
            )  # [W, 30, 5]

            np.save(de_path, de_seq)

            print(f"Saved trial: {trial_path}")
            print(f"Saved DE: {de_path}, shape={de_seq.shape}")

            starts = list(range(0, TRIAL_LEN - WIN_LEN + 1, STEP))

            for win_id, start in enumerate(starts):
                end = start + WIN_LEN

                records.append({
                    "subject_id": subject_id,
                    "file_name": file_name,
                    "diagnosis": diagnosis,
                    "diagnosis_label": diagnosis_label,
                    "emotion": "pos",
                    "emotion_label": 1,
                    "label4": get_label4(diagnosis, "pos"),

                    "trial_id": trial_id,
                    "trial_path": trial_path,
                    "start": start,
                    "end": end,

                    # 新增
                    "de_path": de_path,
                    "de_win_id": win_id,
                })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n保存完成: {out_csv}")
    print(df.head())
    print(df["label4"].value_counts().sort_index())

    if "de_path" in df.columns:
        print("\nCSV 已包含 de_path / de_win_id")
        print(df[["trial_path", "start", "end", "de_path", "de_win_id"]].head())

def load_channel_names(ch_name_path="../ch_name.mat"):
    """
    读取并标准化通道名
    """
    name = loadmat(ch_name_path)
    ch_names = name["labels"].reshape(-1).tolist()
    ch_names = [arr.item() for arr in ch_names]

    name_map = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
        'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz',
        'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    }

    ch_names_std = [name_map.get(ch, ch) for ch in ch_names]
    return ch_names_std


def subject_wise_zscore(clean_all, eps=1e-6):
    """
    对同一个被试的完整 EEG 数据做 subject-wise + channel-wise z-score。

    clean_all: [C, T_total]
        同一个被试的 pos + neu 清洗后拼接数据

    返回:
        clean_all_norm: [C, T_total]
        mean: [C, 1]
        std: [C, 1]
    """

    mean = clean_all.mean(axis=1, keepdims=True)
    std = clean_all.std(axis=1, keepdims=True)

    std = np.where(std < eps, eps, std)

    clean_all_norm = (clean_all - mean) / std

    return clean_all_norm, mean, std


def com_preprocess_subject(pos_data, neu_data, sfreq=250, ch_name_path="../ch_name.mat"):
    """
    对同一个被试的 pos + neu 数据统一预处理。

    流程：
    1. pos 和 neu 拼接
    2. 统一滤波
    3. 统一 RANSAC 检测坏道
    4. 统一坏道插值
    5. 统一 ICA 去伪迹
    6. 切回 pos 和 neu
    7. subject-wise 标准化

    参数:
        pos_data: [C, T_pos]
        neu_data: [C, T_neu]

    返回:
        pos_clean_norm: [C, T_pos]
        neu_clean_norm: [C, T_neu]
        info_dict: 记录坏道、ICA 成分、均值方差等信息
    """

    # =========================
    # 0. 基本检查
    # =========================
    pos_data = np.asarray(pos_data, dtype=np.float64)
    neu_data = np.asarray(neu_data, dtype=np.float64)

    assert pos_data.ndim == 2, f"pos_data should be [C, T], got {pos_data.shape}"
    assert neu_data.ndim == 2, f"neu_data should be [C, T], got {neu_data.shape}"
    assert pos_data.shape[0] == neu_data.shape[0], \
        f"pos and neu channel number mismatch: {pos_data.shape} vs {neu_data.shape}"

    pos_len = pos_data.shape[1]
    neu_len = neu_data.shape[1]

    # =========================
    # 1. 同一被试 pos + neu 拼接
    # =========================
    all_data = np.concatenate([pos_data, neu_data], axis=1)  # [C, T_pos + T_neu]

    # =========================
    # 2. 创建 MNE Raw
    # =========================
    ch_names_std = load_channel_names(ch_name_path)

    info = mne.create_info(
        ch_names=ch_names_std,
        sfreq=sfreq,
        ch_types='eeg'
    )

    montage = mne.channels.make_standard_montage('standard_1020')

    raw = mne.io.RawArray(all_data, info, verbose=False)
    raw.set_montage(montage, on_missing='ignore')

    # =========================
    # 3. 滤波
    #    主数据: 0.1–45 Hz
    #    ICA 拟合数据: 1–45 Hz
    # =========================
    raw_main = raw.copy().filter(
        l_freq=0.1,
        h_freq=45,
        verbose=False
    )

    raw_ica = raw.copy().filter(
        l_freq=1.0,
        h_freq=45,
        verbose=False
    )

    # =========================
    # 4. RANSAC 坏道检测
    #    这里建议不要只做一个超长 epoch，
    #    而是切成多个短 epoch，更适合 RANSAC
    # =========================
    epochs = mne.make_fixed_length_epochs(
        raw_main,
        duration=2.0,
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

    ransac.fit(epochs)
    bads = ransac.bad_chs_

    print(f"RANSAC bad channels: {bads}")

    raw_main.info['bads'] = bads
    raw_ica.info['bads'] = bads

    raw_main.interpolate_bads(reset_bads=True, verbose=False)
    raw_ica.interpolate_bads(reset_bads=True, verbose=False)

    # =========================
    # 5. ICA 去伪迹
    # =========================
    ica = ICA(
        n_components=None,
        random_state=97,
        method='infomax',
        verbose=False
    )

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

    clean_all = raw_clean.get_data()  # [C, T_pos + T_neu]

    # =========================
    # 6. subject-wise 标准化
    #    在整个被试的 pos + neu 上算 mean/std
    # =========================
    clean_all_norm, subj_mean, subj_std = subject_wise_zscore(clean_all)

    # =========================
    # 7. 切回 pos 和 neu
    # =========================
    pos_clean_norm = clean_all_norm[:, :pos_len]
    neu_clean_norm = clean_all_norm[:, pos_len:pos_len + neu_len]

    assert pos_clean_norm.shape == pos_data.shape
    assert neu_clean_norm.shape == neu_data.shape

    info_dict = {
        "bad_channels": bads,
        "ica_labels": labels,
        "ica_exclude_idx": exclude_idx,
        "subject_mean": subj_mean,
        "subject_std": subj_std,
    }

    return pos_clean_norm, neu_clean_norm, info_dict

if __name__ == "__main__":
    # mat_root = "../data/com_rawdata"
    # mat_files = sorted(glob.glob(os.path.join(mat_root, "*.mat")))
    # print(mat_files)
    # count=0
    #
    # for i in mat_files:
    #     mat = load_mat_auto(i)
    #     diagnosis, subject_id = parse_subject_info(i)
    #     pos_data = mat["EEG_data_pos"]
    #     neu_data = mat["EEG_data_neu"]
    #     pos_clean = com_preprocess(pos_data)
    #     neu_clean = com_preprocess(neu_data)
    #     data = {"EEG_data_pos": pos_clean, "EEG_data_neu": neu_clean}
    #     if diagnosis == "DEP":
    #         save_path = f"../data/com_clean/{diagnosis}_{count}.mat"
    #         savemat(save_path, data)
    #         count+=1
    #     else:
    #         save_path = f"../data/com_clean/{diagnosis}_{count}.mat"
    #         savemat(save_path, data)
    #         count+=1

    mat_root = "../data/com_rawdata"
    save_root = "../data/com_clean_domain"

    os.makedirs(save_root, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(mat_root, "*.mat")))
    print(mat_files)

    count = 0

    for file_path in mat_files:
        print("\n" + "=" * 80)
        print(f"Processing file: {file_path}")

        mat = load_mat_auto(file_path)
        diagnosis, subject_id = parse_subject_info(file_path)

        pos_data = mat["EEG_data_pos"]
        neu_data = mat["EEG_data_neu"]

        print(f"diagnosis: {diagnosis}")
        print(f"subject_id: {subject_id}")
        print(f"pos_data shape: {pos_data.shape}")
        print(f"neu_data shape: {neu_data.shape}")

        # 同一个被试 pos + neu 统一预处理
        pos_clean, neu_clean, info_dict = com_preprocess_subject(
            pos_data=pos_data,
            neu_data=neu_data,
            sfreq=250,
            ch_name_path="../ch_name.mat"
        )

        data = {
            "EEG_data_pos": pos_clean,
            "EEG_data_neu": neu_clean,
            #
            # # 建议保存，方便以后检查
            # "subject_mean": info_dict["subject_mean"],
            # "subject_std": info_dict["subject_std"],
            #
            # # ICA 信息保存成 object 有时 MATLAB 不好读，
            # # 所以这里简单转成字符串保存
            # "bad_channels": np.array(info_dict["bad_channels"], dtype=object),
            # "ica_labels": np.array(info_dict["ica_labels"], dtype=object),
            # "ica_exclude_idx": np.array(info_dict["ica_exclude_idx"]),
        }

        save_path = os.path.join(save_root, f"{diagnosis}_{count}.mat")
        savemat(save_path, data)

        print(f"Saved to: {save_path}")
        print(f"pos_clean shape: {pos_clean.shape}")
        print(f"neu_clean shape: {neu_clean.shape}")

        count += 1

