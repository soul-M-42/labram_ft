import os
import re
import numpy as np
import mne
from scipy.io import savemat
from collections import defaultdict
import h5py
from pathlib import Path

print(np.__version__)
# 1.24.4
print(mne.__version__)
# 0.23.0



# ======================= 配置 =======================
dataset_name = 'SEEDV'
target_sfreq = 200
sample_T = 2.0 # EEG sample window timelength in s
sample_stride = 2.0 # # EEG sample window stride in s
label_dim = 5
input_dir = "C:\EEG_data\EEG_raw_cnt"
output_dir = f"V:\daily_eeg_emotion\Data\SEED-V\SEEDV_hdf5_T={sample_T}s_stride={sample_stride}"
sample_samples = int(sample_T * target_sfreq)
stride_samples = int(sample_stride * target_sfreq)

# ======================= trial 时间 =======================
# This is used for trial segmentation from raw eeg. Replace with your own code.
start_seconds = np.array([
    [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
])

end_seconds = np.array([
    [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
])

# ======================= 文件匹配 =======================
pattern = re.compile(r"(\d+)_(\d)_.*\.cnt")
subject_files = defaultdict(dict)

for f in os.listdir(input_dir):
    m = pattern.match(f)
    if m:
        sub, sess = m.groups()
        subject_files[sub][int(sess)] = os.path.join(input_dir, f)

n_sub = len(subject_files)

# ======================= 45 个 trial 标签 =======================

# Case 1. If your label is emotion class for each video:
trial_labels = np.array([
    4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0,
    2, 1, 3, 0, 4, 2, 1, 3, 0, 4, 2, 1, 3, 0, 4,
    2, 1, 3, 0, 4, 2, 1, 3, 0, 4, 2, 1, 3, 0, 4
])
trial_labels_onehot = (trial_labels.reshape(-1, 1) == np.arange(label_dim)).astype(int)
all_labels = np.tile(trial_labels_onehot, (n_sub, 1, 1))
print(all_labels.shape)

# Case 2. If your label is multi-dim dynamic float:
# all_labels = np.zeros((n_sub, n_vid, label_dim, T))

# sub_trial_label = all_labels[sub][i_trial]
# all_labels[sub][i_trial] is [label_dim, T]
# T can be attribute and it will be resampled to match n_eeg_segment 


def stretch_axis(arr, T_prime):
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[np.newaxis, :] # 变为 [1, k]
    
    T = arr.shape[-1]
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, T_prime)
    
    # 对每一行执行插值
    res = np.apply_along_axis(lambda y: np.interp(x_new, x_old, y), axis=-1, arr=arr)
    
    return res.flatten() if is_1d else res


# ======================= 主处理 =======================
ch_names = None
data_save_path = Path(f"{output_dir}.h5")
data_save_path.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(data_save_path, 'w') as f:
    for sub, sessions in subject_files.items():
        sub = int(sub)
        print(sub)
        # if(sub != '7'):
        #     continue
        print(f"\n处理 sub_{sub}")

        raws = []
        sfreq = None

        # ---- 读取 & 拼 session ----
        for sess in [1, 2, 3]:
            print(f'reading {sessions[sess]}')
            raw = mne.io.read_raw_cnt(
                sessions[sess],
                preload=True
            )
            useless_ch = ['M1', 'M2', 'VEO', 'HEO']
            raw.drop_channels(useless_ch)
            ch_names = raw.ch_names
            raw.notch_filter(freqs=50.0, verbose=False)
            raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
            raw.resample(target_sfreq, verbose=False)
            raws.append(raw)
            if sfreq is None:
                sfreq = raw.info['sfreq']


        # ======================= 只保留 trial 片段 =======================
        trial_segments = []

        trial_idx = 0
        for sess in range(3):
            for t in range(15):
                start_sec = start_seconds[sess, t]
                end_sec = end_seconds[sess, t]

                start_sample = int(round(start_sec * target_sfreq))
                end_sample = int(round(end_sec * target_sfreq))

                # segment = data_all[:, start_sample:end_sample]
                segment = raws[sess].get_data()[:, start_sample:end_sample]
                trial_segments.append(segment)

                trial_idx += 1

        print(len(trial_segments))
        print(ch_names)
        sub_group = f.create_group(f'sub{sub}')
        for i_trial, data_trial in enumerate(trial_segments):
            print(f'sub {sub} trial {i_trial}: {data_trial.shape}')
            vid_grp = sub_group.create_group(f"vid{i_trial}")
            n_samples = data_trial.shape[-1]
            sub_trial_label = all_labels[sub][i_trial]
            n_slices = (n_samples - sample_samples + stride_samples) // stride_samples
            sub_trial_label = stretch_axis(sub_trial_label, n_slices)
            for i_slice, start in enumerate(range(0, n_samples - sample_samples + 1, stride_samples)):
                end = start + sample_samples
                slice_data = data_trial[:, start:end]
                slice_grp = vid_grp.create_group(f"sample{i_slice}")
                dset = slice_grp.create_dataset('eeg', data=slice_data)
                dset.attrs['chOrder'] = ch_names
                dset.attrs['rsFreq'] = target_sfreq
                dset.attrs['label'] = sub_trial_label[i_slice]
                dset.attrs['video_id'] = i_trial
                dset.attrs['start_sample'] = start  # 可选：记录切片在原数据中的起始位置
