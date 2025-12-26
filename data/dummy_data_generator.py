import os
import numpy as np
import scipy.io as sio

def create_test_data(
    root_dir="test_data_root",
    n_subjects=2,
    n_trials=3,
    trial_lengths=(100, 150, 50),  # seconds
    n_channels=3,
    fs=200,
    n_label_dim=3,
):
    os.makedirs(root_dir, exist_ok=True)

    for sub_id in range(1, n_subjects + 1):
        sub_dir = os.path.join(root_dir, f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)

        all_data = []
        trial_durations = []

        label_dict = {}

        for trial_id, duration in enumerate(trial_lengths):
            n_points = duration * fs

            # ===== data =====
            # [n_channel, n_points]
            trial_data = np.random.randn(n_channels, n_points).astype(np.float32)
            all_data.append(trial_data)
            trial_durations.append(duration)

            # ===== label =====
            # 模拟时间序列标签 [n_dim, T_label]
            T_label = duration * 10  # label 时间分辨率低一点
            t = np.linspace(0, duration, T_label)

            # 举例：平滑变化的标签
            label = np.stack([
                np.sin(2 * np.pi * t / duration),
                np.cos(2 * np.pi * t / duration),
                np.linspace(0, 1, T_label),
            ], axis=0).astype(np.float32)

            label_dict[f"trial_{trial_id}"] = label

        # 拼接 trials
        all_data = np.concatenate(all_data, axis=1)

        # ===== 保存 data.mat =====
        sio.savemat(
            os.path.join(sub_dir, "data.mat"),
            {
                "data": all_data,
                "trial_duration": np.array(trial_durations, dtype=np.float32)[None, :]
            }
        )

        # ===== 保存 label.mat =====
        sio.savemat(
            os.path.join(sub_dir, "label.mat"),
            label_dict
        )

    print(f"Test data created at: {root_dir}")


if __name__ == "__main__":
    create_test_data(root_dir='/emo-eeg/dummy')
