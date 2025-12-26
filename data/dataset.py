import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from tqdm import tqdm
from pathlib import Path

from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from math import gcd
def resample_ndim_signal(x, fs_old, fs_new):
    """
    x: np.ndarray, shape = [n_dim, n_t]
    fs_old: 原采样率
    fs_new: 目标采样率
    """
    # 计算最简 up / down
    g = gcd(fs_old, fs_new)
    up = fs_new // g
    down = fs_old // g

    # 沿时间轴重采样（axis=1）
    x_resampled = resample_poly(x, up=up, down=down, axis=1)

    return x_resampled

def interp_to_length(x, n_t_new, axis=1):
    """
    将数组 x 在指定轴上线性插值 / 拉伸到指定长度
    支持 n_t_old == 1 的极端情况
    """
    x = np.asarray(x)
    n_t_old = x.shape[axis]

    # ---------- 极端情况 ----------
    if n_t_old == 1:
        # 常数扩展
        reps = [1] * x.ndim
        reps[axis] = n_t_new
        return np.tile(x, reps)

    # ---------- 正常插值 ----------
    t_old = np.linspace(0, 1, n_t_old)
    t_new = np.linspace(0, 1, n_t_new)

    f = interp1d(
        t_old,
        x,
        axis=axis,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return f(t_new)

class SubjectSliceDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        dataset_name,
        cache_root,
        fs_ori,
        fs_tar,
        window_size,
        stride,
        rebuild_cache=False,
    ):
        self.root_dir = os.path.join(data_root_dir, dataset_name)
        self.dataset_name = dataset_name
        self.cache_root = os.path.abspath(cache_root)
        self.fs_ori = fs_ori
        self.fs_tar = fs_tar
        self.window_size = window_size
        self.stride = stride

        self.window_points = int(window_size * fs_tar)
        self.stride_points = int(stride * fs_tar)

        # ===== 自动生成 cache 目录 =====
        cache_dir_name = f"fs{fs_ori}_win{window_size}s_stride{stride}s"
        self.data_cache_dir = os.path.join(
            self.cache_root,
            dataset_name,
            cache_dir_name
        )


        self.index = []  # 保存所有切片路径
        if rebuild_cache or not self._cache_exists():
            print(f"Building cache at: {self.data_cache_dir}")
            os.makedirs(self.data_cache_dir, exist_ok=True)
            self._build_cache()
        else:
            print(f"Using existing cache at: {self.data_cache_dir}")
            self._load_index()
        
        self.n_sub = sum(1 for sub in Path(self.data_cache_dir).iterdir() if sub.is_dir())

    def _cache_exists(self):
        to_check = os.path.join(self.data_cache_dir, "index.npy")
        return os.path.exists(to_check)
    def _build_cache(self):
        self.index = []

        subjects = sorted(
            d for d in os.listdir(self.root_dir)
            if d.startswith("sub_")
        )

        for sub in tqdm(subjects, desc="Processing subjects"):
            sub_dir = os.path.join(self.root_dir, sub)
            cache_sub_dir = os.path.join(self.data_cache_dir, sub)
            os.makedirs(cache_sub_dir, exist_ok=True)

            # ---- load mat files ----
            data_mat = sio.loadmat(os.path.join(sub_dir, "data.mat"))
            label_mat = sio.loadmat(os.path.join(sub_dir, "label.mat"))

            # data.mat:
            #   data: [n_channel, n_total_points]
            #   trial_duration: [1, n_trial]
            data = data_mat["data"]
            data = resample_ndim_signal(data, self.fs_ori, self.fs_tar)
            trial_durations = data_mat["trial_duration"].squeeze()

            point_cursor = 0

            for trial_id, duration in enumerate(trial_durations):
                n_points = int(duration * self.fs_tar)

                trial_data = data[:, point_cursor:point_cursor + n_points]
                trial_label = label_mat[f'trial_{trial_id}']  # [n_dim, T_label]
                if(trial_label.shape[1] < duration):
                    trial_label = interp_to_length(trial_label, int(duration))

                point_cursor += n_points

                n_dim, label_T = trial_label.shape
                label_time = np.linspace(0, duration, label_T)

                slice_id = 0
                for start in range(
                    0,
                    n_points - self.window_points + 1,
                    self.stride_points
                ):
                    end = start + self.window_points
                    data_slice = trial_data[:, start:end]
                    data_slice = np.reshape(data_slice, (data_slice.shape[0], int(self.window_size), int(self.fs_tar)))


                    # label 时间窗口平均
                    t_start = start / self.fs_tar
                    t_end = end / self.fs_tar
                    mask = (label_time >= t_start) & (label_time <= t_end)
                    label_slice = trial_label[:, mask].mean(
                        axis=1, keepdims=True
                    ).squeeze()
                    save_path = os.path.join(
                        cache_sub_dir,
                        f"trial_{trial_id}_slice_{slice_id:04d}.npy"
                    )

                    np.save(save_path, {
                        "data": data_slice.astype(np.float32),
                        "label": label_slice.astype(np.float32),
                    })

                    self.index.append(save_path)
                    slice_id += 1

        np.save(
            os.path.join(self.data_cache_dir, "index.npy"),
            self.index
        )
        print(f'saving {os.path.join(self.data_cache_dir, "index.npy")}')

    def _load_index(self):
        self.index = np.load(
            os.path.join(self.data_cache_dir, "index.npy"),
            allow_pickle=True
        ).tolist()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = np.load(self.index[idx], allow_pickle=True).item()
        data = torch.from_numpy(item["data"])      # [n_channel, n_points]
        label = torch.from_numpy(item["label"])    # [n_dim, 1]
        return data, label

def get_dataset(args):
    return SubjectSliceDataset(
        data_root_dir=args.DATA_ROOT_DIR,
        dataset_name=args.DATASET_NAME,
        cache_root=args.CACHE_ROOT,
        fs=args.FS,
        window_size=args.WINDOW_SIZE,
        stride=args.STRIDE,
        rebuild_cache=False
    )

# ==========================================================
# Unit test
# ==========================================================
if __name__ == "__main__":

    DATASET_NAME = 'dummy'
    ROOT_DIR = f"/emo-eeg/{DATASET_NAME}"
    CACHE_ROOT = "cache"

    FS = 200
    WINDOW_SIZE = 2.0
    STRIDE = 1.0
    BATCH_SIZE = 4

    print("Creating dataset...")
    dataset = SubjectSliceDataset(
        data_root_dir=ROOT_DIR,
        dataset_name=DATASET_NAME,
        cache_root=CACHE_ROOT,
        fs=FS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        rebuild_cache=False
    )

    print("Cache directory:", dataset.data_cache_dir)
    print("Dataset length:", len(dataset))
    assert len(dataset) > 0, "Dataset is empty!"

    # 单样本测试
    data, label = dataset[0]
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)

    assert data.ndim == 2
    assert label.ndim == 2
    assert label.shape[1] == 1
    assert data.dtype == torch.float32
    assert label.dtype == torch.float32

    # DataLoader 测试
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    batch_data, batch_label = next(iter(loader))
    print("Batch data shape:", batch_data.shape)
    print("Batch label shape:", batch_label.shape)

    assert batch_data.ndim == 3
    assert batch_label.ndim == 3
    assert not torch.isnan(batch_data).any()
    assert not torch.isnan(batch_label).any()

    print("\n✅ Dataset unit test passed.")
