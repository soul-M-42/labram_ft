import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from tqdm import tqdm


class SubjectSliceDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        dataset_name,
        cache_root,
        fs,
        window_size,
        stride,
        rebuild_cache=False,
    ):
        self.root_dir = os.path.join(data_root_dir, dataset_name)
        self.dataset_name = dataset_name
        self.cache_root = os.path.abspath(cache_root)
        self.fs = fs
        self.window_size = window_size
        self.stride = stride

        self.window_points = int(window_size * fs)
        self.stride_points = int(stride * fs)

        # ===== 自动生成 cache 目录 =====
        cache_dir_name = f"fs{fs}_win{window_size}s_stride{stride}s"
        self.data_cache_dir = os.path.join(
            self.cache_root,
            dataset_name,
            cache_dir_name
        )


        self.index = []  # 保存所有切片路径
        if rebuild_cache or not self._cache_exists():
            print(f"Building cache at: {self.data_cache_dir}")
            os.makedirs(self.data_cache_dir)
            self._build_cache()
        else:
            print(f"Using existing cache at: {self.data_cache_dir}")
            self._load_index()

    def _cache_exists(self):
        to_check = os.path.join(self.data_cache_dir, "index.npy")
        to_check = self.data_cache_dir
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
            trial_durations = data_mat["trial_duration"].squeeze()

            # label.mat: n_trial 个 label 矩阵
            labels = [
                label_mat[key]
                for key in sorted(label_mat.keys())
                if not key.startswith("__")
            ]

            point_cursor = 0

            for trial_id, duration in enumerate(trial_durations):
                n_points = int(duration * self.fs)

                trial_data = data[:, point_cursor:point_cursor + n_points]
                trial_label = labels[trial_id]  # [n_dim, T_label]

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

                    # label 时间窗口平均
                    t_start = start / self.fs
                    t_end = end / self.fs
                    mask = (label_time >= t_start) & (label_time <= t_end)
                    label_slice = trial_label[:, mask].mean(
                        axis=1, keepdims=True
                    )

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
        root_dir=ROOT_DIR,
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
