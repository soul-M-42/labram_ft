from torch.utils.data import Sampler, DataLoader
from data.dataset import SubjectSliceDataset
import os

class SubListSampler(Sampler):
    """根据指定的 subject 列表进行采样"""
    def __init__(self, dataset, sub_list):
        """
        dataset: SubjectSliceDataset 实例
        sub_list: list[int] 指定的 sub_id 列表
        """
        self.dataset = dataset
        self.sub_list = sub_list
        self.indices = self._get_indices()

    def _get_indices(self):
        selected_indices = []
        for idx, path in enumerate(self.dataset.index):
            # path 类似: .../sub_0/trial_0_slice_0000.npy
            sub_id = int(os.path.basename(os.path.dirname(path)).split('_')[1])
            if sub_id in self.sub_list:
                selected_indices.append(idx)
        return selected_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_loader(args, sub_list=None, shuffle=True, batch_size=4, num_workers=0):
    """
    返回指定 sub_list 的 DataLoader
    sub_list: [0,1,2] 形式，若为 None 则使用全部数据
    """
    dataset = SubjectSliceDataset(
        data_root_dir=args.DATA_ROOT_DIR,
        dataset_name=args.DATASET_NAME,
        cache_root=args.CACHE_ROOT,
        fs=args.FS,
        window_size=args.WINDOW_SIZE,
        stride=args.STRIDE,
        rebuild_cache=False
    )

    if sub_list is not None:
        sampler = SubListSampler(dataset, sub_list)
        shuffle_flag = False  # 使用 sampler 时不要 shuffle
    else:
        sampler = None
        shuffle_flag = shuffle

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
    )
    return loader

# ===========================
# Unit test 示例
# ===========================
if __name__ == "__main__":
    class Args:
        DATA_ROOT_DIR = "/emo-eeg/dummy"
        DATASET_NAME = "dummy"
        CACHE_ROOT = "cache"
        FS = 200
        WINDOW_SIZE = 2.0
        STRIDE = 1.0

    args = Args()

    # 只使用 sub_0 和 sub_1
    train_loader = get_loader(args, sub_list=[1,2], batch_size=2)
    print(len(train_loader))
