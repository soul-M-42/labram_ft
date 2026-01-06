from torch.utils.data import Sampler, DataLoader
from data.dataset import SubjectSliceDataset
from data.dataset_hdf5 import EEGDataset
import os


def get_loader(args, subs=None, vids=None, shuffle=True, batch_size=4, num_workers=0):
    """
    subs: 需要包含的主体 ID 列表 (e.g., [1, 2])
    vids: 需要包含的视频/试验 ID 列表 (e.g., [0, 5])
    """
    dataset = EEGDataset(
        args.h5path,
        subs,
        vids
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

