import torch
import glob
import h5py
import os
import re
from torch.utils.data import Dataset
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, directory_path, subs=None, vids=None):
        """
        subs: list of int, e.g., [1, 2, 3]
        vids: list of int, e.g., [0, 1, 5]
        """
        all_files = sorted(glob.glob(os.path.join(directory_path, "sub*.h5")))
        self.registry = []
        
        # 预过滤子集文件
        if subs is not None:
            # 匹配 filename 中的数字，例如 sub4.h5 -> 4
            all_files = [f for f in all_files if int(re.search(r'sub(\d+)', os.path.basename(f)).group(1)) in subs]

        for fp in tqdm(all_files, desc="Indexing Dataset"):
            with h5py.File(fp, 'r', locking=False) as f:
                # 过滤 vid 级别子集
                target_vids = f.keys()
                if vids is not None:
                    target_vids = [v for v in target_vids if int(re.search(r'vid(\d+)', v).group(1)) in vids]
                
                for vid in target_vids:
                    for sample in f[vid].keys():
                        self.registry.append((fp, f"{vid}/{sample}/eeg"))

    def __len__(self):
        return len(self.registry)

    def __getitem__(self, idx):
        file_path, eeg_path = self.registry[idx]
        with h5py.File(file_path, 'r', locking=False) as f:
            dataset = f[eeg_path]
            data = torch.from_numpy(dataset[:]).float()
            label = torch.tensor(dataset.attrs['label']).float()
        return data, label

# 使用示例: 只读取 sub1, sub2 和 vid0, vid1 的数据