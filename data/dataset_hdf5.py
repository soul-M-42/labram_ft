import torch
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, h5_path, subs=None, vids=None):
        """
        Args:
            h5_path: .h5 文件的路径
            subs: 需要包含的主体 ID 列表 (e.g., [1, 2])
            vids: 需要包含的视频/试验 ID 列表 (e.g., [0, 5])
        """
        self.h5_path = h5_path
        self.samples = []

        with h5py.File(self.h5_path, 'r') as f:
            # 获取所有 subject 键名
            all_sub_keys = list(f.keys())
            
            # 使用 tqdm 包装循环，desc 用于显示进度条提示文字
            for sub_key in tqdm(all_sub_keys, desc="Loading Subjects"):
                sub_id = int(''.join(filter(str.isdigit, sub_key)))
                
                if subs is not None and sub_id not in subs:
                    continue
                
                sub_group = f[sub_key]
                for vid_key in sub_group.keys():
                    vid_id = int(''.join(filter(str.isdigit, vid_key)))
                    
                    if vids is not None and vid_id not in vids:
                        continue
                    
                    vid_group = sub_group[vid_key]
                    for sample_key in vid_group.keys():
                        self.samples.append((sub_key, vid_key, sample_key))
        
        print(f"Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub_key, vid_key, sample_key = self.samples[idx]
        
        # 建议：如果是频繁读取，考虑在外部保持文件开启或使用缓存
        with h5py.File(self.h5_path, 'r') as f:
            dataset = f[sub_key][vid_key][sample_key]['eeg']
            data = dataset[:]  # Shape: (Channels, Time Points)
            label = dataset.attrs['label']
            
        x = torch.from_numpy(data).float()
        # 如果 label 是标量，建议直接转换
        y = torch.tensor(label).float() 
        
        return x, y