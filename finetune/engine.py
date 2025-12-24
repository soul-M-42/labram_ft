from tqdm import tqdm
import torch

def finetune_one_epoch(
        model,
        dataloader,
        input_cha_ids,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):
    for eeg, label in tqdm(dataloader):
        eeg = eeg.to(device)
        label = label.to(device)
        fea = model(eeg, input_cha_ids)
        print(fea.shape)
    return