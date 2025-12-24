from tqdm import tqdm
import torch
import torch.nn as nn

def finetune_one_epoch(
        base_model,
        reg_head,
        dataloader,
        input_cha_ids,
        optimizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):

    base_model.train()
    reg_head.train()

    criterion = nn.MSELoss()
    total_loss = 0.0

    for eeg, label in tqdm(dataloader):
        eeg = eeg.to(device)
        y_real = label.to(device)

        # forward
        fea = base_model(eeg, input_cha_ids)
        y_pred = reg_head(fea)

        loss = criterion(y_pred, y_real)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
