from tqdm import tqdm
import torch
import torch.nn as nn

def finetune_one_epoch(
        base_model,
        reg_head,
        dataloader,
        input_cha_ids,
        optimizer,
        writer,          # TensorBoard writer
        epoch,           # 当前 epoch
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):

    base_model.train()
    reg_head.train()

    criterion = nn.MSELoss()
    total_loss = 0.0

    for step, (eeg, label) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}")):
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

        # 可选：按 step 记录
        global_step = epoch * len(dataloader) + step
        writer.add_scalar("Train/Loss_step", loss.item(), global_step)

    avg_loss = total_loss / len(dataloader)

    # 按 epoch 记录
    writer.add_scalar("Train/Loss_epoch", avg_loss, epoch)

    return avg_loss

def validate_one_epoch(
        base_model,
        reg_head,
        dataloader,
        input_cha_ids,
        writer,          # TensorBoard writer
        epoch,           # 当前 epoch
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):

    base_model.eval()
    reg_head.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for eeg, label in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            eeg = eeg.to(device)
            y_real = label.to(device)

            fea = base_model(eeg, input_cha_ids)
            y_pred = reg_head(fea)

            loss = criterion(y_pred, y_real)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # TensorBoard 记录
    writer.add_scalar("Val/Loss_epoch", avg_loss, epoch)

    return avg_loss
