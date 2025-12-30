import torch
import torch.nn as nn
from pathlib import Path
from timm.utils import get_state_dict
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def model_detail(model: nn.Module):
    """
    打印模型的每一层信息，包括参数是否训练，可训练参数，总参数量和模型大小
    """
    print(f"{'Layer':40s} {'Has Params':10s} {'Trainable Params':15s} {'# Params':10s}")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if name == "":  # 跳过顶层module
            continue
        # 统计参数数量
        num_params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        has_params = num_params > 0
        total_params += num_params
        trainable_params += trainable
        print(f"{name:40s} {str(has_params):10s} {str(trainable):15s} {str(num_params):10s}")
    
    # 计算模型大小（假设每个参数4字节 float32）
    size_MB = total_params * 4 / (1024**2)
    
    print("="*80)
    print(f"Total parameters: {total_params} ({trainable_params} trainable)")
    print(f"Approx. model size: {size_MB:.2f} MB")



def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

def save_model(output_dir, epoch, model, optimizer, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
    epoch_name = str(epoch)
    output_dir = Path(output_dir)

    checkpoint_paths = [output_dir / 'checkpoint.pth']
    if epoch == 'best':
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
    elif (epoch + 1) % save_ckpt_freq == 0:
        checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)
            
        if optimizer_disc is not None:
            to_save['optimizer_disc'] = optimizer_disc.state_dict()

        save_on_master(to_save, checkpoint_path)

def zscore_norm(X):
    print(f'Z-score')
    mean = X.mean(axis=0, keepdims=True)  # [1, n_dim]
    std = X.std(axis=0, keepdims=True)    # [1, n_dim]
    std[std == 0] = 1
    X_z = (X - mean) / std
    return X_z