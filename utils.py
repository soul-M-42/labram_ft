import torch
import torch.nn as nn


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