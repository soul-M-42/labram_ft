from model.base_model import get_base_model
from model.utils import model_detail, get_input_chans, zscore_norm
from model.reg_head_mlp import MLPRegHead
from data.dataloader import get_loader
from finetune.engine import finetune_one_epoch, validate_one_epoch
import yaml
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from tqdm import tqdm
import os
import numpy as np
from downstream.utils import classification, regression

def get_args_from_yaml(yaml_path: str):
    """
    Load configuration from a YAML file and return an object similar to argparse.Namespace
    """
    # Load YAML file
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a default argparse parser to get defaults
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str)

    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--finetune', default='')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--nb_classes', default=0, type=int)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str)

    parser.add_argument('--h5path', default=None, type=str)
    parser.add_argument('--CACHE_ROOT', default='cache', type=str)
    parser.add_argument('--LABEL_DIM', default=3, type=int)
    parser.add_argument('--BATCH_SIZE', default=4, type=int)
    parser.add_argument('--TRAIN_SUBS', default=None, type=int, nargs='+', help="训练子集列表")
    parser.add_argument('--VAL_SUBS', default=None, type=int, nargs='+', help="验证子集列表")
    
    parser.add_argument('--finetune_epochs', default=20, type=int)
    parser.add_argument('--run_name', default='default_run_name', type=str)

    parser.add_argument('--ext', type=bool)
    parser.add_argument('--reg', type=bool)
    parser.add_argument('--cls', type=bool)

    # Build a namespace object with defaults
    args = parser.parse_args([])  # empty list to only get defaults

    # Override defaults with YAML values
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Unknown config key '{key}' in YAML, ignoring.")

    # Handle deepspeed import if enabled
    if getattr(args, 'enable_deepspeed', False):
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            from deepspeed.runtime.engine import DeepSpeedEngine
            args, ds_init = deepspeed.add_config_arguments(args), deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return args, ds_init


if __name__ == '__main__':
    args, ds_init = get_args_from_yaml('./cfgs/arg_ex.yaml')
    fea_save_dir = 'fea'  
    fea_dir = f'{fea_save_dir}/fea_{args.run_name}.npy'
    label_dir = f'{fea_save_dir}/label_{args.run_name}.npy'  
    if(args.ext):
        # ---------- args & model ----------
        print(args)

        base_model = get_base_model(args)
        # model_detail(base_model)
        # ---------- dataloader ----------
        val_loader = get_loader(
            args,
            subs=args.VAL_SUBS,
            batch_size=args.BATCH_SIZE
        )
        print(len(val_loader))
        print(len(val_loader.dataset))
        input_cha_names = val_loader.dataset.ch_names
        input_cha_ids = get_input_chans(input_cha_names)
        device = torch.device('cuda')
        fea_all = []
        label_all = []
        for eeg, label in tqdm(val_loader):
            eeg = eeg.to(device)
            y_real = label.to(device)
            fea = base_model(eeg, input_cha_ids)
            fea_all.append(fea.detach().cpu().numpy())
            label_all.append(y_real.detach().cpu().numpy())
        fea_all = np.concatenate(fea_all)
        label_all = np.concatenate(label_all)
        if not os.path.exists(fea_save_dir):
            os.makedirs(fea_save_dir)
        # optional fea process
        fea_all = zscore_norm(fea_all)
        np.save(fea_dir, fea_all)
        np.save(label_dir,label_all)
        print(f'Feature saved to {fea_dir}')

    print(f'loading feature from {fea_dir}')
    fea = np.load(fea_dir)
    label = np.load(label_dir)
    print(f'fea shape {fea.shape}')


    n = fea.shape[0]
    mask = np.array([0] * (n // 2) + [1] * (n - n // 2))
    # SVR regression
    if(args.reg):
        model, y_pred, metrics = regression(fea, label, mask)
        print(f"SVR Regression ({y_pred.shape[1]}D)")
        for d, m in enumerate(metrics["per_dim"]):
            print(f"[Dim {d}] MSE: {m['mse']:.6f}, R2: {m['r2']:.6f}")
    # SVM classification
    if(args.cls):
        model, y_pred, metrics = classification(fea, label, mask)
        print(metrics["accuracy"])
    
