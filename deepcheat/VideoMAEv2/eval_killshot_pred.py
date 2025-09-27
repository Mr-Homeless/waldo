# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import json
import os
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
from dataset import build_dataset
from engine_for_finetuning import (
    final_test_killshot,
    predict_killshot_scores,
    merge,
    train_one_epoch,
    validation_one_epoch,
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate


import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import torch
from torchvision import transforms
from PIL import Image
from collections import deque

class VideoLabelingDataset:
    def __init__(self, directory_path):
        if not os.path.exists(directory_path):
            raise ValueError(f"{directory_path} does not exist.")

        self.image_paths = sorted([os.path.join(directory_path, filename) 
                                   for filename in os.listdir(directory_path) 
                                   if filename.endswith(".jpg")])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def image_generator(self):
        """
        A generator that yields transformed images one by one.
        """
        for image_path in self.image_paths:
            image = Image.open(image_path)
            image = self.transform(image)
            yield image

    def load_images(self, index_start, index_end):
        if index_end > len(self.image_paths) or index_start < 0:
            raise ValueError("Invalid index range provided.")

        gen = self.image_generator()
        
        # Initialize deque with maximum length as difference between indices
        images = deque(maxlen=index_end - index_start)
        
        # Move generator to start index
        for _ in range(index_start):
            next(gen)
        
        # Populate the deque till the end index
        for _ in range(index_end - index_start):
            images.append(next(gen))

        # Convert deque of tensors to a single tensor
        return torch.stack(list(images),dim=1)

    def sequential_load(self, length):
        """
        Yield batches of images sequentially, shifting by one.
        Args:
            length (int): Length of each sequence.
        """
        gen = self.image_generator()
        
        # Initialize deque with maximum length
        images = deque(maxlen=length)
        
        # Load initial batch
        for _ in range(length):
            images.append(next(gen))
        
        yield torch.stack(list(images),dim=1)

        # For each subsequent image, pop the oldest and append the new one
        for image in gen:
            images.append(image)
            yield torch.stack(list(images),dim=1)

    def __len__(self):
        return len(self.image_paths)

class MultiLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(MultiLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Step 1: Apply sigmoid activation to the logits (x)
        logprobs = torch.sigmoid(x)

        # Step 2: Compute the NLL loss (Negative log-likelihood)
        # We'll use binary log loss as it's multi-label classification
        nll_loss = -(target * torch.log(logprobs) + (1 - target) * torch.log(1 - logprobs))
        
        # Step 3: Compute the smooth loss
        # We take the mean of the logprobs, then negate it (as we did with nll_loss)
        smooth_loss = -logprobs.mean(dim=-1)
        
        # Step 4: Combine the NLL loss and the smooth loss
        loss = self.confidence * nll_loss.mean(dim=-1) + self.smoothing * smooth_loss

        return loss.mean()

class MultiLabelSoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid activation to logits
        logprobs = torch.sigmoid(x)
        #breakpoint()

        # Compute the loss for soft targets in a multi-label setting
        # This uses binary cross-entropy loss for each label
        loss = -(target * torch.log(logprobs) + (1 - target) * torch.log(1 - logprobs))

        # Sum across classes and then take the mean across the batch
        loss = loss.sum(dim=-1).mean()

        return loss


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE fine-tuning and evaluation script for action classification',
        add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='vit_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--input_size', default=224, type=int, help='images input size')

    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)

    parser.add_argument(
        '--drop',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Dropout rate (default: 0.)')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Attention dropout rate (default: 0.)')
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')
    parser.add_argument(
        '--head_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='cls head dropout rate (default: 0.)')

    parser.add_argument(
        '--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument(
        '--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument(
        '--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-8,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.4,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--num_sample', type=int, default=2, help='Repeated_aug (default: 2)')
    parser.add_argument(
        '--aa',
        type=str,
        default='rand-m7-n4-mstd0.5-inc1',
        metavar='NAME',
        help=
        'Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'
    ),
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.1,
        help='Label smoothing (default: 0.1)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help=
        'Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=10)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # * Random Erase params
    parser.add_argument(
        '--reprob',
        type=float,
        default=0.25,
        metavar='PCT',
        help='Random erase prob (default: 0.25)')
    parser.add_argument(
        '--remode',
        type=str,
        default='pixel',
        help='Random erase mode (default: "pixel")')
    parser.add_argument(
        '--recount',
        type=int,
        default=1,
        help='Random erase count (default: 1)')
    parser.add_argument(
        '--resplit',
        action='store_true',
        default=False,
        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.8,
        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument(
        '--cutmix',
        type=float,
        default=1.0,
        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax',
        type=float,
        nargs='+',
        default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help=
        'Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup_switch_prob',
        type=float,
        default=0.5,
        help=
        'Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument(
        '--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/your/data/path/',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--eval_data_path',
        default=None,
        type=str,
        help='dataset path for evaluation')
    parser.add_argument(
        '--nb_classes',
        default=400,
        type=int,
        help='number of the classification types')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sample', default=False, action='store_true')
    parser.add_argument(
        '--data_set',
        default='killshot',
        type=str,
        help='dataset')
    parser.add_argument(
        '--pred_video',
        default='',
        type=str,
        help='which video to load and pred')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:05}.jpg',
        type=str,
        help='filename_tmpl for rawframe dataset')
    parser.add_argument(
        '--start_idx',
        default=1,
        type=int,
        help='start_idx for rwaframe dataset')

    parser.add_argument(
        '--output_dir',
        default='',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        '--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument(
        '--validation', action='store_true', help='Perform validation only')
    parser.add_argument(
        '--dist_eval',
        action='store_true',
        default=False,
        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    parser.add_argument(
        '--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    if args.eval == False and args.pred_video == False:
        raise ValueError("Need to set args.eval or args.pred_video")

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.nb_classes = 1

    if not args.pred_video:
        dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)


    model = create_model(
        args.model,
        pred_each_frame=True,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )
   

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])

    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[
                    k].shape != state_dict[k].shape:
                if checkpoint_model[k].shape[
                        0] == 710 and args.data_set.startswith('Kinetics'):
                    print(f'Convert K710 head to {args.data_set} head')
                    if args.data_set == 'Kinetics-400':
                        label_map_path = 'misc/label_710to400.json'
                    elif args.data_set == 'Kinetics-600':
                        label_map_path = 'misc/label_710to600.json'
                    elif args.data_set == 'Kinetics-700':
                        label_map_path = 'misc/label_710to700.json'

                    label_map = json.load(open(label_map_path))
                    checkpoint_model[k] = checkpoint_model[k][label_map]
                else:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                 (args.num_frames // model.patch_embed.tubelet_size))**0.5)
            # height (== width) for the new position embedding
            new_size = int(
                (num_patches //
                 (args.num_frames // model.patch_embed.tubelet_size))**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, args.num_frames // model.patch_embed.tubelet_size,
                    orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                                                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, args.num_frames // model.patch_embed.tubelet_size,
                    new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        elif args.input_size != 224:
            pos_tokens = model.pos_embed
            org_num_frames = 16
            T = org_num_frames // args.tubelet_size
            P = int((pos_tokens.shape[1] // T)**0.5)
            C = pos_tokens.shape[2]
            new_P = args.input_size // patch_size[0]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
            pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_P, new_P),
                mode='bicubic',
                align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3,
                                            1).reshape(-1, T, new_P, new_P, C)
            pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
            model.pos_embed = pos_tokens  # update
        if args.num_frames != 16:
            org_num_frames = 16
            T = org_num_frames // args.tubelet_size
            pos_tokens = model.pos_embed
            new_T = args.num_frames // args.tubelet_size
            P = int((pos_tokens.shape[1] // T)**0.5)
            C = pos_tokens.shape[2]
            pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
            pos_tokens = pos_tokens.permute(0, 2, 3, 4,
                                            1).reshape(-1, C, T)  # BHW,C,T
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_T, mode='linear')
            pos_tokens = pos_tokens.reshape(1, P, P, C,
                                            new_T).permute(0, 4, 1, 2, 3)
            pos_tokens = pos_tokens.flatten(1, 3)
            model.pos_embed = pos_tokens  # update

        utils.load_state_dict(
            model, checkpoint_model, prefix=args.model_prefix)

    #breakpoint()
    model.to(device)

    preds_file = os.path.join(args.output_dir, '0.txt')
    
    if args.eval:
        acc, distance_from_correct, kth_guess = final_test_killshot(data_loader_test, model, device, preds_file)

        log_stats = {
            "accuracy": acc,
            "distance_from_correct_avg": np.mean(distance_from_correct),
            "guesses_behind_correct_avg": np.mean(kth_guess),
        }
        for k,v in log_stats.items():
            print(k,v)
        with open(
            os.path.join(args.output_dir, "test_results.txt"),
            mode="a",
            encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        exit(0)


    def top_k_indices(arr, k): return arr.argsort()[-k:][::-1]

    if args.pred_video:
        video_loaded = VideoLabelingDataset(args.pred_video)
        out_preds = predict_killshot_scores(video_loaded, model, device, preds_file)
        #breakpoint()
        print(top_k_indices(out_preds, 20))
        np.savetxt(os.path.join(args.pred_video, "preds.txt"), out_preds, delimiter=',')

        exit(0)


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
