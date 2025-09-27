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
from functools import partial
from pathlib import Path
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import ToPILImage

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from packaging import version
from timm.models import create_model

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
from dataset import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate
from einops import rearrange
import cv2


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE v2 pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='pretrain_videomae_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)

    parser.add_argument(
        '--decoder_depth', default=4, type=int, help='depth of decoder')

    parser.add_argument(
        '--mask_type',
        default='tube',
        choices=['random', 'tube'],
        type=str,
        help='encoder masked strategy')
    parser.add_argument(
        '--decoder_mask_type',
        default='run_cell',
        choices=['random', 'run_cell'],
        type=str,
        help='decoder masked strategy')

    parser.add_argument(
        '--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument(
        '--decoder_mask_ratio',
        default=0.0,
        type=float,
        help='mask ratio of decoder')

    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='images input size for backbone')

    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')

    parser.add_argument(
        '--normlize_target',
        default=True,
        type=bool,
        help='normalized the target patch pixels')

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
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)"""
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1.5e-4,
        metavar='LR',
        help='learning rate (default: 1.5e-4)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=40,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        choices=['random', 'bilinear', 'bicubic'],
        help='Training interpolation')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/your/data/annotation/path',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:010}.jpg',
        type=str,
        help='filename_tmpl for rawframe data')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=1)
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

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no_pin_mem', action='store_false', dest='pin_mem', help='')
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

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        decoder_depth=args.decoder_depth,
        with_cp=args.with_checkpoint)

    if version.parse(torch.__version__) > version.parse('1.13.1'):
        torch.set_float32_matmul_precision('high')
        #model = torch.compile(model)

    return model

@torch.no_grad()
def main(args):

    print(args)
    device = torch.device(args.device)
    seed = args.seed
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args,test_mode=True)

    total_batch_size = args.batch_size 

    log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True)

    if not args.finetune: raise ValueError("must provide args.finetune filepath")
    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load ckpt from %s" % args.finetune)
    #breakpoint()
    checkpoint_model = None
    for model_key in ['model', 'module']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    if "/" not in args.finetune:
        modified_dict = {f'encoder.{key}': value for key, value in checkpoint_model.items()}
        modified_dict['encoder.norm.bias'] = checkpoint_model['fc_norm.bias']
        modified_dict['encoder.norm.weight'] = checkpoint_model['fc_norm.weight']
        checkpoint_model = modified_dict

    utils.load_state_dict(model, checkpoint_model)
    model.to(device)


    start_time = time.time()
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

    get_test_loss=False
    total_loss = 0
    for step, batch in enumerate(data_loader_test):
        

        # NOTE: When the decoder mask ratio is 0,
        # in other words, when decoder masking is not used,
        # decode_masked_pos = ~bool_masked_pos
        images, bool_masked_pos, decode_masked_pos = batch

        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        if get_test_loss:
            unnorm_images = images * std + mean  # in [0, 1]

            if args.normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size[0],
                    p2=patch_size[0])
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)

            B, N, C = images_patch.shape
            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)
            #breakpoint()
            #if loss_scaler is None:
            outputs = model(images, bool_masked_pos, decode_masked_pos)

            loss = (outputs - labels)**2
            loss = loss.mean(dim=-1)
            cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
            loss = (loss * cal_loss_mask).sum()
            total_loss+=loss.item()
        else:
            break

    if get_test_loss:
        print("total_loss ", total_loss)
        exit()


    #video_data = vr.get_batch(frame_id_list).asnumpy()
    #print(video_data.shape)
    i = 1
    img = images[i].unsqueeze(0)
    bool_masked_pos =  bool_masked_pos[i].unsqueeze(0)
    decode_masked_pos = decode_masked_pos[i].unsqueeze(0)
    #transforms = DataAugmentationForVideoMAE(args)
    #img, bool_masked_pos = transforms((img, None)) # T*C,H,W
    # print(img.shape)
    #img = img.view((16 , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
    # img = img.view(( -1 , args.num_frames) + img.size()[-2:]) 
    #bool_masked_pos = torch.from_numpy(bool_masked_pos)

    # img = img[None, :]
    # bool_masked_pos = bool_masked_pos[None, :]
    img = img
    print(img.shape)
    #bool_masked_pos = bool_masked_pos[0].unsqueeze(0)
    args.save_path = "out_imgs"
    
    #img = img.to(device, non_blocking=True)
    #bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
    #outputs = model(img, bool_masked_pos)
    outputs = model(img, bool_masked_pos, decode_masked_pos)


    tmp = np.arange(0,32, 2) + 60
    frame_id_list = tmp.tolist()
    #save original video
    
    ori_img = img * std + mean  # in [0, 1]
    #breakpoint()
    imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for video
    frame_size = (imgs[0].size[0], imgs[0].size[1])  # assuming all images have same size
    fps = 3  # frames per second
    out = cv2.VideoWriter(f"{args.save_path}/ori.mp4", fourcc, fps, frame_size)

    # Save each image and write to video
    for id, im in enumerate(imgs):
        im.save(f"{args.save_path}/ori_img{id}.jpg")
        
        # Convert PIL image to NumPy array (OpenCV uses BGR instead of RGB)
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame)

    # Release the video writer
    out.release()

    img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch[bool_masked_pos] = outputs

    #make mask
    mask = torch.ones_like(img_patch)
    mask[bool_masked_pos] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

    #save reconstruction video
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
    rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
    imgsr = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]

    out = cv2.VideoWriter(f"{args.save_path}/rec.mp4", fourcc, fps, frame_size)

    # Save each image and write to video
    for id, im in enumerate(imgsr):
        im.save(f"{args.save_path}/rec_img{id}.jpg")
        
        # Convert PIL image to NumPy array (OpenCV uses BGR instead of RGB)
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame)

    # Release the video writer
    out.release()

    #save masked video 
    img_mask = rec_img * mask
    imgsm = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
    out = cv2.VideoWriter(f"{args.save_path}/mask.mp4", fourcc, fps, frame_size)

    # Save each image and write to video
    for id, im in enumerate(imgsm):
        im.save(f"{args.save_path}/mask_img{id}.jpg")
        
        # Convert PIL image to NumPy array (OpenCV uses BGR instead of RGB)
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame)

    # Release the video writer
    out.release()
    single_frame_size = (imgs[0].size[0], imgs[0].size[1]) 
    frame_size = (single_frame_size[0] * 3, single_frame_size[1])

    out = cv2.VideoWriter(f"{args.save_path}/results.mp4", fourcc, fps, frame_size)

    # Save each image and write to video
    for id, (im, im_left, im_right) in enumerate(zip( imgsr,imgs, imgsm)):
        
        # Convert PIL image to NumPy array (OpenCV uses BGR instead of RGB)
        frame = cv2.cvtColor(np.hstack((np.array(im_left), np.array(im), np.array(im_right))), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame)

    # Release the video writer
    out.release()



if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
