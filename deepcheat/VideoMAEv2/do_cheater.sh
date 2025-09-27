#!/bin/bash

#python -m debugpy --listen 5678 --wait-for-client \
python  \
    train_cheater_pred.py \
        --model vit_giant_patch14_224 \
        --data_set cheater \
        --nb_classes 1 \
        --data_path ~/Desktop/processed_vids \
		--data_root ~/Desktop/processed_vids \
        --finetune ../vit_g_ps14_ak_ft_ckpt_7_clean.pth \
        --log_dir output \
        --output_dir output \
        --batch_size 6 \
        --update_freq 12 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --sampling_rate 1 \
        --num_sample 1 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.1 \
        --clip_grad 1.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.000 \
        --warmup_epochs 10 \
        --epochs 100 \
        --nb_classes 1 \
        --test_num_segment 5 \
        --test_num_crop 3 $1
		
		
	
