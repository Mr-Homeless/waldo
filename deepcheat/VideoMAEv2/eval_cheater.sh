#!/bin/bash

python train_cheater_pred.py \
        --model vit_giant_patch14_224 \
        --data_set cheater \
        --nb_classes 1 \
        --finetune ./output/checkpoint-99.pth \
        --batch_size 8 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 200 \
        --num_frames 16 \
        --sampling_rate 1 \
        --num_sample 1 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 40 \
        --epochs 1000 \
        --test_num_segment 1 \
        --test_num_crop 1 \
		--eval \
        --min_eval_score -1.9871155 \
        --max_eval_score 2.4927201 \
        --output_dir $1 \
        --data_path $1 \
		--data_root $1 
        
		
		
	