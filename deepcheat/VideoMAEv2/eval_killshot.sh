#!/bin/bash

python eval_killshot_pred.py \
        --model vit_base_patch16_224 \
        --data_set killshot \
        --nb_classes 1 \
        --data_path /home/waldo/code/models/VideoMAEv2/csg_processed2 \
		--data_root /home/waldo/code/models/VideoMAEv2/csg_processed2 \
        --finetune ./killshot2/checkpoint-999.pth \
        --log_dir killshot_test \
        --output_dir killshot_test \
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
        --warmup_epochs 50 \
        --epochs 1000 \
        --test_num_segment 5 \
        --test_num_crop 3 \
		--eval
		
		
	