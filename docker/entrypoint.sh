#!/bin/bash

if [ -n "$MODEL_PATH" ]; then
    ln -s "$MODEL_PATH/vit_g_ps14_ak_ft_ckpt_7_clean.pth" /workspace/deepcheat/VideoMAEv2/vit_g_ps14_ak_ft_ckpt_7_clean.pth
fi

echo "Running with model..."
exec python main.py
