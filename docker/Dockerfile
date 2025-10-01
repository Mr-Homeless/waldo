FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG NO_MODEL=false

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install requirements using pip with the system Python
# Fix NumPy compatibility and replace opencv-python with opencv-python-headless for Docker compatibility
RUN pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install opencv-python-headless==4.9.0.80

# Conditionally download model
RUN if [ "$NO_MODEL" != "true" ]; then \
        pip install huggingface_hub && \
        hf download jinggu/jing-model vit_g_ps14_ak_ft_ckpt_7_clean.pth --local-dir /workspace/deepcheat/VideoMAEv2; \
    fi

CMD ["python", "main.py"]