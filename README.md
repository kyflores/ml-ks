# ml-ks
Machine Learning Kitchen Sink

## Mamba/Conda environment
This config assumes a CUDA system. When installing on ROCm you'll
need different pytorch packages, and possibly a different deepspeed.
```
mamba create -n mlks \
    python=3.11 \
    pip \
    pytorch \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    tqdm \
    torchinfo \
    kornia \
    jupyterlab \
    ipywidgets \
    onnx \
    -c pytorch -c nvidia -c conda-forge
```
