# BinomialFlows

This package implements the numerical experiments available at the paper

## Authors:

Yair Shenfeld, Ricardo Baptista, Stefano Peluchetti

Download requirements.txt and run the following scripts:

For synthetic data, run:
python train_model.py; python post_process.py

For music, run:
python train.py; python post_process.py --model_run './runs/run_folder'

# PF with MSE loss (default), 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --precond=pf \
    --arch=ddpmpp

# PF with Bregman/xlogx loss (requires --no-edm)
torchrun --standalone --nproc_per_node=8 train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --precond=pf --pf-loss=xlogx --no-edm \
    --arch=ddpmpp

# PF categorical (cross-entropy over {0,...,255})
torchrun --standalone --nproc_per_node=8 train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --precond=pf-cat \
    --arch=ddpmpp

The data statistics defaults (--pf-mean-data=120.7 --pf-var-data=4115.2) are set for CIFAR-10 in [0,255]. Change these for other datasets.

---
Generation (tau-leaping sampler)

python generate.py \
    --network=training-runs/.../network-snapshot-*.pkl \
    --outdir=out \
    --seeds=0-49999 \
    --sampler=tau \
    --tau-steps=1024

Save as .npd float arrays instead of PNG (useful for FID on count data):

python generate.py --network=... --outdir=samples.npd \
    --seeds=0-49999 --sampler=tau --tau-steps=1024 \
    --save-format=npd

---
Likelihood (bits/dim via Bregman divergence MC estimator)

# Single GPU
python likelihood.py --network=training-runs/.../network-snapshot-*.pkl

# Multi-GPU
torchrun --standalone --nproc_per_node=4 likelihood.py \
    --network=training-runs/.../network-snapshot-*.pkl \
    --mc=256 --batch=64

This runs on the CIFAR-10 test set (auto-downloads if needed) and reports NLL in nats and BPD.

FID

# Generate 50k samples first, then compute FID
torchrun --standalone --nproc_per_node=8 generate.py \
    --network=... --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --sampler=tau --tau-steps=1024

torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

