# ruff: noqa
# fmt: off
"""Estimate log-likelihood (bits/dim) of CIFAR-10 test images under a trained
Poisson–Föllmer model using the Bregman-divergence Monte Carlo estimator.

The negative log-likelihood of a data point x ∈ ℕ^d satisfies

    -log p(x) ≤ ∫₀¹ 𝔼_{y∣x,t}[ Σ_i D_{x log x}(λᵢ(x,y,t) ∥ λ̂ᵢ(y,t)) ] dt

where y ~ Binomial(x, t) is the Poisson–Föllmer bridge sample,
λᵢ = (xᵢ - yᵢ)/(1-t) the true bridge intensity,
λ̂ᵢ = (D(y,t)ᵢ - yᵢ)/(1-t) the model intensity, and
D_{x log x}(p ∥ q) = p log(p/q) - p + q is the Bregman divergence of x log x.

Equality holds for the true data-generating (Föllmer) process, because
its bridge conditioned on X_T = x is exactly the Poisson bridge (by the
h-transform construction). For a trained model the gap is
KL(Poisson bridge ∥ model bridge | X_T = x), which vanishes as the
model approaches the Föllmer drift.

Usage:

    # Single GPU
    python likelihood.py --network=model.pkl

    # Multi-GPU
    torchrun --standalone --nproc_per_node=4 likelihood.py --network=model.pkl
"""

import os
import math
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist


def _load_cifar10_test(data_dir, rank=0):
    """Load CIFAR-10 test batch from raw pickles (no torchvision needed)."""
    import tarfile, urllib.request
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    test_file = os.path.join(cifar_dir, 'test_batch')
    if not os.path.isfile(test_file):
        if rank == 0:
            os.makedirs(data_dir, exist_ok=True)
            tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
            if not os.path.isfile(tar_path):
                url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                print(f'Downloading CIFAR-10 to "{tar_path}"...')
                urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, 'r:gz') as tf:
                tf.extractall(data_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = torch.from_numpy(batch[b'data'].reshape(-1, 3, 32, 32).copy()).to(torch.float32)
    labels = torch.tensor(batch[b'labels'], dtype=torch.int64)
    return images, labels


def bregman_divergence_xlogx(p, q, eps=1e-12):
    """D_{x log x}(p ∥ q) = p log(p/q) - p + q, element-wise."""
    q_safe = torch.clamp(q, min=eps)
    log_term = torch.zeros_like(p)
    mask = p > 0
    log_term[mask] = p[mask] * torch.log(p[mask] / q_safe[mask])
    return log_term - p + q


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle', metavar='PATH', type=str, required=True)
@click.option('--data', 'data_dir', help='CIFAR-10 data root (auto-downloads if needed)', metavar='DIR', type=str, default='~/datasets', show_default=True)
@click.option('--batch', 'batch_size', help='Images per forward pass', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--mc', 'num_mc', help='Monte Carlo samples per image', metavar='INT', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--limit', help='Evaluate only the first N images', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--outdir', help='Save per-image NLL/BPD arrays here', metavar='DIR', type=str, default=None)
@click.option('--seed', help='Global RNG seed', metavar='INT', type=int, default=0, show_default=True)
def main(network_pkl, data_dir, batch_size, num_mc, limit, outdir, seed):
    dist.init()
    device = torch.device('cuda')
    torch.manual_seed(seed)

    # Load model.
    dist.print0(f'Loading network from "{network_pkl}"...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    eps = float(getattr(net, 'log_noise_eps', 1e-5))
    t_max = 1.0 - eps  # upper integration bound

    # Load CIFAR-10 test set (no torchvision dependency).
    images_all, labels_all = _load_cifar10_test(os.path.expanduser(data_dir), rank=dist.get_rank())
    torch.distributed.barrier()
    if limit is not None:
        images_all = images_all[:limit]
        labels_all = labels_all[:limit]
    N = len(images_all)
    dim = images_all[0].numel()  # 3072

    # Split images across ranks.
    indices = list(range(N))
    rank_indices = indices[dist.get_rank()::dist.get_world_size()]

    dist.print0(f'Images: {N}  MC samples: {num_mc}  batch: {batch_size}  GPUs: {dist.get_world_size()}')

    nll_local = torch.zeros(len(rank_indices), dtype=torch.float64)
    se_local  = torch.zeros(len(rank_indices), dtype=torch.float64)

    for local_start in tqdm.tqdm(range(0, len(rank_indices), batch_size),
                                  disable=(dist.get_rank() != 0)):
        local_end = min(local_start + batch_size, len(rank_indices))
        batch_idx = rank_indices[local_start:local_end]
        x = images_all[batch_idx].to(device)
        B = x.shape[0]

        # Class labels (one-hot) for conditional models.
        class_labels = None
        if net.label_dim:
            lab = labels_all[batch_idx].to(device)
            class_labels = torch.zeros(B, net.label_dim, device=device)
            class_labels.scatter_(1, lab.unsqueeze(1), 1.0)

        # Stratified MC with replicated rounds for correct SE estimation.
        # R independent rounds, each with K/R stratified bins covering [0, t_max].
        # SE is computed from variance across rounds.
        num_rounds = min(8, num_mc)
        bins_per_round = num_mc // num_rounds
        bin_width = t_max / bins_per_round
        round_nll = torch.zeros(num_rounds, B, device=device, dtype=torch.float64)

        for r in range(num_rounds):
            round_sum = torch.zeros(B, device=device, dtype=torch.float64)
            for k in range(bins_per_round):
                t = (k + torch.rand(B, 1, 1, 1, device=device)) * bin_width

                # Bridge sample: y ~ Binomial(x, t).
                y = torch.distributions.Binomial(total_count=x, probs=t).sample()

                # Network input: σ = -log(t + ε).
                sigma = -torch.log(t.view(B) + eps)

                with torch.no_grad():
                    D = net(y, sigma, class_labels).to(torch.float32).clamp(0.0, 255.0)

                # True and model intensities.
                inv_gap = 1.0 / (1.0 - t)             # (B,1,1,1)
                lambda_true  = (x - y) * inv_gap
                lambda_model = (D - y) * inv_gap

                # Bregman divergence summed over all pixels.
                div = bregman_divergence_xlogx(lambda_true, lambda_model)
                round_sum += div.sum(dim=(1, 2, 3)).to(torch.float64)

            round_nll[r] = round_sum / bins_per_round * t_max

        nll_local[local_start:local_end] = round_nll.mean(dim=0).cpu()
        se_local[local_start:local_end]  = (round_nll.std(dim=0) / math.sqrt(num_rounds)).cpu()

    # Gather results from all ranks.
    nll_all_gpu = torch.zeros(N, dtype=torch.float64, device=device)
    se_all_gpu  = torch.zeros(N, dtype=torch.float64, device=device)
    for i, global_idx in enumerate(rank_indices):
        nll_all_gpu[global_idx] = nll_local[i].to(device)
        se_all_gpu[global_idx]  = se_local[i].to(device)
    torch.distributed.all_reduce(nll_all_gpu)
    torch.distributed.all_reduce(se_all_gpu)
    nll_all = nll_all_gpu.cpu()
    se_all  = se_all_gpu.cpu()

    bpd    = nll_all / (dim * math.log(2))
    se_bpd = se_all  / (dim * math.log(2))

    dist.print0(f'NLL:  {nll_all.mean().item():.2f} ± {1.96 * se_all.mean().item():.2f} nats')
    dist.print0(f'BPD:  {bpd.mean().item():.4f} ± {1.96 * se_bpd.mean().item():.4f} bits/dim')

    if outdir is not None and dist.get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, 'likelihood.npz')
        np.savez(path, nll=nll_all.numpy(), bpd=bpd.numpy(), mc_se_bpd=se_bpd.numpy())
        dist.print0(f'Saved to {path}')

    torch.distributed.barrier()


if __name__ == '__main__':
    main()
