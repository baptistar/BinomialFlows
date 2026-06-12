# ruff: noqa
# fmt: off
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import math
import hashlib
import struct
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    w_euler=False, w_linear_schedule=False, w_float32=False
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    dtype = torch.float64 if not w_float32 else torch.float32

    # Time step discretization.
    if not w_linear_schedule:
        step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    else:
        t_steps = torch.linspace(sigma_max, sigma_min, num_steps, dtype=dtype, device=latents.device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily. NOTE this is a no-op by default
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0  # NOTE 0 by default
        t_hat = net.round_sigma(t_cur + gamma * t_cur)  # NOTE same as t_cur by default
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)  # NOTE same as x_cur by default

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(dtype)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if not w_euler:
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Rau-leaping sampler.

def tau_leap_pf_sampler(
    net, batch_size, class_labels=None,
    num_steps=1024, grid='sigma', time_input='sigma', out_0_1=False,
    log_noise_eps=None,
    base_seed=None,
):
    assert grid in ['sigma', 't']
    assert time_input in ['sigma', 't']

    device = next(net.parameters()).device
    eps = float(log_noise_eps if log_noise_eps is not None else getattr(net, 'log_noise_eps', 1e-5))
    if not (eps > 0):
        raise ValueError(f'log_noise_eps must be > 0, got {eps}')

    B = int(batch_size)
    x_shape = [B, net.img_channels, net.img_resolution, net.img_resolution]
    is_cat = bool(getattr(net, 'is_pf_cat', False))
    x = torch.zeros(x_shape, device=device, dtype=(torch.int64 if is_cat else torch.float32))

    if base_seed is not None:
        base_seed = int(base_seed) % (1 << 32)
        torch.manual_seed(base_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(base_seed)

    # Precompute global time grid (shared across seeds).
    if grid == 'sigma':
        sigma_max = float(-math.log(eps))  # corresponds to t=0
        sigma_steps = torch.linspace(sigma_max, 0.0, int(num_steps) + 1, device=device, dtype=torch.float32)
        t_steps = torch.clamp(torch.exp(-sigma_steps) - eps, 0.0, 1.0 - eps)
    else:
        t_steps = torch.linspace(0.0, 1.0 - eps, int(num_steps) + 1, device=device, dtype=torch.float32)
        sigma_steps = -torch.log(t_steps + eps)

    scale = 1.0 if is_cat else (255.0 if out_0_1 else 1.0)
    # xlogx log-rate nets (PFLogRatePrecond) parametrize λ̂ directly; recovering
    # it from the D_x output would clamp at 255 and truncate large intensities.
    is_log_rate = (not is_cat) and hasattr(net, 'lambda_clip')
    for i in range(int(num_steps)):
        sigma_cur = sigma_steps[i]
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur

        t_in = sigma_cur if time_input == 'sigma' else t_cur
        t_in_vec = t_in.expand([B])
        if is_cat:
            with torch.no_grad():
                x_1 = net(x, t_in_vec, class_labels)
            g = (x_1 * scale).to(torch.float32).clamp(0.0, 255.0)
            x_f = x.to(torch.float32)
            denom = torch.clamp(1.0 - t_cur, min=1e-8)
            lam = (g - x_f) / denom
            rate = torch.clamp(lam * dt, min=0.0)
            dx = torch.poisson(rate).to(torch.int64)
            x = (x + dx).clamp(0, 255)
        elif is_log_rate:
            with torch.no_grad():
                log_lam = net(x, t_in_vec, class_labels, return_log_rate=True)
            lam = torch.exp(log_lam.to(torch.float32))
            rate = torch.clamp(lam * dt, min=0.0)
            dx = torch.poisson(rate)
            x = (x + dx).clamp(0.0, 255.0)
        else:
            with torch.no_grad():
                x_1 = net(x, t_in_vec, class_labels)
            g = (x_1 * scale).to(torch.float32).clamp(0.0, 255.0)

            denom = torch.clamp(1.0 - t_cur, min=1e-8)
            lam = (g - x) / denom
            rate = torch.clamp(lam * dt, min=0.0)
            dx = torch.poisson(rate)
            x = (x + dx).clamp(0.0, 255.0)

    return x.to(torch.float32)

#----------------------------------------------------------------------------
# ItDPDM sampler with increasing SNR schedule.
# Supports the paper's full resampling update and an incremental Poisson update
# that better matches the forward process when the network predicts x directly.

def itp_sampler(
    net, batch_size, class_labels=None,
    num_steps=100, alpha_min=-13.0, alpha_max=6.0,
    init_mode='zero', update_mode='incremental',
    base_seed=None,
):
    if not (float(alpha_max) > float(alpha_min)):
        raise ValueError(f'alpha_max must be greater than alpha_min, got {alpha_min} and {alpha_max}')
    if init_mode not in ['mean', 'zero']:
        raise ValueError(f'Unknown ITP init mode: {init_mode}')
    if update_mode not in ['incremental', 'resample']:
        raise ValueError(f'Unknown ITP update mode: {update_mode}')

    device = next(net.parameters()).device
    B = int(batch_size)
    x_shape = [B, net.img_channels, net.img_resolution, net.img_resolution]

    if base_seed is not None:
        base_seed = int(base_seed) % (1 << 32)
        torch.manual_seed(base_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(base_seed)

    # SNR schedule: γ increasing from small to large.
    alphas = torch.linspace(float(alpha_min), float(alpha_max), int(num_steps) + 1, device=device, dtype=torch.float32)
    gammas = torch.exp(alphas)

    if init_mode == 'zero':
        z = torch.zeros(x_shape, device=device, dtype=torch.float32)
    else:
        # Match the lowest-SNR marginal more closely than a hard zero start.
        z = torch.poisson(torch.full(x_shape, gammas[0].item() * net.mean_data, device=device, dtype=torch.float32))

    for i in range(int(num_steps)):
        gamma_cur_scalar = gammas[i]
        gamma_cur = gamma_cur_scalar.expand([B])
        gamma_next = gammas[i + 1]
        with torch.no_grad():
            x_hat = net(z, gamma_cur, class_labels)
        x_hat = x_hat.to(torch.float32).clamp(min=0.0)
        if update_mode == 'resample':
            # This matches the paper's full resampling step, but can be brittle
            # for an MSE denoiser because x_hat is a posterior mean.
            z = torch.poisson(gamma_next * x_hat)
        else:
            # For z_γ | x ~ Poisson(γ x), the forward process has independent
            # Poisson increments, so keep the current count state and only add
            # the missing mass for Δγ.
            delta_gamma = torch.clamp(gamma_next - gamma_cur_scalar, min=0.0)
            z = z + torch.poisson(delta_gamma * x_hat)

    # Final denoiser prediction at highest SNR.
    gamma_final = gammas[-1].expand([B])
    with torch.no_grad():
        x_hat = net(z, gamma_final, class_labels)
    x_hat = x_hat.to(torch.float32).clamp(0.0, 255.0)
    return x_hat

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the outputs', metavar='DIR',                        type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=512, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

@click.option('--w_euler',                 help='Use Euler in edm ode flow',                                        is_flag=True)
@click.option('--w_linear_schedule',       help='Use linear schedule in edm ode flow',                              is_flag=True)
@click.option('--w_float32',               help='Use float32 in edm ode flow',                                      is_flag=True)

# Sampler selection.
@click.option('--sampler',                 help='Sampling method', metavar='edm|ablation|tau|itp',             type=click.Choice(['edm', 'ablation', 'tau', 'itp']), default='edm', show_default=True)

# Poisson–Föllmer tau-leaping options.
@click.option('--tau-steps', 'tau_steps',  help='Number of tau-leaping steps (PF)', metavar='INT',                  type=click.IntRange(min=1), default=1024, show_default=True)
@click.option('--grid',                    help='Tau-leaping grid (PF)', metavar='sigma|t',                         type=click.Choice(['sigma', 't']), default='sigma', show_default=True)
@click.option('--time-input', 'time_input', help='Time input to the network (PF)', metavar='sigma|t',               type=click.Choice(['sigma', 't']), default='sigma', show_default=True)
@click.option('--out-0-1/--no-out-0-1', 'out_0_1', help='Interpret network output as [0,1] (PF)',                   default=False, show_default=True)

# ItDPDM sampling options.
@click.option('--itp-steps', 'itp_steps',  help='Number of ITP sampling steps', metavar='INT',                  type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--itp-alpha-min',           help='ITP min log-SNR', metavar='FLOAT',                             type=float, default=-13.0, show_default=True)
@click.option('--itp-alpha-max',           help='ITP max log-SNR', metavar='FLOAT',                             type=float, default=6.0, show_default=True)
@click.option('--itp-init', 'itp_init',    help='ITP initial count state', metavar='mean|zero',                 type=click.Choice(['mean', 'zero']), default='zero', show_default=True)
@click.option('--itp-update', 'itp_update', help='ITP state update rule', metavar='incremental|resample',       type=click.Choice(['incremental', 'resample']), default='incremental', show_default=True)

# Output format options.
@click.option('--save-format', 'save_format', help='Output format', metavar='png|npd',                             type=click.Choice(['png', 'npd']), default='png', show_default=True)
@click.option('--npd-array-key', 'npd_array_key', help='Array key name for images in .npd', metavar='STR',        type=str, default='images', show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), save_format='png', npd_array_key='images', **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)

    # Select sampler & split kwargs.
    sampler = sampler_kwargs.pop('sampler', 'auto')
    tau_steps = sampler_kwargs.pop('tau_steps', 1024)
    tau_grid = sampler_kwargs.pop('grid', 'sigma')
    tau_time_input = sampler_kwargs.pop('time_input', 'sigma')
    tau_out_0_1 = sampler_kwargs.pop('out_0_1', False)
    itp_steps = sampler_kwargs.pop('itp_steps', 100)
    itp_alpha_min = sampler_kwargs.pop('itp_alpha_min', -13.0)
    itp_alpha_max = sampler_kwargs.pop('itp_alpha_max', 6.0)
    itp_init = sampler_kwargs.pop('itp_init', 'zero')
    itp_update = sampler_kwargs.pop('itp_update', 'incremental')

    if sampler in ['edm', 'ablation'] and not hasattr(net, 'sigma_min'):
        raise click.ClickException(f'Network does not appear to support EDM-style sampling (missing sigma_min/sigma_max); use --sampler=tau for PF models or --sampler=itp for ITP models.')

    all_sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in all_sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])

    edm_keys = {
        'num_steps', 'sigma_min', 'sigma_max', 'rho',
        'S_churn', 'S_min', 'S_max', 'S_noise',
        'w_euler', 'w_linear_schedule', 'w_float32',
    }
    ablation_keys = {
        'num_steps', 'sigma_min', 'sigma_max', 'rho',
        'solver', 'discretization', 'schedule', 'scaling',
        'epsilon_s', 'C_1', 'C_2', 'M', 'alpha',
        'S_churn', 'S_min', 'S_max', 'S_noise',
    }
    edm_kwargs = {k: v for k, v in all_sampler_kwargs.items() if k in edm_keys}
    ablation_kwargs = {k: v for k, v in all_sampler_kwargs.items() if k in ablation_keys}

    if sampler == 'edm':
        sampler_fn = edm_sampler
        sampler_call_kwargs = edm_kwargs
    elif sampler == 'ablation':
        sampler_fn = ablation_sampler
        sampler_call_kwargs = ablation_kwargs
    elif sampler == 'auto' and have_ablation_kwargs:
        sampler_fn = ablation_sampler
        sampler_call_kwargs = ablation_kwargs
    else:
        sampler_fn = edm_sampler
        sampler_call_kwargs = edm_kwargs

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} samples to "{outdir}" as {save_format}...')

    # If saving to .npd, determine aggregate path (rank 0 will write it at the end).
    out_npd_path = None
    if save_format == 'npd':
        if str(outdir).lower().endswith('.npd'):
            out_npd_path = outdir
            outdir_for_file = os.path.dirname(out_npd_path)
            npd_base_dir = outdir_for_file if outdir_for_file != '' else '.'
        else:
            outdir_for_file = outdir
            out_npd_path = os.path.join(outdir_for_file, 'samples.npd')
            npd_base_dir = outdir
        if dist.get_rank() == 0 and outdir_for_file:
            os.makedirs(outdir_for_file, exist_ok=True)
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick labels (and latents for continuous samplers).
        rnd = StackedRandomGenerator(device, batch_seeds)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        if sampler == 'itp':
            seed_list = [int(s) for s in batch_seeds.detach().cpu().tolist()]
            h = hashlib.blake2b(digest_size=4)
            h.update(struct.pack('<I', int(dist.get_rank()) % (1 << 32)))
            for s in seed_list:
                h.update(struct.pack('<I', int(s) % (1 << 32)))
            base_seed = int.from_bytes(h.digest(), byteorder='little', signed=False)
            images = itp_sampler(
                net,
                batch_size=batch_size,
                class_labels=class_labels,
                num_steps=itp_steps,
                alpha_min=itp_alpha_min,
                alpha_max=itp_alpha_max,
                init_mode=itp_init,
                update_mode=itp_update,
                base_seed=int(base_seed),
            )
        elif sampler == 'tau':
            # Batch-level RNG seed: deterministic for a given seed list and rank,
            # but not invariant to different batching/grouping of --seeds.
            seed_list = [int(s) for s in batch_seeds.detach().cpu().tolist()]
            h = hashlib.blake2b(digest_size=4)
            h.update(struct.pack('<I',int(dist.get_rank()) % (1 << 32)))
            for s in seed_list:
                h.update(struct.pack('<I', int(s) % (1 << 32)))
            base_seed = int.from_bytes(h.digest(), byteorder='little', signed=False)
            images = tau_leap_pf_sampler(
                net,
                batch_size=batch_size,
                class_labels=class_labels,
                num_steps=tau_steps,
                grid=tau_grid,
                time_input=tau_time_input,
                out_0_1=tau_out_0_1,
                base_seed=int(base_seed),
            )
        else:
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_call_kwargs)

        # Save outputs in requested format.
        if save_format == 'png':
            if sampler in ('tau', 'itp'):
                images_np = images.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            else:
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                seed = int(seed)
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        else:
            # Save each sample as an .npd file (np.savez) with float data in NCHW.
            images_cpu = images.to(torch.float32).cpu().numpy()  # NCHW floats
            for i, seed in enumerate(batch_seeds):
                seed = int(seed)
                image_dir = os.path.join(npd_base_dir, f'{seed-seed%1000:06d}') if subdirs else npd_base_dir
                os.makedirs(image_dir, exist_ok=True)
                npd_path = os.path.join(image_dir, f'{seed:06d}.npd')
                # Use file handle to avoid numpy adding '.npz' suffix.
                with open(npd_path, 'wb') as f:
                    np.savez(f, **{npd_array_key: images_cpu[i:i+1]})

    # If saving to a single .npd, write the file on rank 0.
    if save_format == 'npd':
        # After all per-sample .npd files are written, aggregate into one .npd on rank 0.
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            C = net.img_channels
            H = net.img_resolution
            W = net.img_resolution
            N = len(seeds)
            full_images = np.empty((N, C, H, W), dtype=np.float32)
            for i, seed in enumerate(seeds):
                image_dir = os.path.join(npd_base_dir, f'{seed-seed%1000:06d}') if subdirs else npd_base_dir
                npd_path = os.path.join(image_dir, f'{seed:06d}.npd')
                # if not os.path.isfile(npd_path):
                #     # Fallback for any previously created files with tensor-style names.
                #     alt_paths = [
                #         os.path.join(image_dir, f'tensor({seed}).npd'),
                #         npd_path + '.npz',
                #     ]
                #     for ap in alt_paths:
                #         if os.path.isfile(ap):
                #             npd_path = ap
                #             break
                #     if not os.path.isfile(npd_path):
                #         raise FileNotFoundError(f'Could not find per-sample .npd for seed {seed} at {npd_path} or fallbacks')
                with np.load(npd_path, allow_pickle=False) as data:
                    imgs = data[npd_array_key]
                    # Expect shape (1, C, H, W); accept (C,H,W) as fallback.
                    if imgs.ndim == 3:
                        full_images[i] = imgs.astype(np.float32)
                    else:
                        full_images[i] = imgs[0].astype(np.float32)
            # Use file handle to keep '.npd' extension.
            with open(out_npd_path, 'wb') as f:
                np.savez(f, **{npd_array_key: full_images})
            dist.print0(f'Saved aggregated {N} samples to "{out_npd_path}"')
        torch.distributed.barrier()

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
