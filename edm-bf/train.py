# ruff: noqa
# fmt: off
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--sigma-data',    help='Data σ', metavar='FLOAT',                                    type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|pf|pf-cat|itp', type=click.Choice(['vp', 've', 'edm', 'pf', 'pf-cat', 'itp']), default='edm', show_default=True)
@click.option('--no-edm',        help='Disable time-dependent loss weighting + skip/output scaling (keeps input centering/scaling; uses constant output scale; PF/ITP only)', is_flag=True, default=False, show_default=True)

# Poisson–Föllmer options.
# Loss type (PF only).
@click.option('--pf-loss',               help='PF loss type (PF only). xlogx requires --no-edm', metavar='mse|xlogx', type=click.Choice(['mse', 'xlogx']), default='mse', show_default=True)
@click.option('--pf-xlogx-lambda-clip',  help='PF xlogx: clip λ_true and λ̂ at this value', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=4096.0, show_default=True)
# σ-noise
@click.option('--pf-log-noise-eps',     help='PF epsilon used in the log-noise mapping σ=-log(t+eps)', metavar='FLOAT', type=click.FloatRange(min=0), default=1e-5, show_default=True)
# σ-noise sampling
@click.option('--pf-log-noise-mu-mul',      help='PF log-noise mean multiplier: mu = lo + mu_mul*(hi-lo) in log(t+eps) space', metavar='FLOAT', type=float, default=0.5, show_default=True)
@click.option('--pf-log-noise-sigma-mul',   help='PF log-noise std multiplier: std = sigma_mul*(hi-lo) in log(t+eps) space', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.2, show_default=True)
# X_t rescaling via c_in(t) and s_in(t)
@click.option('--pf-mean-data',         help='PF mean of x_T in [0,255] (used for scaling/centering)', metavar='FLOAT', type=click.FloatRange(min=0), default=120.7, show_default=True)
@click.option('--pf-var-data',          help='PF variance of x_T in [0,255] (used for scaling/centering)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=4115.2, show_default=True)
@click.option('--pf-cin-eps',           help='PF epsilon added inside c_in denominator', metavar='FLOAT', type=click.FloatRange(min=0), default=1e-1, show_default=True)
# PF categorical model options.
@click.option('--pf-cat-embed-dim',     help='PF categorical embedding dim (per-channel)', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)

# ItDPDM options.
@click.option('--itp-mean-data',       help='ITP mean of data in [0,255]', metavar='FLOAT',              type=click.FloatRange(min=0), default=120.7, show_default=True)
@click.option('--itp-var-data',        help='ITP variance of data in [0,255]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=4115.2, show_default=True)
@click.option('--itp-cin-eps',         help='ITP epsilon in c_in denominator and loss-weight denominator floor', metavar='FLOAT',          type=click.FloatRange(min=0), default=1e-2, show_default=True)
@click.option('--itp-logistic-loc',    help='ITP Logistic location for log-SNR sampling', metavar='FLOAT', type=float, default=-6.0, show_default=True)
@click.option('--itp-logistic-scale',  help='ITP Logistic scale for log-SNR sampling', metavar='FLOAT',  type=click.FloatRange(min=0, min_open=True), default=2.0, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--optimizer',     help='Optimizer', metavar='adam|adamw',                            type=click.Choice(['adam', 'adamw']), default='adam', show_default=True)
@click.option('--adamw-weight-decay', help='AdamW weight decay', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision (float16) for model execution', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--bf16',          help='Enable mixed-precision (bfloat16) for model execution', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--grad-clip',     help='Gradient clipping max norm (0 = disable)', metavar='FLOAT',  type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# Compilation.
@click.option('--compile',       help='Enable torch.compile for the network (PyTorch 2+)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--compile-mode',  help='torch.compile mode', metavar='default|reduce-overhead|max-autotune', type=click.Choice(['default','reduce-overhead','max-autotune']), default='default', show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# Weights & Biases options.
@click.option('--wandb',         help='Enable Weights & Biases logging',                            is_flag=True, default=True)
@click.option('--wandb-project', help='W&B project name', metavar='STR',                            type=str, default='edm', show_default=True)
@click.option('--wandb-run-name',help='W&B run name  [default: auto-generated]', metavar='STR',     type=str)
@click.option('--wandb-tags',    help='W&B tags (comma-separated)', metavar='STR',                  type=str)

# Method (keep dd).
@click.option('--method',        help='Method', metavar='dd|bm|bmf',                                type=click.Choice(['dd', 'bm', 'bmf']), default='dd', show_default=True)

# Learn the derivatives related.
@click.option('--dd-jvp-weight', help='Weight for JVP loss in DD method', metavar='FLOAT',          type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--dd-time-weight',help='Weight for σ-derivative loss in DD method', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--dd-enable-jvp', help='Enable JVP loss in DD method',                               is_flag=True)
@click.option('--dd-enable-time',help='Enable σ-derivative loss in DD method',                      is_flag=True)
# This is expensive, debugging only.
@click.option('--dd-log-grad',   help='Log gradient norms of DD derivative losses',                 is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    # Choose dataset class: .npd => NpdDataset (float arrays untouched), else ImageFolderDataset.
    if isinstance(opts.data, str) and opts.data.lower().endswith('.npd'):
        c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.NpdDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    else:
        c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    if opts.optimizer == 'adamw':
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.adamw_weight_decay)
    else:
        assert opts.optimizer == 'adam'
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'pf':
        if opts.pf_loss == 'xlogx':
            if not opts.no_edm:
                raise click.ClickException('--pf-loss=xlogx requires --no-edm')
            c.network_kwargs.class_name = 'training.networks.PFLogRatePrecond'
            c.network_kwargs.no_edm = True
            c.network_kwargs.log_noise_eps = opts.pf_log_noise_eps
            c.network_kwargs.mean_data = opts.pf_mean_data
            c.network_kwargs.var_data = opts.pf_var_data
            c.network_kwargs.cin_eps = opts.pf_cin_eps
            c.network_kwargs.rate_scale = opts.pf_mean_data
            c.network_kwargs.lambda_clip = opts.pf_xlogx_lambda_clip
            c.loss_kwargs.class_name = 'training.loss.PoissonFollmerXLogXLoss'
            c.loss_kwargs.T = 1
            c.loss_kwargs.log_noise_eps = opts.pf_log_noise_eps
            c.loss_kwargs.log_noise_mu_mul = opts.pf_log_noise_mu_mul
            c.loss_kwargs.log_noise_sigma_mul = opts.pf_log_noise_sigma_mul
            c.loss_kwargs.mean_data = opts.pf_mean_data
            c.loss_kwargs.var_data = opts.pf_var_data
            c.loss_kwargs.lambda_clip = opts.pf_xlogx_lambda_clip
        else:
            assert opts.pf_loss == 'mse'
            c.network_kwargs.class_name = 'training.networks.PFPrecond'
            c.network_kwargs.log_noise_eps = opts.pf_log_noise_eps
            c.network_kwargs.mean_data = opts.pf_mean_data
            c.network_kwargs.var_data = opts.pf_var_data
            c.network_kwargs.cin_eps = opts.pf_cin_eps
            c.network_kwargs.no_edm = opts.no_edm
            c.loss_kwargs.class_name = 'training.loss.PoissonFollmerLoss'
            c.loss_kwargs.T = 1
            c.loss_kwargs.log_noise_eps = opts.pf_log_noise_eps
            c.loss_kwargs.log_noise_mu_mul = opts.pf_log_noise_mu_mul
            c.loss_kwargs.log_noise_sigma_mul = opts.pf_log_noise_sigma_mul
            c.loss_kwargs.mean_data = opts.pf_mean_data
            c.loss_kwargs.var_data = opts.pf_var_data
            c.loss_kwargs.cin_eps = opts.pf_cin_eps
            c.loss_kwargs.no_edm = opts.no_edm
    elif opts.precond == 'pf-cat':
        c.network_kwargs.class_name = 'training.networks.PFCatPrecond'
        c.network_kwargs.log_noise_eps = opts.pf_log_noise_eps
        c.network_kwargs.embed_dim = opts.pf_cat_embed_dim
        c.loss_kwargs.class_name = 'training.loss.PoissonFollmerCatLoss'
        c.loss_kwargs.T = 1
        c.loss_kwargs.log_noise_eps = opts.pf_log_noise_eps
        c.loss_kwargs.log_noise_mu_mul = opts.pf_log_noise_mu_mul
        c.loss_kwargs.log_noise_sigma_mul = opts.pf_log_noise_sigma_mul
    elif opts.precond == 'itp':
        c.network_kwargs.class_name = 'training.networks.ITPPrecond'
        c.network_kwargs.mean_data = opts.itp_mean_data
        c.network_kwargs.var_data = opts.itp_var_data
        c.network_kwargs.cin_eps = opts.itp_cin_eps
        c.network_kwargs.no_edm = opts.no_edm
        c.loss_kwargs.class_name = 'training.loss.ITPLoss'
        c.loss_kwargs.mean_data = opts.itp_mean_data
        c.loss_kwargs.var_data = opts.itp_var_data
        c.loss_kwargs.cin_eps = opts.itp_cin_eps
        c.loss_kwargs.logistic_loc = opts.itp_logistic_loc
        c.loss_kwargs.logistic_scale = opts.itp_logistic_scale
        c.loss_kwargs.no_edm = opts.no_edm
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
        c.loss_kwargs.sigma_max = 80
        c.loss_kwargs.sigma_data = opts.sigma_data
        c.loss_kwargs.method = opts.method
        c.loss_kwargs.dd_jvp_weight = opts.dd_jvp_weight
        c.loss_kwargs.dd_time_weight = opts.dd_time_weight
        c.loss_kwargs.dd_enable_jvp = opts.dd_enable_jvp
        c.loss_kwargs.dd_enable_time = opts.dd_enable_time
        c.loss_kwargs.dd_log_grad = opts.dd_log_grad

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    # Precision options.
    if opts.fp16 and opts.bf16:
        raise click.ClickException('--fp16 and --bf16 are mutually exclusive')
    amp_dtype = 'fp16' if opts.fp16 else ('bf16' if opts.bf16 else None)
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)
    if amp_dtype is not None:
        c.network_kwargs.amp_dtype = amp_dtype

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.grad_clip = opts.grad_clip
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    # torch.compile settings
    c.compile_mode = opts.compile_mode
    c.torch_compile = opts.compile
    # Derivative-based DD losses require higher-order autodiff that is not
    # currently supported by torch.compile(aot_autograd) (double backward).
    # Disable compilation automatically in that case to avoid runtime errors.
    if c.torch_compile and opts.precond == 'edm' and opts.method == 'dd' and (opts.dd_enable_jvp or opts.dd_enable_time):
        dist.print0('Disabling torch.compile because DD derivative losses require double backward not supported by aot_autograd.')
        c.torch_compile = False

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    if getattr(c.network_kwargs, 'amp_dtype', None) == 'bf16':
        dtype_str = 'bf16'
    elif c.network_kwargs.use_fp16:
        dtype_str = 'fp16'
    else:
        dtype_str = 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.no_edm and opts.precond in ['pf', 'itp']:
        desc += '-noedm'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Configure Weights & Biases.
    if opts.wandb:
        c.wandb_project = opts.wandb_project
        c.wandb_run_name = opts.wandb_run_name if opts.wandb_run_name is not None else desc
        c.wandb_tags = opts.wandb_tags.split(',') if opts.wandb_tags is not None else None

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'AMP dtype:               {dtype_str}')
    dist.print0(f'torch.compile:           {c.torch_compile} (mode={c.compile_mode})')
    dist.print0(f'Weights & Biases:        {opts.wandb}')
    if opts.wandb:
        dist.print0(f'  Project:               {c.wandb_project}')
        dist.print0(f'  Run name:              {c.wandb_run_name}')
        dist.print0(f'  Tags:                  {c.wandb_tags}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
