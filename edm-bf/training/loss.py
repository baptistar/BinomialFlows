# ruff: noqa
# fmt: off
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import math
import torch.distributions as dist
import torch.nn.functional as F
from torch_utils import persistence, training_stats

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Poisson–Föllmer loss.

@persistence.persistent_class
class PoissonFollmerLoss:
    def __init__(self, T=1.0, log_noise_eps=1e-5, log_noise_mu_mul=0.5, log_noise_sigma_mul=0.2,
                 mean_data=120.7, var_data=4115.2, cin_eps=1e-1, no_edm=False):
        self.T = float(T)
        self.log_noise_eps = float(log_noise_eps)
        self.log_noise_mu_mul = float(log_noise_mu_mul)
        self.log_noise_sigma_mul = float(log_noise_sigma_mul)
        self.mean_data = float(mean_data)
        self.var_data = float(var_data)
        self.cin_eps = float(cin_eps)
        self.no_edm = bool(no_edm)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        N = images.shape[0]
        # σ := -log(t + ε)
        eps = self.log_noise_eps
        t_max = max(self.T - eps, 0.0)  # ensures t+ε <= T = 1
        lo = math.log(eps)
        hi = math.log(t_max + eps)
        mu = lo + self.log_noise_mu_mul * (hi - lo)
        std = self.log_noise_sigma_mul * (hi - lo)

        z = torch.randn([N, 1, 1, 1], device=device, dtype=torch.float32) * std + mu
        z = torch.clamp(z, lo, hi)
        t = torch.exp(z) - eps
        t = torch.clamp(t, 0.0, t_max)
        sigma = -z

        # Augmentation pipeline.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # Convert augmented float image in [-1,1] back to integer counts in [0,255].
        x_T = torch.clamp((y + 1) * 127.5, 0, 255).round().to(torch.float32)
        alpha = t / self.T
        x_t = dist.Binomial(total_count=x_T, probs=alpha).sample()
        D_xt = net(x_t, sigma, labels, augment_labels=augment_labels)
        if self.no_edm:
            scale = math.sqrt(self.var_data + self.mean_data ** 2)
            w2 = 1.0 / (scale ** 2)
            loss = w2 * (D_xt - x_T) ** 2
        else:
            # Loss weighting.
            mean = self.mean_data
            var = self.var_data
            denom_out = mean * (1.0 - t) + var * t
            c_skip = var / torch.clamp(denom_out, min=1e-20)
            c_out = torch.sqrt(var * mean * (1.0 - t) / denom_out)
            c_loss_2 = 1.0 / (c_out ** 2 + (mean - c_skip * mean * t) ** 2 + self.cin_eps)
            loss = c_loss_2 * (D_xt - x_T) ** 2
        training_stats.report('Loss/loss_base', loss)
        return loss

#----------------------------------------------------------------------------
# Poisson–Föllmer loss (xlogx / Poisson KL on bridge intensities).

@persistence.persistent_class
class PoissonFollmerXLogXLoss:
    def __init__(self, T=1.0, log_noise_eps=1e-5, log_noise_mu_mul=0.5, log_noise_sigma_mul=0.2,
                 mean_data=120.7, var_data=4115.2, lambda_clip=4096.0, lambda_eps=1e-12):
        self.T = float(T)
        self.log_noise_eps = float(log_noise_eps)
        self.log_noise_mu_mul = float(log_noise_mu_mul)
        self.log_noise_sigma_mul = float(log_noise_sigma_mul)
        self.mean_data = float(mean_data)
        self.var_data = float(var_data)
        self.lambda_clip = float(lambda_clip)
        self.lambda_eps = float(lambda_eps)
        if not (self.lambda_clip > 0):
            raise ValueError(f'lambda_clip must be > 0, got {self.lambda_clip}')
        if not (self.lambda_eps > 0):
            raise ValueError(f'lambda_eps must be > 0, got {self.lambda_eps}')
        if not (self.lambda_clip > self.lambda_eps):
            raise ValueError(f'lambda_clip must be > lambda_eps, got {self.lambda_clip} and {self.lambda_eps}')

        # Normalization for reasonable loss/grad magnitudes in --no-edm mode.
        self._scale = math.sqrt(self.var_data + self.mean_data ** 2)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        N = images.shape[0]

        # σ := -log(t + ε)
        eps = self.log_noise_eps
        t_max = max(self.T - eps, 0.0)  # ensures t+ε <= T = 1
        lo = math.log(eps)
        hi = math.log(t_max + eps)
        mu = lo + self.log_noise_mu_mul * (hi - lo)
        std = self.log_noise_sigma_mul * (hi - lo)

        z = torch.randn([N, 1, 1, 1], device=device, dtype=torch.float32) * std + mu
        z = torch.clamp(z, lo, hi)
        t = torch.exp(z) - eps
        t = torch.clamp(t, 0.0, t_max)
        sigma = -z

        # Augmentation pipeline.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # Convert augmented float image in [-1,1] back to integer counts in [0,255].
        x_T = torch.clamp((y + 1) * 127.5, 0, 255).round().to(torch.float32)
        alpha = t / self.T
        x_t = dist.Binomial(total_count=x_T, probs=alpha).sample()

        # Network predicts log-intensity.
        log_lambda_hat = net(x_t, sigma, labels, augment_labels=augment_labels, return_log_rate=True)
        log_lambda_hat = log_lambda_hat.to(torch.float32)
        log_lambda_hat = torch.clamp(log_lambda_hat, min=math.log(self.lambda_eps), max=math.log(self.lambda_clip))
        lambda_hat = torch.exp(log_lambda_hat)

        # True bridge intensity: λ = (x_T - x_t) / (T - t).
        gap = torch.clamp(self.T - t, min=1e-20)
        lambda_true = (x_T - x_t) / gap
        lambda_true = torch.clamp(lambda_true, 0.0, self.lambda_clip)

        # D_{x log x}(p ∥ q) = p log(p/q) - p + q, element-wise.
        q_safe = torch.clamp(lambda_hat, min=self.lambda_eps)
        log_term = torch.zeros_like(lambda_true)
        mask = lambda_true > 0
        log_term[mask] = lambda_true[mask] * (torch.log(lambda_true[mask]) - torch.log(q_safe[mask]))
        div = log_term - lambda_true + lambda_hat
        loss = div / self._scale

        training_stats.report('Loss/loss_base', loss)
        return loss

#----------------------------------------------------------------------------
# ItDPDM loss (Poisson noise channel, MSE).

@persistence.persistent_class
class ITPLoss:
    def __init__(self, mean_data=120.7, var_data=4115.2, cin_eps=1e-2,
                 logistic_loc=-6.0, logistic_scale=2.0, no_edm=False):
        self.mean_data = float(mean_data)
        self.var_data = float(var_data)
        self.cin_eps = float(cin_eps)
        self.logistic_loc = float(logistic_loc)
        self.logistic_scale = float(logistic_scale)
        self.no_edm = bool(no_edm)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        N = images.shape[0]
        mu = self.mean_data
        sigma2 = self.var_data

        # Sample SNR: α ~ Logistic(loc, scale), γ = exp(α).
        u = torch.rand([N, 1, 1, 1], device=device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        alpha = self.logistic_loc + self.logistic_scale * torch.log(u / (1.0 - u))
        gamma = torch.exp(alpha)

        # Augmentation pipeline.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # Convert augmented float image in [-1,1] back to integer counts in [0,255].
        x_T = torch.clamp((y + 1) * 127.5, 0, 255).round().to(torch.float32)

        # Corrupt: z_γ ~ Poisson(γ · x_T).
        z_gamma = torch.poisson(gamma * x_T)

        # Network prediction.
        D_z = net(z_gamma, gamma, labels, augment_labels=augment_labels)

        if self.no_edm:
            scale = math.sqrt(self.var_data + self.mean_data ** 2)
            w2 = 1.0 / (scale ** 2)
            loss = w2 * (D_z - x_T) ** 2
        else:
            # Implemented loss weight: derived denominator plus a small +cin_eps floor.
            denom_out = mu + sigma2 * gamma
            c_out_sq = mu * sigma2 / denom_out
            c_skip = sigma2 / denom_out
            bias_sq = (mu - c_skip * gamma * mu) ** 2
            w2 = 1.0 / (c_out_sq + bias_sq + self.cin_eps)
            loss = w2 * (D_z - x_T) ** 2
        training_stats.report('Loss/loss_base', loss)
        return loss

#----------------------------------------------------------------------------
# Poisson–Föllmer categorical loss (cross entropy over {0,...,255} per channel).

@persistence.persistent_class
class PoissonFollmerCatLoss:
    def __init__(self, T=1.0, log_noise_eps=1e-5, log_noise_mu_mul=0.5, log_noise_sigma_mul=0.2, num_classes=256):
        self.T = float(T)
        self.log_noise_eps = float(log_noise_eps)
        self.log_noise_mu_mul = float(log_noise_mu_mul)
        self.log_noise_sigma_mul = float(log_noise_sigma_mul)
        self.num_classes = int(num_classes)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        N = images.shape[0]

        # σ := -log(t + ε)
        eps = self.log_noise_eps
        t_max = max(self.T - eps, 0.0)  # ensures t+ε <= T = 1
        lo = math.log(eps)
        hi = math.log(t_max + eps)
        mu = lo + self.log_noise_mu_mul * (hi - lo)
        std = self.log_noise_sigma_mul * (hi - lo)

        z = torch.randn([N, 1, 1, 1], device=device, dtype=torch.float32) * std + mu
        z = torch.clamp(z, lo, hi)
        t = torch.exp(z) - eps
        t = torch.clamp(t, 0.0, t_max)
        sigma = -z

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        x_1 = torch.clamp((y + 1) * 127.5, 0, 255).round().to(torch.int64)

        alpha = t / self.T
        x_t = dist.Binomial(total_count=x_1.to(torch.float32), probs=alpha).sample().to(torch.int64)

        logits = net(x_t, sigma, labels, augment_labels=augment_labels, return_logits=True)

        N, C, H, W = x_1.shape
        K = int(logits.shape[2])
        logits_f32 = logits.to(torch.float32)
        loss_ce = F.cross_entropy(
            logits_f32.reshape(N * C, K, H, W),
            x_1.clamp(0, K - 1).reshape(N * C, H, W),
            reduction='none',
        ).reshape(N, C, H, W)

        training_stats.report('Loss/loss_base', loss_ce)
        return loss_ce

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, sigma_max=80.0, method='bm',
                 dd_jvp_weight=0.0, dd_time_weight=0.0,
                 dd_enable_jvp=False, dd_enable_time=False,
                 dd_log_grad=False):
        """
        sigma_max plays the role of γ (total standard deviation at t=1).
        method: 'bm' here implements the independent-endpoint bridge in normalized time t ∈ [0,1].
                'dd' preserved for reference/testing (direct EDM noising).
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max  # γ
        self.method = method
        self.dd_jvp_weight = float(dd_jvp_weight)
        self.dd_time_weight = float(dd_time_weight)
        self.dd_enable_jvp = bool(dd_enable_jvp)
        self.dd_enable_time = bool(dd_enable_time)
        self.dd_log_grad = bool(dd_log_grad)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        sigma_data = self.sigma_data
        gamma = self.sigma_max  # total std at t=1

        # Sample sigma from the EDM log-normal schedule
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=device)
        sigma_raw = (rnd_normal * self.P_std + self.P_mean).exp()

        # Prepare data (y) and optional augmentation
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        if self.method == 'dd':
            # Direct EDM: x = y + sigma_raw * eps
            sigma_used = sigma_raw
            eps = torch.randn_like(y)
            x_sigma = y + sigma_used * eps

        else:  # 'bm' and 'bmf'
            # Normalized-time bridge for dX_t = γ sqrt(2 t) dW_t, t ∈ [0,1]
            # Map log-normal sigma schedule to t ∈ [0,1], then σ_used = γ t.
            t = torch.clamp(sigma_raw / gamma, max=1.0)    # shape [N,1,1,1]
            a = t ** 2                                     # a = ψ(t)/ψ(1) = ψ(t) = t^2
            sigma_used = gamma * t                         # σ used for conditioning/weight: clamped sigma_raw

            # Sample endpoint z ~ N(0, (σ_data^2 + γ^2) I) independently of y
            z_scale = math.sqrt(gamma**2 + sigma_data**2)
            z = torch.randn_like(y) * z_scale

            # Bridge noise with variance Γ(t)(1-a) = γ^2 t^2 (1 - t^2)
            eps = torch.randn_like(y)
            x_sigma = (1.0 - a) * y + a * z + gamma * torch.sqrt(torch.clamp(a * (1.0 - a), min=0.0)) * eps

        # EDM loss weight computed with the same σ that is fed to the network.
        weight = (sigma_used ** 2 + sigma_data ** 2) / ((sigma_used * sigma_data) ** 2)

        # Network prediction and base EDM denoising loss.
        D_yn = net(x_sigma, sigma_used, labels, augment_labels=augment_labels)
        loss_base = weight * ((D_yn - y) ** 2)
        training_stats.report('Loss/loss_base', loss_base)
        loss = loss_base

        # Optional derivative-based losses (EDM σ-time parametrization, method='dd' only).
        if self.method != 'dd' or (not self.dd_enable_jvp and not self.dd_enable_time):
            return loss

        params = None
        if self.dd_log_grad:
            params = [p for p in net.parameters() if p.requires_grad]

        # Dimension of a single sample (d in dd.tex).
        d = float(y[0].numel())
        # Residual R = X0 - D_theta(X, σ).
        R = y - D_yn

        # JVP loss: match J(D_theta(X, σ))v:
        if self.dd_enable_jvp:
            # JVP direction v drawn from the unit sphere (per sample).
            v = torch.randn_like(y)
            v_norm = v.reshape(v.shape[0], -1).norm(dim=1, keepdim=True)
            v = v / v_norm.view(-1, 1, 1, 1)

            # Compute J(D_θ) v via JVP w.r.t. input x.
            x_sigma_jvp = x_sigma.detach().requires_grad_(True)

            def _func_x(x_in):
                return net(x_in, sigma_used, labels, augment_labels=augment_labels)

            _, jvp_x = torch.autograd.functional.jvp(
                _func_x,
                (x_sigma_jvp,),
                (v,),
                create_graph=True,
            )

            # JVP regression target κ.
            R_dot_v = (R * v).sum(dim=[1, 2, 3], keepdim=True)
            kappa_jvp = ((R * R_dot_v) / (sigma_used ** 2)).detach()

            # λ_J(σ) = 1 / c_skip(σ)^2, where c_skip(σ) = σ_data^2 / (σ^2 + σ_data^2).
            c_skip = sigma_data ** 2 / (sigma_used ** 2 + sigma_data ** 2)
            lambda_jvp = 1.0 / (c_skip ** 2)

            loss_jvp = lambda_jvp * (jvp_x - kappa_jvp) ** 2
            # Log raw JVP loss (before multiplying by dd_jvp_weight).
            training_stats.report('Loss/loss_jvp_raw', loss_jvp)
            loss = loss + self.dd_jvp_weight * loss_jvp

            # Optionally log gradient norm contributed by the JVP loss only:
            if self.dd_log_grad and self.dd_jvp_weight != 0 and params:
                L_jvp = (self.dd_jvp_weight * loss_jvp).mean()
                grads = torch.autograd.grad(L_jvp, params, retain_graph=True, allow_unused=True)
                sq = torch.zeros([], device=device)
                for g in grads:
                    if g is not None:
                        g32 = g.detach().to(torch.float32)
                        sq = sq + g32.square().sum()
                grad_norm_jvp = sq.sqrt()
                training_stats.report('Loss/grad_norm_jvp', grad_norm_jvp)

        # Noise/time-derivative loss in σ-time:
        if self.dd_enable_time:
            sigma_jvp = sigma_used.detach().requires_grad_(True)

            def _func_sigma(sigma_in):
                return net(x_sigma, sigma_in, labels, augment_labels=augment_labels)

            # ∂_σ D_θ(X, σ) via JVP in σ-direction (unit direction in σ-time).
            _, dD_dsigma = torch.autograd.functional.jvp(
                _func_sigma,
                (sigma_jvp,),
                (torch.ones_like(sigma_jvp),),
                create_graph=True,
            )

            # β and κ_σ targets.
            delta = x_sigma - y
            delta_sq = (delta ** 2).sum(dim=[1, 2, 3], keepdim=True)
            beta = (delta_sq - d * (sigma_used ** 2)) / (sigma_used ** 3)
            kappa_sigma = (R * beta).detach()

            # λ_σ(σ) = ((σ^2 + σ_data^2)^3) /
            #          (2 σ_data^2 (d σ^4 + 2(d+1) σ^2 σ_data^2 + (d+4) σ_data^4)).
            s2 = sigma_used ** 2
            sdata2 = sigma_data ** 2
            numer = (s2 + sdata2) ** 3
            denom = 2 * sdata2 * (d * s2 ** 2 + 2 * (d + 1) * s2 * sdata2 + (d + 4) * sdata2 ** 2)
            lambda_sigma = numer / denom

            loss_sigma = lambda_sigma * (dD_dsigma - kappa_sigma) ** 2
            # Log raw σ-derivative loss (before multiplying by dd_time_weight).
            training_stats.report('Loss/loss_sigma_raw', loss_sigma)
            loss = loss + self.dd_time_weight * loss_sigma

            # Optionally log gradient norm contributed by the σ-derivative loss only:
            if self.dd_log_grad and self.dd_time_weight != 0 and params:
                L_sigma = (self.dd_time_weight * loss_sigma).mean()
                grads = torch.autograd.grad(L_sigma, params, retain_graph=True, allow_unused=True)
                sq = torch.zeros([], device=device)
                for g in grads:
                    if g is not None:
                        g32 = g.detach().to(torch.float32)
                        sq = sq + g32.square().sum()
                grad_norm_sigma = sq.sqrt()
                training_stats.report('Loss/grad_norm_sigma', grad_norm_sigma)

        return loss

#----------------------------------------------------------------------------
