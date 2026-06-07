import torch

def bregman_divergence_quadratic(a, b):
    return (a - b).pow(2).mean(dim=-1) 

def bregman_divergence_xlogx(a, b, eps=1e-8):
    # Clamp to avoid log(0) and division by zero
    a_safe = torch.clamp(a, min=eps)
    b_safe = torch.clamp(b, min=eps)
    # Stable computation
    loss = a_safe * (torch.log(a_safe) - torch.log(b_safe)) - (a_safe - b_safe)
    return loss.mean(dim=-1)

def denoising_loss(forward, model, cfg, minibatch, divergence='quad'):
    
    # get batch size and device
    B, _ = minibatch.shape
    device = next(model.parameters()).device

    # split data and move to device
    if cfg.data.condition_dim > 0:
        conditioner = minibatch[:, 0:cfg.data.condition_dim].to(device)
        x0 = minibatch[:, cfg.data.condition_dim:].to(device).float()
    else:
        x0 = minibatch.to(device).float()

    # sample times uniformly from [min_time, max_time] and convert to sigma
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0

    # call Binomial to sample X_t | X_T = m
    xt = forward.conditional_sample(ts, x0)

    # evaluate model 
    if cfg.data.condition_dim > 0:
        model_x0 = model(ctx_ids=conditioner, xt_ids=xt, t=ts)
    else:
        model_x0 = model(xt, ts)

    # compute loss per sample
    if divergence == 'quad':
        loss = bregman_divergence_quadratic(model_x0, x0).sum(dim=-1)
    elif divergence == 'xlogx':
        loss = bregman_divergence_xlogx(model_x0, x0).sum(dim=-1)
    else:
        raise ValueError("Unsupported divergence type")

    if cfg.training.use_loss_weighting:
        loss *= torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)
 
    return loss.mean()
