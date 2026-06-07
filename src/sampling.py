import torch
from torch import Tensor
from typing import Optional

def Euler(
    forward,
    model,
    time_grid: Tensor,
    x0: Tensor,
    device,
    conditioner=None,
):
    """
    Euler scheme for sampling CMTC

    Inputs: 
      forward: Properties of forward process
      conditioner: Second model input
      model : Provides transition(), rate(), and p(x0|xt)
      time_grid : Tensor Monotonically increasingg time grid [0,...,T]

    Returns
    -------
      x0_hat : ndarray (N, D)
      x_hist : ndarray (K, N, D)
    """

    
    # define 
    S = forward.S

    # initialize x and t
    x_hist = [x0.clone()]
    x = x0.clone()
    B, D = x.shape[0], x.shape[1]

    for step in range(len(time_grid)-1):
        h = time_grid[step+1] - time_grid[step]

        # Evaluate model at (x, t) to get intensities at x
        t_tensor = torch.tensor([time_grid[step]], dtype=torch.float32, device=device)
        t_tensor = t_tensor.repeat(x.shape[0])                     # (B,)
        with torch.no_grad():
            if conditioner is None:
                intensities = model(x.float(), t_tensor)
            else:
                intensities = model(x.float(), t_tensor, conditioner)

        # Mask out jumps that would exceed S-1
        state_mask = (x.long() + 1) < S                            # (B, D)
        intensities = intensities * state_mask.to(intensities)

        # Next-state indices (clamped, but masked intensities are 0 anyway)
        x_curr = x.long()                                          # (B, D)
        x_plus = (x_curr + 1).clamp(max=S - 1)                     # (B, D)

        # Define rates matrix
        rates = torch.zeros((B, D, S), device=device)
        batch_idx = torch.arange(B, device=device)[:, None]        # (B, 1)
        dim_idx   = torch.arange(D, device=device)[None, :]        # (1, D)
        rates[batch_idx, dim_idx, x_plus.long()] = intensities
        rates[batch_idx, dim_idx, x.long()]      = -intensities

        # Construct transition probabilities using Euler method
        for i in range(D):
           # One-hot over current state: (B, S)
           p_cond_i = torch.eye(S, device=device)[x[:, i].long()]    # (B, S)
           p_cond_i = p_cond_i + h * rates[:, i, :]
           p_cond_i = p_cond_i.clamp(min=0.0)                        # ensure non-negative
           p_cond_i = p_cond_i / p_cond_i.sum(dim=-1, keepdim=True)  # normalize over the state space
           # Sample new state and increment time
           x[:,i] = torch.multinomial(p_cond_i, num_samples=1).squeeze(-1)

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )

def _tau_leap_step(
    x: Tensor,
    rates: Tensor,
    tau: float,
    max_state: Optional[int] = None,
    reject_multi_jumps: bool = False,
    max_expected_jumps: float = 0.5,
) -> Tensor:
    """Single tau-leaping state update."""
    rates_pos = rates.clamp(min=0.0)

    # Stabilize numerics so late-time intensities do not explode.
    if max_expected_jumps is not None:
        rates_pos = rates_pos.clamp(max=max_expected_jumps / max(tau, 1e-8))

    # Categorical setting: reject multi-jumps by sampling at most one jump.
    if reject_multi_jumps:
        jump_prob = 1.0 - torch.exp(-rates_pos * tau)
        jumps = torch.bernoulli(jump_prob)
    else:
        jumps = torch.poisson(rates_pos * tau)

    # Update state
    x_next = x + jumps
    if max_state is not None:
        x_next = x_next.clamp(max=max_state)
    return x_next

def TauLeaping(
    forward,
    model,
    time_grid: Tensor,
    x0: Tensor,
    device,
    conditioner=None,
    corrector_steps: int = 0,
    late_stage_start: float = 0.9,
    corrector_tau_scale: float = 0.1,
    reject_multi_jumps: bool = False,
    corrector_noise_scale: float = 0.0,
    max_expected_jumps: float = 0.5,
):
    """
    Tau-leaping sampler with optional late-stage predictor-corrector updates.

    Inputs: 
      forward: Properties of forward process
      conditioner: Second model input
      model : Provides transition(), rate(), and p(x0|xt)
      time_grid : Tensor Monotonically increasingg time grid [0,...,T]
      corrector_steps : Number of corrector steps per predictor step in late stage
      late_stage_start : Fraction of sampling trajectory after which corrector starts
      corrector_tau_scale : Tau scaling for corrector steps (usually smaller than predictor tau)
      reject_multi_jumps : If True, reject >1 jumps per dimension per step (categorical stability)
      corrector_noise_scale : Uniform exploration rate added during corrector step
      max_expected_jumps : Upper bound for rate*tau per dimension for numerical stability

    Returns
    -------
      x0_hat : ndarray (N, D)
      x_hist : ndarray (K, N, D)
    """

    # initialize x and t
    x_hist = [x0.clone()]
    x = x0.clone()
    
    total_steps = max(len(time_grid) - 1, 1)
    max_state = getattr(forward, "S", None)
    if max_state is not None:
        max_state = int(max_state) - 1

    for step, (t_prev, t) in enumerate(zip(time_grid[:-1], time_grid[1:])):
        tau = float(t - t_prev)

        # Compute rates Q(x -> y)
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
        t_tensor = t_tensor.repeat(x.shape[0])  # (B,)
        with torch.no_grad():
            if conditioner is None:
                rates = model(x.float(), t_tensor)
            else:
                rates = model(x.float(), t_tensor, conditioner)

        # Predictor step
        x = _tau_leap_step(
            x=x,
            rates=rates,
            tau=tau,
            max_state=max_state,
            reject_multi_jumps=reject_multi_jumps,
            max_expected_jumps=max_expected_jumps,
        )

        # Late-stage corrector steps (paper-style schedule near end of trajectory)
        frac_done = (step + 1) / total_steps
        if corrector_steps > 0 and frac_done >= late_stage_start:
            corrector_tau = tau * corrector_tau_scale
            for _ in range(corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        corr_rates = model(x.float(), t_tensor)
                    else:
                        corr_rates = model(x.float(), t_tensor, conditioner)
                if corrector_noise_scale > 0.0:
                    # Proxy for forward-rate mixing in discrete PC samplers.
                    corr_rates = corr_rates + corrector_noise_scale * torch.ones_like(corr_rates)
                x = _tau_leap_step(
                    x=x,
                    rates=corr_rates,
                    tau=corrector_tau,
                    max_state=max_state,
                    reject_multi_jumps=reject_multi_jumps,
                    max_expected_jumps=max_expected_jumps,
                )

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )
