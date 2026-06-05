import torch

# ============================================================
# Forward process: Poisson-Folmer bridge
# ============================================================

class PoissonFolmer:
    def __init__(self, S: int, T0: float, T: float, dim: int):
        self.S   = S
        self.T0  = T0
        self.T   = T
        self.dim = dim

    def sample_initial(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(N, self.dim, dtype=torch.long, device=device)

    def conditional_sample(self, t: torch.Tensor, XT: torch.Tensor) -> torch.Tensor:
        # Sample from P[X_t = k|X_T = m]
        assert(XT.shape[1] == self.dim)
        assert(XT.shape[0] == len(t))
        return torch.distributions.Binomial(
            total_count=XT.T, probs=t / self.T
        ).sample().long().T

