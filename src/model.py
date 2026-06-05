import torch
import math
import torch.nn as nn
from torch.nn import functional as F

def timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # set the embedding frequencies to form a geometric progression from 1 to max_positions^(half_dim-1)/half_dim
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class _ResidualBackbone(nn.Module):
    def __init__(self, cfg):
        """ FiLM-conditioned residual MLP shared model types. """
        super().__init__() 
        # save model parameters
        self.S           = cfg.data.S
        self.D           = cfg.data.dim
        self.TIME_SCALE  = cfg.model.time_scale_factor
        self.D_MODEL     = cfg.model.d_model
        self.HIDDEN_DIM  = cfg.model.hidden_dim
        self.NUM_LAYERS  = cfg.model.num_layers
        self.TEMB_DIM    = cfg.model.temb_dim
        # define all middle layers
        self.input_proj  = nn.Linear(1, self.D_MODEL)
        self.layers1     = nn.ModuleList([nn.Linear(self.D_MODEL, self.HIDDEN_DIM)    for _ in range(self.NUM_LAYERS)])
        self.layers2     = nn.ModuleList([nn.Linear(self.HIDDEN_DIM, self.D_MODEL)    for _ in range(self.NUM_LAYERS)])
        self.norms       = nn.ModuleList([nn.LayerNorm(self.D_MODEL)                  for _ in range(self.NUM_LAYERS)])
        self.film        = nn.ModuleList([nn.Linear(4*self.TEMB_DIM, 2*self.D_MODEL)  for _ in range(self.NUM_LAYERS)])
        self.temb_net    = nn.Sequential(
            nn.Linear(self.TEMB_DIM, self.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, 4*self.TEMB_DIM)
        )
        self.act = nn.ReLU()

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x

    def unnormalize_input(self, x):
        return (x + 1.0) / 2.0 * self.S

    def _backbone(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 1) float in [0, S],  t: (B,) float in [0, 1]
        te = self.temb_net(timestep_embedding(t * self.TIME_SCALE, self.TEMB_DIM))
        h  = self.input_proj(x / self.S * 2.0 - 1.0)
        for i in range(self.NUM_LAYERS):
            h        = self.norms[i](h + self.layers2[i](self.act(self.layers1[i](h))))
            gam, bet = self.film[i](te).chunk(2, dim=-1)
            h        = gam * h + bet
        return h  # (B, d_model)


# ============================================================
# Denoising model: outputs scalar E[X_T | X_t]
# ============================================================
class DenoisingMLP(_ResidualBackbone):
    """
    Predicts E[X_T | X_t] as a scalar in approximately [0, S].
    Trained with MSE loss against the true x0.
    """
    def __init__(self, cfg): #S: int):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.D)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # returns: (B, 1)  predicted x0, in approximately (0, S)
        h = self._backbone(x, t)
        raw = self.out(h)  # (B, 1)
        # Linear rescaling: maps 0 -> S/2 (safe midpoint init), unbounded
        return self.unnormalize_input(raw)


# ============================================================
# Cross-entropy model: outputs logits over state space
# ============================================================
class CrossEntropyMLP(_ResidualBackbone):
    """
    Outputs logits over {0, ..., S-1} for CE training.
    E[X_T | X_t] is recovered as the expectation under softmax(logits).
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.S)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # returns: (B, S) logits
        h = self._backbone(x, t)
        return self.out(h)

    def expected_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, t)                                         # (B, S)
        probs  = F.softmax(logits, dim=-1)                                  # (B, S)
        k      = torch.arange(self.S, dtype=torch.float32, device=x.device) # (S,)
        return (probs * k).sum(dim=-1, keepdim=True)                        # (B, 1)


if __name__ == "__main__":

    # define model parameters
    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.NUM_LAYERS         = 3   
    cfg.D_MODEL            = 128  
    cfg.HIDDEN_DIM         = 256  
    cfg.TEMB_DIM           = 128  
    cfg.S                  = 32  
    cfg.TIME_SCALE         = 100
    cfg.D                  = 1
    batch_size = 8

    # define inputs
    x = torch.randn(batch_size, cfg.D)
    t = torch.randint(0, 1000, (batch_size,)).float()
    print(x.shape)
    print(t.shape)

    # evaluate models
    model = DenoisingMLP(cfg)
    print(model(x, t).shape)  # (batch_size, spatial_dim)

    model = CrossEntropyMLP(cfg)
    print(model(x, t).shape)  # (batch_size, S)
