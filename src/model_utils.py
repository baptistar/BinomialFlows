import torch
import torch.nn as nn

# define intensity model
class Intensity(nn.Module):
    def __init__(self, model, max_time, S, denormalize):
        super(Intensity, self).__init__()
        self.model = model
        self.max_time = max_time
        self.S = S
        self.normalize_targets = denormalize

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)
        if ctx is not None:
            model_output = self.model(ctx, x, t)
        else:
            model_output = self.model(x, t)
        if self.normalize_targets:
            model_output *= self.S
        return (model_output - x.float()) / (self.max_time - t[:,None])

