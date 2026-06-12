import torch
import matplotlib.pyplot as plt
import ml_collections
import copy 
import yaml
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model import DenoisingMLP
from loss import denoising_loss
from forward import PoissonFolmer
from utils import create_run_dir
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create directory for saving models and figures
pwd = '/scratch/baptista'
if not os.path.exists(pwd+'/runs'):
    os.makedirs(pwd+'/runs')
run_dir = create_run_dir(pwd+'/runs')

# load config file
with open(pwd+"/config.yaml", "r") as f:
     cfg_dict = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(cfg_dict)

# define set of distributions
dists = [datasets.Poisson(5.0, 40, device), \
        datasets.PoissonMixture(device), \
        datasets.ZeroInflatedPoisson(device), \
        datasets.NegativeBinomialMixture(device), \
        datasets.BetaNegativeBinomial(device), \
        datasets.ZipfDistribution(device), \
        datasets.YuleSimonDistribution(device)]
num_trials = 5

for dist in dists:
    for trial in range(num_trials):
        print(f"Training model for distribution: {dist.__class__.__name__}, Trial: {trial+1}")
    
        # load module and distribution
        data = dist.sample(cfg.data.N)
        cfg.data.S = dist.S

        # define forward process and model
        forward = PoissonFolmer(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim)

        # define approximate model
        NNmodel = DenoisingMLP(cfg)
        NNmodel = NNmodel.to(device)

        # EMA model (a copy of the NNmodel)
        EMAmodel = copy.deepcopy(NNmodel)
        for p in EMAmodel.parameters():
            p.requires_grad_(False)  # EMA model is not trained by gradient

        # define optimizer
        optimizer = torch.optim.Adam(NNmodel.parameters(), cfg.training.lr, 
                                    weight_decay=cfg.training.weight_decay)

        # define dataloader
        dataloader = torch.utils.data.DataLoader(data,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.data.shuffle)

        # EMA update function
        def update_ema(model, ema_model, decay):
            with torch.no_grad():
                msd = model.state_dict()
                for k, ema_v in ema_model.state_dict().items():
                    model_v = msd[k].detach()
                    ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

        training_loss  = []
        for _ in range(cfg.training.n_epochs):
            for minibatch in dataloader:
                
                optimizer.zero_grad()
                l = denoising_loss(forward, NNmodel, cfg, minibatch)
                l.backward()
                
                if cfg.training.clip_grad:
                    torch.nn.utils.clip_grad_norm_(NNmodel.parameters(), 1.0)
                optimizer.step()

                update_ema(NNmodel, EMAmodel, cfg.model.ema_decay)

                training_loss.append(l.item())

        # save model
        torch.save(EMAmodel.state_dict(), run_dir + f'/ema_model_{dist.__class__.__name__}_trial{trial+1}.pth')

        plt.figure()
        plt.plot(training_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Denoising Loss')
        plt.tight_layout()
        plt.savefig(run_dir + f'/training_loss_{dist.__class__.__name__}_trial{trial+1}.pdf', bbox_inches='tight')
        plt.close()
