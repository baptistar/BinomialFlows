import torch
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from scipy.stats import wasserstein_distance
import yaml
import copy

import sys
sys.path.append('../src') 

from model import DenoisingMLP
from loss import denoising_loss
from forward import PoissonFolmer
import datasets
from sampling import Euler, TauLeaping
from model_utils import Intensity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20,          # base font size
})


# load config file
with open("./config.yaml", "r") as f:
     cfg_dict = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(cfg_dict)

# # define data and sampling parameters 
num_MC_samples = 100 #10000
cfg.sampling.num_samples = 100

run_dir = './runs/run_20260605_124805'

# class Intensity(nn.Module):
#     def __init__(self, model, max_time):
#         super(Intensity, self).__init__()
#         self.model = model
#         self.max_time = max_time

#     def forward(self, x, t):
#         assert(t.ndim == 1)
#         return (self.model(x, t) - x.float()) / (self.max_time - t[:,None])

def bregman_divergence_xlogx(x, y, eps=1e-12):
    # Avoid division by zero in log
    y_safe = torch.clamp(y, min=eps)
    # Compute the log term only where x > 0
    log_term = torch.zeros_like(x)
    mask = x > 0
    log_term[mask] = x[mask] * torch.log(x[mask] / y_safe[mask])
    # Full divergence formula
    divergence = log_term - x + y
    return divergence

def log_likelihood(x, forward, model, num_samples=1000):
    # sample times uniformly from 0 to T
    t = torch.rand(num_samples) * forward.T
    negloglik_mean = torch.zeros(x.shape[0], device=x.device)
    negloglik_var  = torch.zeros(x.shape[0], device=x.device)
    # sample from the bridge
    for i, xi in enumerate(x):
        x_input = xi.unsqueeze(0).repeat(num_samples,1)
        y = forward.conditional_sample(t, x_input)
        # evaluate model and divergence
        integrand = bregman_divergence_xlogx((xi - y)/(forward.T - t).unsqueeze(-1), model(y, t))
        negloglik_mean[i] = torch.mean(integrand)
        negloglik_var[i]  = torch.var(integrand)
    return negloglik_mean, negloglik_var

# define set of distributions
dists = [datasets.Poisson(5.0, 40, device), \
        datasets.PoissonMixture(device), \
        datasets.ZeroInflatedPoisson(device), \
        datasets.NegativeBinomialMixture(device), \
	    datasets.BetaNegativeBinomial(device), 
        datasets.ZipfDistribution(device), \
        datasets.YuleSimonDistribution(device)]
num_trials = 5

# define arrays to store results
nll = np.zeros((len(dists), num_trials))
nll_base = np.zeros((len(dists), num_trials))
W1_all  = np.zeros((len(dists), num_trials))

ctr = 0
for (kk,dist) in enumerate(dists):
    ctr += 1
    
    # load distribution
    data = dist.sample(cfg.sampling.num_samples)
    cfg.data.S = dist.S

    print('============================================')
    print(f"Distribution: {dist.__class__.__name__}")
    for trial in range(num_trials):
        print('============================================')
        print(f"Testing model for distribution: {dist.__class__.__name__}, Trial: {trial+1}")

        # define forward process and model
        forward = PoissonFolmer(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim)

        # define approximate model
        NNmodel = DenoisingMLP(cfg)
        NNmodel = NNmodel.to(device)

        # EMA model (a copy of the NNmodel)
        EMAmodel = copy.deepcopy(NNmodel)
        for p in EMAmodel.parameters():
            p.requires_grad_(False)  # EMA model is not trained by gradient

        # load model and define intensity
        EMAmodel.load_state_dict(torch.load(f'{run_dir}/ema_model_{dist.__class__.__name__}_trial{trial+1}.pth'))
        NNintensity = Intensity(EMAmodel, max_time=cfg.data.T, S=cfg.data.S, denormalize=False)

        #######################################
        # SAMPLE MODEL   
        #######################################

        # define initial condition
        x0 = forward.sample_initial(cfg.sampling.num_samples, device=device)

        # define time grid and samples
        time_grid = torch.linspace(forward.T0, forward.T-1e-4, cfg.sampling.num_steps)
        if cfg.sampling.ode_solver == 'euler':
            samples, x_hist = Euler(forward, NNintensity, time_grid, x0, device=device)
        else:
            samples, x_hist = TauLeaping(forward, NNintensity, time_grid, x0, device=device)
        print('Generated samples')

        # plot samples
        bins = np.arange(0, dist.S + 0.5) - 0.5
        plt.figure()
        plt.hist(data, bins, density=True, weights=np.ones(data.shape[0])/data.shape[0], alpha=0.5, label='True Samples')
        plt.hist(samples.detach().numpy(), bins, density=True, weights=np.ones(samples.shape[0])/samples.shape[0], alpha=0.6, label='Generated Samples')
        plt.plot(bins[:-1] + 0.5, dist.probs(), '-k', lw=1.5, label='True PMF')
        plt.ylim([0, 0.03])
        #if ctr <= 2:
        #    plt.legend()
        plt.tight_layout()
        plt.savefig(f'{run_dir}/samples_{dist.__class__.__name__}_trial{trial+1}.pdf', bbox_inches='tight')
        plt.close('all')

        #######################################
        # EVALUATE LOG-LIKELIHOOD    
        #######################################

        # evaluate log-likelihood on a Grid
        x_all = torch.arange(0, dist.S).long().to(device)
        negloglik_grid_mean, negloglik_grid_var = log_likelihood(x_all, forward, NNintensity, num_samples=num_MC_samples)

        # evaluate standard error
        negloglik_grid_sderror = 1.96 * torch.sqrt(negloglik_grid_var / num_MC_samples)

        # plot samples
        bins = np.arange(0, dist.S + 0.5) - 0.5
        plt.figure()
        plt.plot(x_all.cpu().numpy(), -negloglik_grid_mean.cpu().numpy(), '-r', label='Approx Model Log-PMF')
        plt.plot(x_all.cpu().numpy(), -negloglik_grid_mean.cpu().numpy() + negloglik_grid_sderror.cpu().numpy(), '--r')
        plt.plot(x_all.cpu().numpy(), -negloglik_grid_mean.cpu().numpy() - negloglik_grid_sderror.cpu().numpy(), '--r')
        plt.plot(x_all.cpu().numpy(), dist.log_probs().cpu().numpy(), '-k', lw=1.5, label='True Log-PMF')
        plt.ylim([-5, torch.max(dist.log_probs()).item()+0.5])
        if ctr <= 2:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{run_dir}/likelihood_{dist.__class__.__name__}_trial{trial+1}.pdf', bbox_inches='tight')
        plt.close('all')

        #######################################
        # COMPUTE METRICS  
        #######################################

        samples_np = samples.float().cpu().numpy().squeeze()
        print(f'Range of data [{np.min(samples_np)}, {np.max(samples_np)}]')

        negloglik_mean, negloglik_var = log_likelihood(data.to(device), forward, NNintensity, num_samples=5000)
        print(f"Log-Likelihood on Data with Approx Model: {-1*negloglik_mean.mean().item():.4f} +/- {1.96*torch.sqrt(negloglik_var.mean()/len(data)).item():.4f}")
        
        z = (negloglik_mean - negloglik_mean.mean()) / negloglik_mean.std()
        negloglik_filt = negloglik_mean[torch.abs(z) <= 3.0]
        negloglik_var_filt = negloglik_var[torch.abs(z) <= 3.0]
        print(f"Log-Likelihood on Data with Approx Model (Filtered): {-1*negloglik_filt.mean().item():.4f} +/- {1.96*torch.sqrt(negloglik_var_filt.mean()/len(data)).item():.4f}")

        loglik_true = dist.log_probs()[data.to(device)]
        print(f"Log-Likelihood on Data with True Model: {loglik_true.mean().item():.4f}")

        W1 = wasserstein_distance(samples_np, data.float().cpu().numpy().squeeze()).item()
        print(f"Wasserstein distance: {W1:.4f}")

        # save results
        torch.save({'data':data, 'generated':samples, 'NLLmean': negloglik_mean, 'NLLvar': negloglik_var}, \
                    f'{run_dir}/samples_{dist.__class__.__name__}_trial{trial+1}.pt')

        # store results
        nll[kk, trial] = negloglik_filt.mean().item()
        nll_base[kk, trial] = -1*loglik_true.mean().item()
        W1_all[kk, trial] = W1


# print results
print(f"Summary of Results")
for (k, dist) in enumerate(dists):
    print(f"Distribution: {dist.__class__.__name__}")
    print(f"Approx Model NLL: {nll[k,:].mean():.4f} +/- {nll[k,:].std()/np.sqrt(num_trials):.4f}")
    print(f"Base Model NLL:   {nll_base[k,:].mean():.4f} +/- {nll_base[k,:].std()/np.sqrt(num_trials):.4f}")
    print(f"Wasserstein Distance: {W1_all[k,:].mean():.4f} +/- {W1_all[k,:].std()/np.sqrt(num_trials):.4f}")