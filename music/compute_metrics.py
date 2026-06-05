import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
import bookkeeping
import copy
import yaml

from pathlib import Path
import numpy as np
import argparse

sys.path.append('model')
from forward import PoissonFolmerBase
from denoiser import ConditionalSequenceTransformer, PFConditionalPrecond
from model_utils import load_model, Intensity
from sampling import TauLeaping, Euler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluation script")
    parser.add_argument(
        "--model_run",
        type=str,
        required=True,
        help="Path to the model run directory (e.g. ./runs/run_xxx)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_run = args.model_run

    # load config
    with open(model_run + "/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ml_collections.ConfigDict(cfg_dict)

    ###############################
    # LOAD DATA
    ###############################

    # load data
    data_folder = './data/'
    train_data  = np.load(data_folder + 'train.npy')
    test_data   = np.load(data_folder + 'test.npy')

    # descramble data back to original ordering
    descramble_key = np.loadtxt(data_folder + 'descramble_key.txt')
    def descramble(samples):
        return descramble_key[samples.flatten()].reshape(*samples.shape)
    descrambled_train_data  = descramble(train_data)
    descrambled_test_data   = descramble(test_data)

    mean_data = (descrambled_train_data).mean()
    var_data  = (descrambled_train_data).var()

    ###############################
    # LOAD MODEL
    ###############################

    # define model
    model, _ = load_model(cfg, mean_data, var_data, device)

    # define forward process
    forward = PoissonFolmerBase(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim-cfg.data.condition_dim)
    
    # load NN model
    model_path = Path(model_run + '/checkpoint.pt')
    if model_path.exists():
        print('Loading model from', model_path)
        checkpoint = torch.load(model_path, map_location=device)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['ema_state_dict'])

    # define intensity model
    model_intensity = Intensity(model, max_time=cfg.data.T, S=cfg.data.S, denormalize=cfg.training.normalize_targets)

    ###############################
    # POST PROCESS
    ###############################

    def hellinger_distance(p, q):
        """Compute the Hellinger distance between two discrete distributions p and q."""
        return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))

    # define time grid and samples
    time_grid = torch.linspace(forward.T0, forward.T-1e-4, cfg.sampling.num_steps)
    time_grid = time_grid.to(device)

    # -------------- Sample the model ------------------
    test_indices = range(len(descrambled_test_data))[1:]  # use all test data indices
    distances = torch.zeros((len(test_indices), cfg.sampling.num_samples_per_cond))
    for (j,test_data_idx) in enumerate(test_indices):

        # extract true samples for this conditioner
        conditioner     = torch.from_numpy(descrambled_test_data[test_data_idx, 0:cfg.data.condition_dim]).view(1, cfg.data.condition_dim)
        conditioner_all = torch.tile(conditioner, (cfg.sampling.num_samples_per_cond, 1)).to(device)
        
        true_samples = torch.from_numpy(descrambled_test_data[test_data_idx, :]).long().view(1, -1).numpy()
    
        # define initial condition for sampling
        x0 = forward.sample_initial_cond(cfg.sampling.num_samples_per_cond).to(device)

        if cfg.sampling.ode_solver == 'euler':
            samples, _ = Euler(forward, conditioner_all, model_intensity, time_grid, x0, device=device)
        elif cfg.sampling.ode_solver == 'tau-leaping':
            samples, _ = TauLeaping(forward, conditioner_all, model_intensity, time_grid, x0, device=device, corrector_steps=2) #Euler(forward, conditioner, NNintensity, time_grid, x0, device=device)
        else:
            raise ValueError(f"Unknown ODE solver: {cfg.sampling.ode_solver}")

        for k in range(cfg.sampling.num_samples_per_cond):
            gen_samples_k    = torch.cat((conditioner[0,:], samples[k,:].long().detach().cpu())).view(1, -1).numpy()
            empirical_dist_p = torch.from_numpy(np.bincount(gen_samples_k.flatten().astype(np.int64), minlength=forward.S)).float()
            empirical_dist_p = empirical_dist_p / empirical_dist_p.sum()  # normalize to get probabilities
            empirical_dist_q = torch.from_numpy(np.bincount(true_samples.flatten().astype(np.int64), minlength=forward.S)).float()
            empirical_dist_q = empirical_dist_q / empirical_dist_q.sum()  # normalize to get probabilities
            distances[j,k]   = hellinger_distance(empirical_dist_p, empirical_dist_q)

    torch.save(distances, model_run+'/hellinger_distances.pt')

if __name__=='__main__':
    main()
