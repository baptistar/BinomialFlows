import matplotlib
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
import matplotlib.pyplot as plt
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
    # parser.add_argument(
    #     "--test_idx",
    #     type=list,
    #     required=False,
    #     help="Indices as a list [10, 12, 14]"
    # )
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

    test_data_idx_all = np.arange(9)

    # define array to save samples
    samples_all = torch.zeros((len(test_data_idx_all), cfg.data.dim))

    # -------------- Sample the model ------------------
    for (k,test_data_idx) in enumerate(test_data_idx_all): #args.test_idx:
        conditioner = torch.from_numpy(descrambled_test_data[test_data_idx, 0:cfg.data.condition_dim]).to(device).view(1, cfg.data.condition_dim)
        # define initial condition for sampling
        x0 = forward.sample_initial_cond(1)
        # define time grid and samples
        time_grid = torch.linspace(forward.T0, forward.T-cfg.sampling.early_stopping, cfg.sampling.num_steps)
        if cfg.sampling.ode_solver == 'euler':
            samples, _ = Euler(forward, conditioner, model_intensity, time_grid, x0, device=device)
        elif cfg.sampling.ode_solver == 'tau-leaping':
            samples, _ = TauLeaping(forward, conditioner, model_intensity, time_grid, x0, device=device, corrector_steps=2)
        else:
            raise ValueError('Sampler not recognized')
        samples_all[k,:] = torch.cat((conditioner.cpu(), samples), dim=1)
    # save results
    torch.save(samples_all, model_run + f'/samples.pt')

    plt.figure(figsize=(15,9))
    for k in range(len(test_data_idx_all)):
        plt.subplot(3,3,k+1)
        plt.scatter(np.arange(256)/16, samples_all[k,:].cpu().numpy(), alpha=0.5, label='Generated sample')
        plt.axvline(x=2.0, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Bar')
        plt.ylabel('Notes')
        plt.xlim(0,16)
        plt.grid()
        plt.title('Sample '+str(k), fontsize=14)
        plt.tight_layout()
    file_name = model_run + '/samples_plot.pdf'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    main()