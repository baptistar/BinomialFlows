from tabnanny import check
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
import numpy as np
import os
import yaml
from datetime import datetime

sys.path.append('../src')
from forward import PoissonFolmer
from loss import denoising_loss
from denoiser import load_model
from utils import create_run_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#pwd = '/scratch/baptista/GenerativeDiscreteDiffusionBPS/music'
pwd = '.'
run_dir = create_run_dir(pwd + '/runs')

# load config file
with open(pwd + "/config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(cfg_dict)

####################################
# DEFINE DATA
####################################

# load data
data_folder = pwd+'/data/'
train_data  = np.load(data_folder + 'train.npy')

# descramble data back to original ordering
descramble_key = np.loadtxt(data_folder + 'descramble_key.txt')
def descramble(samples):
    return descramble_key[samples.flatten()].reshape(*samples.shape)
descrambled_train_data  = descramble(train_data)

# compute mean and variance
mean_data = (descrambled_train_data).mean()
var_data  = (descrambled_train_data).var()

# define dataloader
dataloader = torch.utils.data.DataLoader(descrambled_train_data,
    batch_size=cfg.training.batch_size,
    shuffle=cfg.data.shuffle)

####################################
# TRAIN MODEL
####################################

# define model
model, EMAmodel = load_model(cfg, mean_data, var_data, device)

# define forward process and model
forward = PoissonFolmer(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim-cfg.data.condition_dim)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), cfg.training.lr, 
                            weight_decay=cfg.training.weight_decay)

# EMA update function
def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    EMAmodel.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch    = checkpoint["epoch"] + 1
    training_loss  = checkpoint["loss"]
    grad_norms     = checkpoint["grad_norms"]

    print(f"Resumed from epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch.")
    start_epoch    = 0 
    training_loss  = []
    grad_norms     = []

for ep in range(start_epoch, cfg.training.n_epochs):
    for minibatch in dataloader:
        
        optimizer.zero_grad()
        l = denoising_loss(forward, model, cfg, minibatch)
        l.backward()
        
        if cfg.training.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms.append(grad_norm.item())
        optimizer.step()

        update_ema(model, EMAmodel, cfg.model.ema_decay)

        training_loss.append(l.item())
    
    if ep % cfg.training.save_freq == 0:
        print(f'Epoch {ep}, Loss: {l.item()}')
        # save model after training
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": EMAmodel.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": training_loss,
            "grad_norms": grad_norms,
        }, checkpoint_path)
        # save config as yaml file
        cfg_dict = cfg.to_dict()
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)
