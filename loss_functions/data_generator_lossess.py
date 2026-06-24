import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% ========= Generator Loss ==========
def loss_generator(fake_inputs, teacher, batch_size=64, metadata=None):

    h = teacher.initHidden(fake_inputs.shape[1]).to(device)
    t_logits, t_features = teacher(fake_inputs, h)

    feat_mean = t_features.mean(dim=0); feat_var = t_features.var(dim=0)
    if metadata is None:
        metadata = torch.load('./saved_models/metadata.pt')
    meta_mean = metadata.mean(dim=0); meta_var = metadata.var(dim=0)

    metadata_loss = F.mse_loss(feat_mean, meta_mean) + F.mse_loss(feat_var, meta_var)
   
    return metadata_loss