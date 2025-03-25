
import os 
import random
import torch
import numpy as np
import mne
mne.set_log_level(verbose='ERROR')
from scipy.linalg import sqrtm, inv,pinv


def randomEpochs(raw, n_epochs=10, epoch_length=5):
    """
    Extract random epochs from a raw MNE object.
    raw: mne.io.Raw
    n_epochs: int, number of epochs to extract
    epoch_length: int, length of each epochs in seconds
    """
    sfreq = raw.info['sfreq']  # sampling frequency
    total_duration = raw.times[-1]  # total duration in seconds

    # Make sure we have enough room to extract 10 full epochs
    max_start_time = total_duration - epoch_length
    start_times = sorted(random.sample(range(int(max_start_time)), n_epochs))

    epochs = []
    for start in start_times:
        segment = raw.copy().crop(tmin=start, tmax=start + epoch_length)
        data, _ = segment[:]
        epochs.append(data[:,:-1])  # shape: (n_channels, n_times)

    # Stack into a tensor: shape (n_epochs, n_channels, n_times)
    tensor_data = torch.tensor(np.stack(epochs), dtype=torch.float32)
    return tensor_data

def compute_r_op(X):
    r = torch.einsum('bet, tab -> bea',X,X.T).mean(0)
    r_op = torch.from_numpy(inv(sqrtm(r)))
    return r_op

def EA_transform(X):
    new_X =[]
    sqrt_R_s = compute_r_op(X).float()
    new_X = torch.einsum("fe,bet->bft",sqrt_R_s, X)
    return new_X