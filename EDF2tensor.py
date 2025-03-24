# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:35:47 2025

@author: lucas
"""

import mne 
import os 
import random
import torch
import numpy as np

mne.set_log_level(verbose='ERROR')

#%%
bipolar_pairs = [
    ('F3', 'C3', 'F3-C3'),
    ('C3', 'O1', 'C3-O1'),
    ('F4', 'C4', 'F4-C4'),
    ('C4', 'O2', 'C4-O2'),
    ('T3', 'C3', 'T3-C3'),
    ('T4', 'C4', 'T4-C4'),
    ('T3', 'O1', 'T3-O1'),
    ('T4', 'O2', 'T4-O2'),
    ('F3', 'F4', 'F3-F4'),
    ('C3', 'C4', 'C3-C4'),
    ('O1', 'O2', 'O1-O2'),
    ('T3', 'T4', 'T3-T4'),
    ('F3', 'Cz', 'F3-Cz'),
    ('F4', 'Cz', 'F4-Cz'),
    ('C3', 'Cz', 'C3-Cz'),
    ('C4', 'Cz', 'C4-Cz'),
    ('F3', 'T3', 'F3-T3'),
    ('F4', 'T4', 'F4-T4'),
]

def randomEpochs(raw):
    n_epochs = 10
    epoch_length = 10  # seconds
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

file_path = 'C:/Users/lucas/Documents/THESE/Neonat/OTooleetal.ScientificData_2023/EDF_format/'
file_save = 'C:/Users/lucas/Documents/THESE/Neonat/OTooleetal.ScientificData_2023/TensorFormat/'

name_files = os.listdir(file_path)
for i in range(len(name_files)):
# for i in range(1):

    print(name_files[i])
    raw = mne.io.read_raw_edf(file_path + name_files[i], preload=True, verbose=False)

    if raw.info['sfreq'] != 200:
        raw.resample(200)

    # Apply each bipolar refererance
    for anode, cathode, new_name in bipolar_pairs:
        # raw = raw.set_bipolar_reference(anode=anode, cathode=cathode, ch_name=new_name, drop_refs=False)

        raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode,ch_name=new_name,drop_refs=False)
    raw.pick_channels([name for _, _, name in bipolar_pairs])

    tensor_data = randomEpochs(raw)

    torch.save(tensor_data, file_save+name_files[i][:-4]+".pt")



