# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:35:47 2025

@author: lucas
"""

import os 
import random
import torch
import numpy as np
from models import BIOTClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import mne
import seaborn as sns
import pandas as pd
mne.set_log_level(verbose='ERROR')

# GENERATE EMBEDDINGS
data_id = "bipolar_18"
pretrained_model_path = "pretrained_models/EEG-six-datasets-18-channels.ckpt"

biot_classifier = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=5,
        n_fft=200,
        hop_length=100,
        n_channels=18,  # here is 18
    )
biot_classifier.biot.load_state_dict(torch.load(pretrained_model_path))


if data_id == "bipolar_18":
    data_path = "/Brain/private/OTooleetal.ScientificData_2023/bipolar_18/"
elif data_id == "bipolar_8":
    data_path = "/Brain/private/OTooleetal.ScientificData_2023/bipolar_8/"
elif data_id == "bipolar_18_interp":
    data_path = "/Brain/private/OTooleetal.ScientificData_2023/bipolar_18_interp/"
else:
    raise ValueError("Invalid data")


data = [torch.load(os.path.join(data_path, f)) for f in os.listdir(data_path) if f.endswith(".pt")]
# Get subject ID
subjects = [f[:4] for f in os.listdir(data_path) if f.endswith(".pt")]
# Repeat 10 times each subject since 10 trials per subject
subjects = [name for name in subjects for _ in range(10)]

# Get grade
df = pd.read_csv('/Brain/private/OTooleetal.ScientificData_2023/eeg_grades.csv')
grades = df["grade"].values
# Repeat 10 times each subject
grades = [grade for grade in grades for _ in range(10)]

# Get the embeddings from BIOT
tensor_data = torch.cat(data, 0)
embedding = biot_classifier.biot(tensor_data)

# Project in lower dimensions
X_emb = TSNE(n_components=2).fit_transform(embedding.detach().numpy())

# Plot the embeddings
save_path = "tmp"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Don't plot all the data when coloring by subjects id 
sns.scatterplot(x=X_emb[:200, 0], y=X_emb[:200, 1], hue=subjects[:200])
plt.title(f"BIOT embeddings - {data_id} - subjects")
plt.savefig(f"{save_path}/BIOT_{data_id}_subjects.png")
plt.close()


sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=grades)
plt.title(f"BIOT embeddings - {data_id} - grades")
plt.savefig(f"{save_path}/BIOT_{data_id}_grades.png")
