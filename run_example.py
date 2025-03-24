import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models import BIOTClassifier

# Load the model
x = torch.randn(16, 16, 1000)
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
embedding = biot_classifier.biot(x)
print(embedding.shape)