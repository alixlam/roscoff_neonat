import numpy as np
import torch 
import pandas as pd
import mne
import os 
from pathlib import Path 
from models import BIOTClassifier, ReveEncoder
from pathlib import Path
from utils_data import get_positions, EA_transform
import sys
import tqdm
import argparse
# GENERATE EMBEDDINGS
########## OUTPUT FOR THE EMBEDDING ##########

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for output folder, model ID, and data ID.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--model_id', type=str, required=True, help='ID of the model to be used.')
    parser.add_argument('--data_id', type=str, required=True, help='ID of the data to be processed.')
    parser.add_argument('--EA', type=bool, required=False, default=False, help='Whether to perform Euclidian Alignment.')
    
    args = parser.parse_args()
    return args


def load_model(model_id):
    if model_id == "biot":
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=5,
            n_fft=200,
            hop_length=100,
            n_channels=18,
        )
        pretrained_model_path = "pretrained_models/EEG-six-datasets-18-channels.ckpt"
        model.biot.load_state_dict(torch.load(pretrained_model_path))
    
    elif model_id == "reve":
        model = ReveEncoder(
            patch_size=200,
            overlap_size=20,
            noise_ratio=0.000125,
            embed_dim=512,
            depth=4,
            heads=8,
            mlp_dim_ratio=2.66,
            dim_head=64,
            use_flash=False,
            geglu=True,
        )
        checkpoint_path = '/Brain/private/a19lamou/reve/reve_small_weights.pth'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint)
    return model


def generate_embeddings(data_path, model_id, data_id, output_embeds_path, EA=False):
    if data_id == "bipolar_18":
        data_path = os.path.join(data_path, "bipolar_18")
    elif data_id == "bipolar_8":
        data_path = os.path.join(data_path, "bipolar_8")
    elif data_id == "bipolar_18_interp":
        data_path = os.path.join(data_path, "bipolar_18_interp")
    else:
        raise ValueError("Invalid data")
    
    file_names = [ f for f in os.listdir(data_path) if f.endswith(".pt") ]
    # Get the embeddings from BIOT and REVE
    model = load_model(model_id)
    # Sort file names
    subjects = [f.split("_")[0] for f in file_names]
    subjects = list(set(subjects))

    # Group by subjects for Euclidian Alignment
    sub = {ID : [f for f in file_names if f.split("_")[0] == ID] for ID in subjects}

    # Generate embeddings
    for s, f in tqdm.tqdm(sub.items()): 
        if EA:
            data = [torch.load(os.path.join(data_path, file)) for file in f]
            data = EA_transform(data)
        else:
            data = [torch.load(os.path.join(data_path, file)) for file in f]
        
        for name, run in zip(f,data):
            with torch.no_grad():
                if model_id == "biot": 
                    embedding = model.biot(run)
                elif model_id == "reve":
                    pos = get_positions(data_id)
                    pos = pos.unsqueeze(0).repeat(run.shape[0], 1, 1)
                    embedding = model(run,pos)
                    embedding = embedding.mean(dim=1)
                
                torch.save(embedding, os.path.join(output_embeds_path, f"{name[:-3]}.pt" ) )

if __name__ == "__main__":
    args = parse_arguments()
    output_path = os.path.join(args.output_folder, "embeddings", args.model_id, args.data_id, "EA" if args.EA else "no_EA")
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    generate_embeddings(args.data_folder, args.model_id, args.data_id, output_path, args.EA)