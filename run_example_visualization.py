import os 
import torch
import numpy as np
from models import BIOTClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import hashlib
from collections import defaultdict
from utils_data import EA_transform, get_positions
from models.reve import ReveEncoder

# Load model and cache it
@st.cache_resource
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
        model.eval()
    
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

# Load dataset and group by subject
@st.cache_data
def load_dataset(data_id, model_id):
    if data_id == "bipolar_18":
        data_path = "/Brain/private/OTooleetal.ScientificData_2023/Tensor20_format/bipolar_18/"
    elif data_id == "bipolar_8":
        data_path = "/Brain/private/OTooleetal.ScientificData_2023/Tensor20_format/bipolar_8/"
    elif data_id == "bipolar_18_interp":
        data_path = "/Brain/private/OTooleetal.ScientificData_2023/Tensor20_format/bipolar_18_interp/"
    else:
        raise ValueError("Invalid data")

    files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
    files.sort()

    data = []
    subjects = []
    subject_to_runs = defaultdict(list)

    for f in files:
        tensor = torch.load(os.path.join(data_path, f))  # shape: [B, T, C]
        subj_id = f[:4]
        data.append(tensor)
        subjects.extend([subj_id] * 10)
        subject_to_runs[subj_id].append(tensor)

    df = pd.read_csv('/Brain/private/OTooleetal.ScientificData_2023/eeg_grades.csv')
    grades = [int(df[df['file_ID'] == f[:-3]]['grade'].values[0]) for f in files]
    grades = [grade for grade in grades for _ in range(tensor.shape[0])]

    if model_id == "reve":
        pos = get_positions(data_id)
        pos = pos.unsqueeze(0).repeat(len(data)*10, 1, 1)
    else:
        pos = None

    return data, subjects, grades, subject_to_runs, pos


# Apply EA to each subject
def apply_EA(subject_to_runs: dict) -> list[torch.Tensor]:
    aligned_data = []
    for subj, runs in subject_to_runs.items():
        aligned = EA_transform(runs)  # list of (10, c, t) tensors
        aligned = torch.cat(aligned, dim=0)  # (N, C, T)
        aligned_data.append(aligned)
    return aligned_data  # list of (N, C, T) tensors

# Hash array for caching
def get_array_hash(array: np.ndarray) -> str:
    return hashlib.md5(array.view(np.uint8)).hexdigest()

# Generate embeddings (EA or not) 
@st.cache_data
def get_embeddings(data_id, model_id, EA=False):
    data, subjects, grades, subject_to_runs, pos = load_dataset(data_id, model_id)

    if EA:
        data = apply_EA(subject_to_runs)  # list of (N, C, T)

    tensor_data = torch.cat(data, dim=0)

    model = load_model(model_id)
    if model_id == "reve":
        embeddings = model(tensor_data, pos)
        embeddings = embeddings.mean(dim=1).detach().numpy()
    else:
        embeddings = model.biot(tensor_data).detach().numpy()

    return embeddings, subjects, grades

# Run TSNE and cache
@st.cache_data
def run_tsne(embedding_hash: str, embeddings: np.ndarray):
    return TSNE(n_components=2, perplexity=50).fit_transform(embeddings)

# Streamlit app UI
st.title("BIOT Embedding Visualization with Euclidean Alignment")

# Dataset selection
data_id = st.selectbox('Select the dataset', ('bipolar_18', 'bipolar_8', 'bipolar_18_interp'))


# Model selection
model_id = st.selectbox('Select the model', ('biot', 'reve'))

# Load embeddings (with and without EA)
emb_raw, subjects, grades = get_embeddings(data_id, model_id, EA=False)
emb_EA, _, _ = get_embeddings(data_id, model_id, EA=True)

# Compute hashes
hash_raw = get_array_hash(emb_raw)
hash_EA = get_array_hash(emb_EA)

# Run t-SNE
X_tsne_raw = run_tsne(hash_raw, emb_raw)
X_tsne_EA = run_tsne(hash_EA, emb_EA)

# UI options
color_option = st.selectbox('Select the color', ('Subjects', 'Grades'))
hue = subjects if color_option == 'Subjects' else grades
n_points = st.slider('Number of points', 0, len(X_tsne_raw), 100)

# Create plots
fig1 = plt.figure()
ax1 = sns.scatterplot(x=X_tsne_raw[:n_points, 0], y=X_tsne_raw[:n_points, 1], hue=hue[:n_points])
sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
plt.title(f"Raw BIOT embeddings - {data_id} - {color_option}")

fig2 = plt.figure()
ax2 = sns.scatterplot(x=X_tsne_EA[:n_points, 0], y=X_tsne_EA[:n_points, 1], hue=hue[:n_points])
sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
plt.title(f"EA BIOT embeddings - {data_id} - {color_option}")

# Display side by side
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

