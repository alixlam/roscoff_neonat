import numpy as np
import torch 
import pandas as pd
import mne
import os 
from pathlib import Path 
from models import BIOTClassifier
from pathlib import Path

"""

First : load data and extract the embeddings 
Second: use the dataset_folds to create the tensors for train and test for each fold.


!! Change the paths if necessary

!! sorry the code is ugly, if problem contact Sarah
"""

mne.set_log_level(verbose='ERROR')
home = os.path.expanduser("~")



# GENERATE EMBEDDINGS
########## OUTPUT FOR THE EMBEDDING ##########
output_embeds_path = Path( home, "Documents" , "ROSCOFF", "data", "OTool", "TensorFormatEmbeds" )
os.makedirs( output_embeds_path, exist_ok=True)


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

data_path = Path( home, "Documents" , "ROSCOFF", "data", "OTool", "TensorFormat" )


file_names = [ f for f in os.listdir(data_path) if f.endswith(".pt") ]
# Get the embeddings from BIOT
for f in file_names: 
    data = torch.load(os.path.join(data_path, f))
    with torch.no_grad(): 
        embedding = biot_classifier.biot(data)
    torch.save(embedding, Path( output_embeds_path, f"{f[:-3]}.pt" ) )

#################### GENERATE DATA TENSORS ####################
data_type       = "TensorFormatEmbeds"
output_folder   = "TensorFormatFolds"
data_folder     = Path( "Documents", "ROSCOFF", "data", "OTool" )

data_path   = Path( home, data_folder, data_type )
output_path = Path(home, data_folder, output_folder)


## this function is not used here but can be used if generating dataset_folds is necessary ##
def fold_to_dataset(fold_file="5_folds_cross_validation.npy", metadata_file="metadata.csv", output_file="dataset_folds.npy"):
    """
    Convert the fold repartition of the subjects to a fold repartition of the data such that classes are balanced.
    Inputs: 
    - fold_file: name of the fold repartition of the subjects file (c.f. cross_validation_stratification.ipynb)
    - metadata_file: name of the metadata file
    - output_file: name of the file to save with data repartition

    Return : 
    - dataset_folds: repartition of the data for each fold.
    """
    
    folds = np.load(fold_file, allow_pickle=True).item()
    metadata = pd.read_csv(metadata_file)

    dataset_folds = {}

    for k in range(len(folds)):  # fold
        train_subjects = folds[k]['train']
        test_subjects = folds[k]['test']

        dataset_folds[k] = {"train": {"file_ids": [], "labels": []},
                            "test": {"file_ids": [], "labels": []}}

        for grade in [4, 3, 2, 1]:  # iterate over grades
            # Filter train and test subjects by grade
            train = metadata[metadata['baby_ID'].isin(train_subjects)]
            test = metadata[metadata['baby_ID'].isin(test_subjects)]

            train_subset = train[train['grade'] == grade]
            test_subset = test[test['grade'] == grade]

            # Determine the number of samples for train and test
            if grade==4: 
                N_train = folds[k]["train_label_counts"][-1]
                N_test = folds[k]["test_label_counts"][-1]

            # Randomly sample train and test data
            if len(train_subset) >= N_train:
                train_rand_idx = np.random.choice(train_subset.index, N_train, replace=False)
                dataset_folds[k]["train"]["file_ids"].extend(train_subset.loc[train_rand_idx, 'file_ID'].to_numpy())
                dataset_folds[k]["train"]["labels"].extend([grade] * N_train)

            if len(test_subset) >= N_test:
                test_rand_idx = np.random.choice(test_subset.index, N_test, replace=False)
                dataset_folds[k]["test"]["file_ids"].extend(test_subset.loc[test_rand_idx, 'file_ID'].to_numpy())
                dataset_folds[k]["test"]["labels"].extend([grade] * N_test)

    # Save the dataset folds to a file
    np.save(output_file, dataset_folds)
    print(f"Dataset folds saved to {output_file}")

    return dataset_folds
##########

def folds_to_tensor(dataset_folds, output_path, emb_length=256, n_trials_per_ep=10): 
    """
    Generate tensors of data and labels for each of the folds in dataset_folds
    Inputs: 
    - dataset_folds: npy array of the dataset folds
    - output_paht: path of where to save the tensors
    - data_length: number of the sample in the temporal dimension
    - n_electrodes: number of electrodes
    - n_trials_per_ep: number of trials per epoch
    
    The tensor save for each fold is a tuple such as t[0] corresponds to the data and t[1] corresponds to the labels.
    Data is not shuffled. 
    """

    os.makedirs( Path(output_path), exist_ok=True )
    for k in dataset_folds:
        c = 0
        # Train data
        train_file_ids = dataset_folds[k]["train"]["file_ids"]
        train_labels = dataset_folds[k]["train"]["labels"]
        train_tensor = torch.zeros( (10*len(train_file_ids), emb_length) ) 
        train_label_tensor = torch.zeros( (10*len(train_file_ids), 1) ) 
        n_train = len(train_file_ids)
        ## load data
        for i,f in enumerate(train_file_ids): 
            train_tensor[c:c+n_trials_per_ep,:] = torch.load( Path(data_path, f"{f}.pt" ) )
            train_label_tensor[c:c+n_trials_per_ep,0] = train_labels[i]
            c += n_trials_per_ep
    
        # Test data
        c = 0
        test_file_ids = dataset_folds[k]["test"]["file_ids"]
        test_labels = dataset_folds[k]["test"]["labels"]
        test_tensor = torch.zeros( (10*len(test_file_ids), emb_length) ) 
        test_label_tensor = torch.zeros( (10*len(test_file_ids), 1) ) 
        n_train = len(test_file_ids)
        ## load data
        for i,f in enumerate(test_file_ids): 
            test_tensor[c:c+n_trials_per_ep,: ] = torch.load( Path(data_path, f"{f}.pt" ) )
            test_label_tensor[c:c+n_trials_per_ep,0] = test_labels[i]
            c += n_trials_per_ep
    
        torch.save((train_tensor, train_label_tensor), Path( output_path, f"train_fold_{k}.pt" ) )
        torch.save((test_tensor, test_label_tensor), Path( output_path, f"test_fold_{k}.pt" ) )

    print("Tensors for train and test data saved for each fold.")


### GENERATE TENSORS ###
dataset_folds = np.load("dataset_folds.npy", allow_pickle=True).item()
folds_to_tensor(dataset_folds=dataset_folds, output_path=output_path, emb_length=256, n_trials_per_ep=10)