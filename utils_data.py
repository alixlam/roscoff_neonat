
import os 
import random
import torch
import numpy as np
import mne
mne.set_log_level(verbose='ERROR')
from scipy.linalg import sqrtm, inv,pinv

BIPOLAR_18 = [
    ('F3', 'T3', 'F3-T3'),
    ('T3', 'O1', 'T3-O1'),
    ('F3', 'C3', 'F3-C3'),
    ('C3', 'O1', 'C3-O1'),
    ('F3', 'Cz', 'F3-Cz'),
    ('Cz', 'O1', 'Cz-O1'),
    ('F4', 'T4', 'F4-T4'),
    ('T4', 'O2', 'T4-O2'),
    ('F4', 'C4', 'F4-C4'),
    ('C4', 'O2', 'C4-O2'),
    ('F4', 'Cz', 'F4-Cz'),
    ('Cz', 'O2', 'Cz-O2'),
    ('Cz', 'C3', 'Cz-C3'),
    ('C3', 'T3', 'C3-T3'),
    ('Cz', 'C4', 'Cz-C4'),
    ('C4', 'T4', 'C4-T4'),
    ('F3', 'F4', 'F3-F4'),
    ('O2', 'O1', 'O2-O1')
]

BIPOLAR_8 = [
    ('F3', 'C3', 'F3-C3'),
    ('T3', 'C3', 'T3-C3'),
    ('O1', 'C3', 'O1-C3'),
    ('C3', 'Cz', 'C3-Cz'),
    ('Cz', 'C4', 'Cz-C4'),
    ('F4', 'C4', 'F4-C4'),
    ('T4', 'C4', 'T4-C4'),
    ('O2', 'C4', 'O2-C4')
]

BIPOLAR_18_interp = [
    ('FP1', 'F7', 'FP1-F7'),
    ('F7', 'T7', 'F7-T7'),
    ('T7', 'P7', 'T7-P7'),
    ('P7', 'O1', 'P7-O1'),
    ('FP2', 'F8', 'FP2-F8'),
    ('F8', 'T8', 'F8-T8'),
    ('T8', 'P8', 'T8-P8'),
    ('P8', 'O2', 'P8-O2'),
    ('FP1', 'F3', 'FP1-F3'),
    ('F3', 'C3', 'F3-C3'),
    ('C3', 'P3', 'C3-P3'),
    ('P3', 'O1', 'P3-O1'),
    ('FP2', 'F4', 'FP2-F4'),
    ('F4', 'C4', 'F4-C4'),
    ('C4', 'P4', 'C4-P4'),
    ('P4', 'O2', 'P4-O2'),
    ('C3', 'A2', 'C3-A2'),
    ('C4', 'A1', 'C4-A1')
]

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

def EA_transform(Xlist):
    X = torch.cat(Xlist,0)
    sqrt_R_s = compute_r_op(X).float()
    new_X = [torch.einsum("fe,bet->bft",sqrt_R_s, x) for x in Xlist]
    return new_X

def generate_tensors(basepath, savepath, n_epochs=10, epoch_length=5, bipolar_pairs=None):
    """
    Generate tensors from EDF files and save them to disk.
    basepath: str, path to the directory containing the EDF files
    savepath: str, path to the directory where the tensors will be saved
    n_epochs: int, number of epochs to extract
    epoch_length: int, length of each epochs in seconds
    """
    name_files = os.listdir(basepath)
    
    for i in range(len(name_files)):
        file = name_files[i]
        raw = mne.io.read_raw_edf(basepath + name_files[i], preload=True, verbose=False)

        if raw.info['sfreq'] != 200:
            raw.resample(200)

        # Apply each bipolar refererance
        for anode, cathode, new_name in bipolar_pairs:
            # raw = raw.set_bipolar_reference(anode=anode, cathode=cathode, ch_name=new_name, drop_refs=False)

            raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode,ch_name=new_name,drop_refs=False)
        raw.pick_channels([name for _, _, name in bipolar_pairs])

        tensor_data = randomEpochs(raw)
        tensor_data = tensor_data / (
                np.quantile(
                    np.abs(tensor_data), q=0.95, method="linear", axis=-1, keepdims=True
                )
                + 1e-8
            )
        tensor_data = torch.FloatTensor(tensor_data)
        if os.path.exists(savepath) == False:
            os.makedirs(savepath)
        torch.save(tensor_data, os.path.join(savepath, file[:-4] + ".pt"))
    return

if __name__ == "__main__":
    import os
    data_path = "/Brain/private/OTooleetal.ScientificData_2023/"
    save_path = "/Brain/private/OTooleetal.ScientificData_2023/"

    # Generate for BIPOLAR 18
    generate_tensors(os.path.join(data_path, "EDF_format/"), os.path.join(save_path, "bipolar_18"),bipolar_pairs=BIPOLAR_18)
    generate_tensors(os.path.join(data_path, "EDF_format/"), os.path.join(save_path, "bipolar_8"),bipolar_pairs=BIPOLAR_8)
    #generate_tensors(os.path.join(data_path, "EDF_format_interp"), os.path.join(save_path, "bipolar_18_interp"),bipolar_pairs=BIPOLAR_18_interp)


    
