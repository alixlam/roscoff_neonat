import os
import numpy as np
import mne 
from pathlib import Path


def interpolate( data_path ): 
    """
    data_path: path to the edf file
    """
    eeg = mne.io.read_raw_edf(data_path )

    ## create a standard montage: 
    montage = mne.channels.make_standard_montage("standard_1020")

    ## subset of electrodes from the BIOT montage
    biot_electrodes = ["Fp1", "F7", "T7", "P7", "Fp2", "F8", "T8", "P8", "F3", "C3", "P3", "F4", "C4", "P4", "O1", "O2", "A1", "A2"]

    ## subset of electrodes from the raw data
    neonat_electrodes = eeg.info.ch_names #['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'T3', 'T4', 'Cz']


    ## info object for the interpolation
    biot_info = mne.create_info(
        biot_electrodes,
        eeg.info["sfreq"],
        ch_types=["eeg"] * len(biot_electrodes),
        verbose=0,
    )
    biot_info.set_montage(montage)

    neonat_info = mne.create_info(
        neonat_electrodes,
        eeg.info["sfreq"],
        ch_types=["eeg"] * len(neonat_electrodes),
        verbose=0,
    )
    neonat_info.set_montage(montage)

    from mne.channels.interpolation import _make_interpolation_matrix

    ### compute the interpolation matrix using spherical spline interpolation (ssi)
    n_chan_from = len(neonat_electrodes) 
    n_chan_to = len(biot_electrodes)

    pos_from = np.zeros((n_chan_from, 3))
    pos_to = np.zeros((n_chan_to, 3))
    for k in range(n_chan_from):
        pos_from[k, :] = neonat_info["chs"][k]["loc"][:3]
    for k in range(n_chan_to): 
        pos_to[k, :] = biot_info["chs"][k]["loc"][:3]

    interpolation_mat = _make_interpolation_matrix(pos_from, pos_to)

    ## interpolation of the raw data
    ssi_biot_data = np.matmul(interpolation_mat, eeg.get_data())
    ## raw object from mne-python
    ssi_biot_data = mne.io.RawArray( ssi_biot_data, biot_info )

    return ssi_biot_data 

if __name__ == "__main__":
    home = os.path.expanduser("~")
    data_folder = Path("Documents/ROSCOFF/data") # folder path from home, to adapt
    patient = "01"
    epoch = "1"
    file = f"ID{patient}_epoch{epoch}"
    data_path = Path( home,  data_folder, "OTool" , "EDF_format", f"{file}.edf" ) # t

    ssi_biot_data = interpolate( data_path )
    ssi_biot_data.plot()
