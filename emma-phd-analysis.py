"""
Emma PhD Analysis

@author: Emma & Chloe 2023
"""

import mne
import numpy as np
import os.path as op
import glob
import pandas as pd

from mne.time_frequency import tfr_morlet

#%matplotlib

#to create a list of raw files to use in a loop (avoid for now)
#raw_files = glob.glob('data/Pasithea_RESTING/*_RESTING.vhdr')

# Set parameters for EOG and reference

eye_channels = ['FP1','FP2']
ref_channels = ['TP9','TP10']

#set position of electrodes
montage = 'standard_1020'

#Pre-Processing

subj = op.split(r)[1][0:4] # extract first 3 characters (i.e., CP01)
session = op.split(r)[1][6:7]

raw = mne.io.read_raw_brainvision(r, eog=eye_channels, preload=True)   



#from the internet (specific to xdf): https://mne.tools/dev/auto_examples/io/read_xdf.html
import pyxdf
import mne
from mne.datasets import misc
#
fname = "data\p2_Bob1_main_eeg.xdf"
streams, header = pyxdf.load_xdf(fname)
data = streams[2]["time_series"].T

print(data[:10])
assert data.shape[0] == 16  # four raw EEG plus one stim channel
#data[:4:2] -= data[1:4:2]  # subtract (rereference) to get two bipolar EEG

data[:14:2] -= data[1:15:2]
data = data[:2:]  # subselect

data[:16] *= 1e-6 / 50 / 2  # uV -> V and preamp gain
sfreq = float(streams[2]["info"]["nominal_srate"][0])
info = mne.create_info(16, sfreq, "eeg")
raw = mne.io.RawArray(data, info)
raw.plot(scalings=dict(eeg=100e-6), duration=2, start=2*60)
# print(raw)
# fname = (
#     misc.data_path() / 'xdf' /
#     'sub-P001_ses-S004_task-Default_run-001_eeg_a2.xdf')
# streams, header = pyxdf.load_xdf(fname)
# data = streams[0]["time_series"].T
# assert data.shape[0] == 5  # four raw EEG plus one stim channel
# data[:4:2] -= data[1:4:2]  # subtract (rereference) to get two bipolar EEG
# data = data[::2]  # subselect
# data[:2] *= (1e-6 / 50 / 2)  # uV -> V and preamp gain
# testdata = data[:20]
# sfreq = float(streams[0]["info"]["nominal_srate"][0])
# info = mne.create_info(3, sfreq, ["eeg", "eeg", "stim"])
# raw = mne.io.RawArray(data, info)
# raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14)


 

raw.drop_channels(['FP1', 'FP2', 'x_dir', 'y_dir', 'z_dir'])
    
raw.set_montage(montage)
     
# re-reference our EEG to linked mastoids
raw = mne.io.set_eeg_reference(raw,ref_channels)[0]
    
raw.set_channel_types({'TP9': 'misc', 'TP10': 'misc'})
        
# downsample to 250 Hz
raw = raw.resample(250)
    
# apply a basic preprocessing step to the data i.e., filter from 1-40 Hz.
#  raw = raw.filter(1, 40.,
#                 l_trans_bandwidth='auto',
#                     h_trans_bandwidth='auto',
#                     filter_length='auto',
#                     method='fir',
#                     fir_window='hamming',
#                     phase='zero',
#                  n_jobs=2)


