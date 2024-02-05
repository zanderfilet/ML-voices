import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random
import IPython.display as ipd
import matplotlib.pyplot as plt


def pitch_shift(audioObject): 
    pass
    




def load_audio(filepath): 
    waveform, sample_rate = torchaudio.load(filepath, normalize=True) 
    ipd.Audio(filepath)
    plt.plot(waveform)
    plt.show()
    
load_audio("../datasets/dylanMnist/7_dylan.wav")


    
    
    