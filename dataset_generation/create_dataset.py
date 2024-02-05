import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import time
from torch.utils.data import Dataset
import random
import torch
import torchaudio
from torch.utils.data import DataLoader
from randomDataset import RandomDataset
import sys
sys.path.append("..")
from utils.cnn import CNNNetwork
from utils.lstm import LSTMNetwork
from utils.codec import CodecTransform
from utils.augmentation import AudioAugmentation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sr = 16000
aug = AudioAugmentation(sample_rate=sr)

src = 'denoised_speeches'
codec = CodecTransform(sr, bandwidth=6.0)
none = lambda x: x
dataset = RandomDataset(src, 16000, 10000, none, 8, 7)

augmentations = [
    ("volume_adjustment", {'factor': 1.1}),
    ("volume_adjustment", {'factor': 0.9}),
    ("volume_adjustment", {'factor': 1.3}),
    ("volume_adjustment", {'factor': 0.7}),
    ("highpass_filter", {'cutoff_freq': 100}),
    ("highpass_filter", {'cutoff_freq': 200}),
    ("lowpass_filter", {'cutoff_freq': 3000}),
    ("lowpass_filter", {'cutoff_freq': 6000})
]

first, speaker = dataset[0]
x = codec(first)
y = [speaker]

for i in range(20000):
    if i % 1000 == 0:
        print(i)
    wav, speaker = dataset[i]
    a1, a2 = random.sample(augmentations, 2)
    wav1 = aug.apply(wav, a1)
    wav2 = aug.apply(wav, a2)
    
    codec0 = codec(wav)
    codec1 = codec(wav1)
    codec2 = codec(wav2)
    x = torch.cat((x, codec0, codec1, codec2), dim=0)
    y.extend([speaker, speaker, speaker])
    
x = x.cpu().numpy()
y = np.array(y)


with open('trainX_denoised_8.npy', 'wb') as f:
    np.save(f, x)
    
with open('trainY_denoised_8.npy', 'wb') as f:
    np.save(f, y)