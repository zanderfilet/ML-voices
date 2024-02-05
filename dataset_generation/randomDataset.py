import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import time
from torch.utils.data import Dataset
import random
import torch
import torchaudio
import sys
sys.path.append('..')
from utils import LABELS

class RandomDataset(Dataset):

    def __init__(self, src_dir, sr, dataset_size, transformation, max_len_sec, min_len_sec=1):
        super().__init__()
        self.max_len = max_len_sec
        self.min_len = min_len_sec
        self.dataset_size = dataset_size
        self.sr = sr
        self.raw_audio = []
        self.speeches = []
        self.transformation = transformation

        files = os.listdir(src_dir)
        files.sort()
        for file in files:
            if file[0] == '.':
                continue
            waveform, sample_rate = self.load_audio(f"{src_dir}/{file}")
            assert sample_rate == self.sr
            self.speeches.append(file)

            self.raw_audio.append(waveform[0])
            
    def load_audio(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath, normalize=True)
        return waveform, sample_rate

    def __getitem__(self, index):
        # which file to take a random clip from
        speaker = index % len(self.raw_audio)
        clip_length = self.sr*np.random.randint(self.min_len, self.max_len)

        if clip_length > len(self.raw_audio[speaker]):
            return self.raw_audio[speaker]

        starting_loc = np.random.randint(
            0, len(self.raw_audio[speaker]) - clip_length)

        wav = self.raw_audio[speaker][starting_loc:starting_loc+clip_length].unsqueeze(0)
        signal = self.transformation(wav)
        label = None
        for i, name in enumerate(LABELS):
            if name.lower() in self.speeches[speaker].lower():
                label = i
        return signal, label

    def __len__(self):
        return self.dataset_size