import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random
from encodec.utils import convert_audio


class SoundDataset(Dataset):

    def __init__(self,
                 src_directories,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device='cpu'):

        self.src_directories = src_directories
        all_names = []
        all_labels = []
        for direc in self.src_directories:
            files = os.listdir(f'../datasets/{direc}')
            labels = [int(f[0]) for f in files if f[0] != '.']
            files = [f'{direc}/{f}' for f in files if f[0] != '.']
            all_names += files
            all_labels += labels

        
        combined = list(zip(all_names, all_labels))
        random.shuffle(combined)
        self.file_names = combined
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name, label = self.file_names[index]
        signal, sr = torchaudio.load(f'../datasets/{file_name}')
        signal = signal.to(self.device)
        signal = convert_audio(signal, sr, self.target_sample_rate, 1)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal