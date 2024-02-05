import os
from torch.utils.data import Dataset
import torchaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr
import re


class smartDataset(Dataset):

    def __init__(self, src_dir, sr, transformation):
        super().__init__()
        # self.dataset_size = dataset_size
        # self.max_len
        self.sr = sr
        self.clips = []
        self.labels = []
        self.dataset_size = 0
        self.transformation = transformation

        for file in os.listdir(src_dir):
            if file[0] == '.':
                continue
            waveform, sample_rate = self.load_audio(f"{src_dir}/{file}")
            assert sample_rate == self.sr
            self.clips.append(waveform[0])
            self.labels.append(re.match(r"(.*)_chunk", file)[1])
            self.dataset_size += 1

    def load_audio(self, filepath):
        return torchaudio.load(filepath, normalize=True)

    def __getitem__(self, idx):
        return self.clips[idx], self.labels[idx]

    def __len__(self):
        return self.dataset_size



nfft=512
nmels=60
mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20, melkwargs={'n_fft':nfft, 'n_mels':nmels})
dataSet = smartDataset("Smart_Split_Speeches", 16000, mfcc)
print(dataSet[0])