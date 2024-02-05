import torch
import torchaudio
from torchaudio.transforms import Resample, TimeStretch, TimeMasking, FrequencyMasking

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def time_stretch(self, waveform, rate=0.9):
        stretch = TimeStretch(hop_length=None, fixed_rate=rate)
        return stretch(waveform)

    def pitch_shift(self, waveform, n_steps=2):
        return torchaudio.functional.pitch_shift(waveform, self.sample_rate, n_steps)

    def add_noise(self, waveform, noise_factor=0.005):
        noise = torch.randn(waveform.size()) * noise_factor
        return waveform + noise

    def volume_adjustment(self, waveform, factor=1.2):
        return waveform * factor
    
    def lowpass_filter(self, waveform, cutoff_freq =3000, q=0.707):
        return torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff_freq, q)

    def highpass_filter(self, waveform, cutoff_freq=200, q=0.707):
        return torchaudio.functional.highpass_biquad(waveform, self.sample_rate, cutoff_freq, q)

    def apply(self, waveform, augmentation):
        func = getattr(self, augmentation[0])

        return func(waveform, **augmentation[1])