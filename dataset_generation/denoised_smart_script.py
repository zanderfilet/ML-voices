import numpy as np
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr
from scipy.io import wavfile

src_dir = 'speeches/'
paths = os.listdir(src_dir)

# denoise
for path in paths:
   sr, data = wavfile.read(src_dir + path)
   reduced_noise = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.33)
   reduced_noise_path = f'denoised_speeches/{path}'
   wavfile.write(reduced_noise_path, sr, reduced_noise)


src_dir = 'denoised_speeches/'
paths = os.listdir(src_dir)

# split denoised
for path in paths:
   sound = AudioSegment.from_file(file = src_dir + path, format="wav")
   chunked = split_on_silence(sound, min_silence_len=250, silence_thresh=-40)

   for i, chunk in enumerate(chunked):
      output_file = f'denoised_smart_split_speeches/{path[:-4]}_chunk{i}.wav'
      print("Exporting file: ", output_file)
      chunk.export(out_f=output_file, format="wav")
