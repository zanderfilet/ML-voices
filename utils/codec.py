import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import nn

class CodecTransform(nn.Module):

    def __init__(self, sample_rate, bandwidth = 6.0):
        
        super(CodecTransform, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EncodecModel.encodec_model_24khz().to(self.device)
        self.model.set_target_bandwidth(bandwidth)
        self.sr = sample_rate

    def __call__(self, wav):
            
        wav = convert_audio(wav.to('cpu'), self.sr, self.model.sample_rate, self.model.channels)
        wav = wav.to(self.device).unsqueeze(0)

        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes.float() / 1024
    
    def decode(self, codes):
        codes = codes * 1024
        codes = codes.type(torch.int64)
        
        with torch.no_grad():
            reconstruction = self.model.decode([(codes, None)])[0]
            
        return reconstruction