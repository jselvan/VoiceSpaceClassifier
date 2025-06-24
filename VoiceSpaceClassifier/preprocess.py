import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample

TARGET_SAMPLE_RATE = 16000

transform = MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_mels=40,
    n_fft=1024,
    hop_length=512
)

def preprocess(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != TARGET_SAMPLE_RATE:
        resample = Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resample(waveform)
    mel_spec = transform(waveform).log2().clamp(min=-20)
    return mel_spec  # (1, n_mels, time_frames)
