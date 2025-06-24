from VoiceSpaceClassifier.preprocess import preprocess

from torch.utils.data import Dataset

from pathlib import Path
import re

class SpeakerDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = preprocess(self.file_paths[idx])  # shape: (1, mel, time)
        y = self.labels[idx]
        return x, y

    @classmethod
    def from_directory(cls, directory, template, glob_pattern='*.wav', transform=None):
        directory = Path(directory)
        files = list(directory.glob('*.wav'))
        labels = [re.findall(template, file.stem)[0] for file in files]
        file_paths= [file.as_posix() for file in files]
        return cls(file_paths, labels, transform)