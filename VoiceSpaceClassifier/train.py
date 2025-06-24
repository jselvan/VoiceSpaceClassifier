from VoiceSpaceClassifier.model import ConvSpeakerNet
from VoiceSpaceClassifier.dataset import SpeakerDataset

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from pathlib import Path

def train(directory, template, model_path=None, ftype='wav', device='cpu', n_epochs=100):
    directory = Path(directory)
    dataset = SpeakerDataset.from_directory(
        directory=directory,
        template=template,
        glob_pattern=f'*.{ftype}'
    )
    model = ConvSpeakerNet(n_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(n_epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # save the model
    if model_path is None:
        model_path = directory / 'model.pth'
    else:
        model_path = Path(model_path)
    torch.save(model.state_dict(),  model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a speaker classification model.')
    parser.add_argument('directory', type=str, help='Directory containing audio files.')
    parser.add_argument('--template', type=str, default='{}', help='Template for file names.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save the trained model.')
    parser.add_argument('--ftype', type=str, default='wav', help='File type of audio files.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda).')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs.')

    args = parser.parse_args()
    train(args.directory, args.template, args.model_path, args.ftype, args.device, args.n_epochs)