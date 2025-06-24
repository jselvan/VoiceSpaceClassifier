import torch
import torch.nn.functional as F

def get_similarity_matrix(model, all_inputs):
    # get embeddings: (48, D)
    embeddings = model.extract_features(all_inputs)

    # cosine similarity: (48, 48)
    sim_matrix = F.cosine_similarity(embeddings[:, None, :], embeddings[None, :, :], dim=-1)

    # to get a 16x16 matrix (speaker-level), average embeddings for each speaker first
    speaker_embeddings = torch.stack([
        embeddings[i*3:(i+1)*3].mean(dim=0) for i in range(16)
    ])
    speaker_sim = F.cosine_similarity(
        speaker_embeddings[:, None, :],
        speaker_embeddings[None, :, :],
        dim=-1
    )
    return sim_matrix, speaker_sim

if __name__ == '__main__':
    import argparse
    from VoiceSpaceClassifier.model import ConvSpeakerNet
    from VoiceSpaceClassifier.dataset import SpeakerDataset
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Get similarity matrix from a trained model.')
    parser.add_argument('model_path', type=str, help='Path to the trained model.')
    parser.add_argument('directory', type=str, help='Directory containing audio files.')
    parser.add_argument('--template', type=str, default='{}', help='Template for file names.')
    parser.add_argument('--ftype', type=str, default='wav', help='File type of audio files.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference (cpu or cuda).')
    args = parser.parse_args()
    model_path = Path(args.model_path)
    directory = Path(args.directory)
    dataset = SpeakerDataset.from_directory(
        directory=directory,
        template=args.template,
        glob_pattern=f'*.{args.ftype}'
    )
    model = ConvSpeakerNet.from_pretrained(model_path, n_classes=16, device=args.device)
    sim_matrix, speaker_sim = get_similarity_matrix(model, dataset[:].to(args.device))  # get all inputs as a batch
    # convert to numpy and plot
    import numpy as np
    import matplotlib.pyplot as plt
    sim_matrix_np = sim_matrix.cpu().numpy()
    speaker_sim_np = speaker_sim.cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(sim_matrix_np, cmap='hot', interpolation='nearest')
    plt.title('Cosine Similarity Matrix (All Inputs)')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(speaker_sim_np, cmap='hot', interpolation='nearest')
    plt.title('Cosine Similarity Matrix (Speakers)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # save the matrices
    np.savez_compressed(directory / 'similarity_matrices.npz', 
                        sim_matrix=sim_matrix_np, 
                        speaker_sim=speaker_sim_np)
    print(f'Similarity matrices saved to {directory / "similarity_matrices.npz"}')
    print('Done.')
