import torch
from tqdm import tqdm
import sys
from pathlib import Path
import dill
from torch.utils.data.dataset import Dataset
from utils.utils import Normalizer


class LatentSequencesDataset(Dataset):
    def __init__(self, latent_sequences=None):
        self.latent_sequences = latent_sequences

    def __len__(self):
        return self.latent_sequences.shape[0]

    def __getitem__(self, item):
        return self.latent_sequences[item]

    def mean(self):
        return torch.mean(self.latent_sequences, (0, 1))

    def std(self):
        return torch.std(self.latent_sequences, (0, 1))

    def normalize(self):
        self.latent_sequences = (self.latent_sequences - self.mean()) / self.std()

    def get_normalizer(self):
        return Normalizer(self.mean(), self.std())

    def memory_size(self):
        return self.latent_sequences.numel() * self.latent_sequences.element_size()

    def save(self, filepath):
        filepath = Path(filepath)  # Make sure it's a Path object
        filepath.parent.mkdir(parents=True, exist_ok=True)  # Create parent dirs if they don't exist
        torch.save(self.latent_sequences, str(filepath), pickle_module=dill)

    def load(self, filepath):
        filepath = Path(filepath)  # Make sure it's a Path object
        self.latent_sequences = torch.load(filepath, pickle_module=dill)
        return self


class ImageSequencesDataset(Dataset):
    def __init__(self, image_sequences: torch.Tensor, return_sequences=True):
        """
        :param image_sequences: shape [total_sequences, frames_per_sequence, channels, height, width]
        :param return_sequences: whether to return sequences or frames
        """
        self.image_sequences: torch.Tensor = image_sequences
        self.return_sequences = return_sequences

    def memory_size(self):
        return self.image_sequences.numel() * self.image_sequences.element_size()

    def __len__(self):
        if self.return_sequences:
            return self.image_sequences.shape[0]
        else:
            return self.image_sequences.shape[0] * self.image_sequences.shape[1]

    def __getitem__(self, item):
        if self.return_sequences:
            return self.image_sequences[item]
        else:
            # Return as if sequences and sequence dimensions were merged
            return self.image_sequences.view((-1, *self.image_sequences.shape[-3:]))[item]

    def make_latent_dataset(self, vae):
        latents = torch.zeros((*self.image_sequences.shape[:2], vae.latent_dim))
        for i in tqdm(range(len(self)), desc='Generating latents', unit='sequences', file=sys.stdout):
            latents[i] = vae.infer(self.image_sequences[i])
        return LatentSequencesDataset(latents)


class StateActionSequencesDataset(Dataset):
    def __init__(self, y_dataset: torch.Tensor, a_dataset: torch.Tensor, return_sequences=True, return_collated=True):
        self.y_dataset = y_dataset
        self.a_dataset = a_dataset
        self.return_sequences = return_sequences
        self.return_collated = return_collated

    def mean(self):
        if self.return_collated:
            return torch.cat((self.y_dataset.view(-1, 4).mean(0), self.a_dataset.view(-1, 2).mean(0)), dim=0)
        else:
            return self.y_dataset.view(-1, 4).mean(0), self.a_dataset.view(-1, 2).mean(0)

    def std(self):
        if self.return_collated:
            return torch.cat((self.y_dataset.view(-1, 4).std(0), self.a_dataset.view(-1, 2).std(0)), dim=0)
        else:
            return self.y_dataset.view(-1, 4).std(0), self.a_dataset.view(-1, 2).std(0)

    def memory_size(self):
        return self.y_dataset.numel() * self.y_dataset.element_size() + self.a_dataset.numel() * self.a_dataset.element_size()

    def normalize(self):
        return_collated = self.return_collated
        self.return_collated = False
        means = self.mean()
        stds = self.std()
        self.return_collated = return_collated
        self.y_dataset = (self.y_dataset - means[0]) / stds[0]
        self.a_dataset = (self.a_dataset - means[1]) / stds[1]

    def __len__(self):
        if self.return_sequences:
            return self.y_dataset.shape[0]
        else:
            return self.y_dataset.shape[0] * self.y_dataset.shape[1]

    def __getitem__(self, item):
        if self.return_sequences and self.return_collated:
            return torch.cat((self.y_dataset[item], self.a_dataset[item]), dim=-1)
        elif self.return_sequences and not self.return_collated:
            return self.y_dataset[item], self.a_dataset[item]
        elif not self.return_sequences and self.return_collated:
            return torch.cat((self.y_dataset.view((-1, self.y_dataset.shape[-1]))[item],
                              self.a_dataset.view((-1, self.a_dataset.shape[-1]))[item]),
                             dim=-1)
        elif not self.return_sequences and not self.return_collated:
            # Return as if sequences and sequence dimensions were merged
            return (self.y_dataset.view((-1, self.y_dataset.shape[-1]))[item],
                    self.a_dataset.view((-1, self.a_dataset.shape[-1]))[item])

    def make_latent_dataset(self, vae):
        if not self.return_collated:
            raise NotImplementedError("Dataset is set not collate states and actions, but only collated is allowed for making a latent dataset")
        latents = torch.zeros((*self.y_dataset.shape[:2], vae.latent_dim))
        for i in tqdm(range(len(self)), desc='Generating latents', file=sys.stdout):
            latents[i] = vae.infer(torch.cat((self.y_dataset[i], self.a_dataset[i]), dim=-1))
        return LatentSequencesDataset(latents)
