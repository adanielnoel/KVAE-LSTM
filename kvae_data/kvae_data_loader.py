import numpy as np
import torch

"""
Code borrowed from https://github.com/simonkamronn/kvae
"""


class NPZ_ImageSequencesLoader(object):
    """
    Load sequences of images
    """

    def __init__(self, file_path):
        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)

        # Get data dimensions
        self.sequences, self.timesteps, self.d1, self.d2 = self.images.shape

    def get_as_tensor(self, add_channel_dim=True):
        images = torch.tensor(self.images)
        if add_channel_dim:
            images = torch.unsqueeze(images, 2)
        return images

    def shuffle(self):
        permutation = np.random.permutation(self.sequences)
        self.images = self.images[permutation]
