import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional
import torch.utils.data

from utils.utils import Constants
from utils.plots import torch_img_grid, torch_imgs_side_by_side
from .vae import VAE

"""
Code adapted from https://github.com/iffsid/mmvae/blob/public/src/models/vae_cub_image.py
"""

# Constants
fBase = 64


class Enc(nn.Module):
    """ Generate latent parameters for CUB image data. """

    def __init__(self, latent_dim):
        super(Enc, self).__init__()
        modules = [
            # size: 1 x 32 x 32
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: fBase x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.Dropout(0.1),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.Dropout(0.08),
            nn.ReLU(True)
            # size: (fBase * 4) x 4 x 4
        ]
        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x: torch.Tensor):
        e = self.enc(x)
        return self.c1(e).squeeze(), nn.functional.softplus(self.c2(e)).squeeze() + Constants.eta


class Dec(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Dec, self).__init__()
        modules = [
            # size: latentDim x 1 x 1
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 2, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.Dropout(0.1),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.Dropout(0.08),
            nn.ReLU(True),
            # size: fBase x 16 x 16
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # size: 1 x 32 x 32
        ]
        self.dec = nn.Sequential(*modules)
        self.latent_dim = latent_dim

    def forward(self, z):
        out = self.dec(z.view(-1, self.latent_dim, 1, 1))
        return out, torch.tensor(0.01).to(z.device)


class Image32x32_VAE(VAE):
    def __init__(self, latent_dim, learn_prior=False, **kwargs):
        super(Image32x32_VAE, self).__init__(
            prior_dist=dist.Normal,
            likelihood_dist=dist.Normal,
            post_dist=dist.Normal,
            enc=Enc(latent_dim),
            dec=Dec(latent_dim),
            latent_dim=latent_dim
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, latent_dim), requires_grad=learn_prior)  # logvar
        ])
        self.modelName = 'Image 32 x 32 VAE'
        self.data_shape = [1, 32, 32]
        self.llik_scaling = latent_dim / (32. * 32.)  # Scale factor for the log-likelihood in the loss function

    @property
    def pz_params(self):
        return self._pz_params[0], nn.functional.softplus(self._pz_params[1]) + Constants.eta

    def generate(self, N, K=None, filepath=None):
        data = super().generate(N, K)
        if filepath:
            torch_img_grid(data, filepath)
        return data

    def compare_input_output(self, data, filepath=None):
        reconstruction = self.reconstruct(data)
        return torch_imgs_side_by_side(data, reconstruction, filepath)
