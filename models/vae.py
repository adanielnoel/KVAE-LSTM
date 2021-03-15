# Base VAE class definition

import torch
import torch.nn as nn

from utils.utils import get_mean

"""
Code borrowed from https://github.com/iffsid/mmvae
"""


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, latent_dim):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = enc
        self.dec = dec
        self._latent_dim = latent_dim
        self.modelName = None
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0
        self.data_shape = None # defined in subclass

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @property
    def latent_dim(self):
        return self._latent_dim

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K])).squeeze()  # if K > 1 then zs.shape : [K, batch_size, latent_dim] else [batch_size, latent_dim]
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def generate(self, N, K=None):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            px_z = self.px_z(*self.dec(latents))
            data = get_mean(px_z) # take mean instead of K samples, see comment below.
            # data = px_z.sample(torch.Size([K])) # Samples output distribution too, but not much changes in the output
            # torch.clip_(data, min=0.0, max=1.0)
        return data.view(-1, *self.data_shape)

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            latents = self.infer(data)
            recon = self.decode(latents)
        return recon

    def infer(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # no dim expansion
        return latents

    def decode(self, latents):
        self.eval()
        with torch.no_grad():
            px_z = self.px_z(*self.dec(latents))
            recon = get_mean(px_z)
        return recon



