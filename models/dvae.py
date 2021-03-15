import torch
import torch.nn as nn

from .vae import VAE
from utils.utils import Normalizer


class Dyn_LSTM(nn.Module):
    def __init__(self, latent_dim, dynamic_latent_dim, dyn_normalizer: Normalizer = None):
        super(Dyn_LSTM, self).__init__()
        self.nn1 = nn.LSTM(latent_dim, dynamic_latent_dim, num_layers=3, batch_first=True, dropout=0.05)
        self.nn2 = nn.Linear(in_features=dynamic_latent_dim, out_features=latent_dim)
        self.latent_dim = latent_dim
        self.dynamic_latent_dim = dynamic_latent_dim
        self.dyn_normalizer = dyn_normalizer

    def forward(self, x: torch.Tensor):  # expects x to be already normalized
        pred, _ = self.nn1(x)
        pred = self.nn2(pred)
        return pred

    def predict(self, x: torch.Tensor, steps: int):
        # data.shape : [batch_size, sequence_length, state_dim]
        self.eval()
        with torch.no_grad():
            x_ = self.dyn_normalizer.apply(x) if self.dyn_normalizer else x
            next_x = torch.zeros((x.shape[0], steps, self.latent_dim))
            for step in range(steps):
                next_x[:, step] = self(torch.cat((x, next_x[:, :step]), dim=1))[:, -1]
            next_x = self.dyn_normalizer.inverse(next_x) if self.dyn_normalizer else next_x
        return next_x


class DVAE(nn.Module):
    def __init__(self, vae: VAE, dyn: nn.Module):
        super(DVAE, self).__init__()
        self.vae = vae
        self.dyn = dyn
        self.modelName = None
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self.vae.pz_params

    @property
    def latent_dim(self):
        return self.vae.latent_dim

    @property
    def dynamic_latent_dim(self):
        return self.dyn.dynamic_latent_dim

    @property
    def data_shape(self):
        return self.vae.data_shape

    def forward(self, x, K=1):
        # x.shape : [batch_size, sequence_length, *data_shape]
        batch_size, sequence_length = x.shape[:2]

        # === ENCODER ===
        # Merge dimensions batch_size, sequence_length for the encoder
        # since it does not use temporal relations in the data
        self._qz_x_params = self.vae.enc(x.view((-1, *self.data_shape)))
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        # -> [batch_size * sequence_length, latent_dim]

        # === DYNAMICS ===
        # Recover batches and sequences for the dynamics
        # TODO: think how to handle distribution (bayesian LSTM?)
        zs_d = self.dyn(zs.view((batch_size, sequence_length, self.latent_dim)))
        # -> [batch_size, sequence_length, latent_dim]

        # === DECODER ===
        # Merge back again for the decoder
        px_z_params = self.vae.dec(zs_d.view(-1, self.latent_dim))
        # -> mean and variance have shape [batch_size * sequence_length, *data_shape]
        # Recover mini batch and sequences again for the return
        px_z_params = [p.view((batch_size, sequence_length, *self.data_shape)) for p in px_z_params]
        # -> mean and variance have shape [batch_size, sequence_length, *data_shape]
        px_z = self.px_z(*px_z_params)
        # -> samples have shape [batch_size, sequence_length, *data_shape]

        return qz_x, px_z, zs

    def infer(self, data):
        # data.shape : [batch_size, sequence_length, *data_shape]
        self.eval()
        batch_size, sequence_length = data.shape[:2]
        with torch.no_grad():
            d_ = data.reshape((-1, *self.data_shape))
            # -> [batch_size * sequence_length, *data_shape]
            latents = self.vae.infer(d_).reshape(batch_size, sequence_length, self.latent_dim)
            # -> [batch_size, sequence_length, *data_shape]
        return latents

    def decode(self, latents):
        # latents.shape : [batch_size, sequence_length, latent_dim]
        self.eval()
        batch_size, sequence_length = latents.shape[:2]
        with torch.no_grad():
            l_ = latents.reshape((-1, self.latent_dim))
            # -> [batch_size * sequence_length, latent_dim]
            decoded = self.vae.decode(l_).reshape((batch_size, sequence_length, *self.data_shape))
            # -> [batch_size, sequence_length, latent_dim]
        return decoded

    def reconstruct(self, data):
        # data.shape : [batch_size, sequence_length, *data_shape]
        self.eval()
        with torch.no_grad():
            latents = self.infer(data)
            recon = self.decode(latents)
        return recon  # [batch_size, sequence_length, *data_shape]

    # def predict_latent(self, latents: torch.Tensor, steps: int, filepath=None):
    #     # data.shape : [batch_size, sequence_length, latent_dim]
    #     self.eval()
    #     with torch.no_grad():
    #         next_latents = torch.zeros((latents.shape[0], steps, self.latent_dim))
    #         for step in range(steps):
    #             next_latents[:, step] = self.dyn(torch.cat((latents, next_latents[:, :step]), dim=1))[:, -1]
    #     return next_latents

    def predict(self, data: torch.Tensor, steps: int):
        # data.shape : [batch_size, sequence_length, *data_shape]
        self.eval()
        with torch.no_grad():
            latents = self.infer(data)
            predicted_latents = self.dyn.predict(latents, steps)
            predictions = self.decode(predicted_latents)
        return predictions
