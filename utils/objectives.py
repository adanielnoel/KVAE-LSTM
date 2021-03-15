import torch
import torch.nn.functional as F

from utils.utils import kl_divergence


def neg_elbo_imgs(model, x):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    # px_z : [batch_size, channels, height, width]
    # qz_x : [batch_size, latent_dim]
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:-3], -1) * model.llik_scaling
    # lpx_z : [batch_size, channels * height * width]
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    # kld : [batch_size, latent_dim]
    # print(f'log_px = {-(lpx_z.sum(-1)).mean(0).sum().item():.3f}   kl = {(kld.sum(-1)).mean(0).sum().item():.3f}')
    return -(lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()


def mse_rnn(model, x):
    y = model(x[:, :-1])
    return F.mse_loss(y, x[:, 1:])

