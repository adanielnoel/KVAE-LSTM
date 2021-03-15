# KVAE-LSTM
An implementation of a variational auto-encoder with latent state predictions of non-linear dynamics

The results are relatable to the paper [A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning](https://arxiv.org/abs/1710.05741) (see repo https://github.com/simonkamronn/kvae).
The main difference is that here an LSTM is used instead of a Kalman Filter.
Moreover, the paper discuses other functionality such as data imputation. This repo instead focuses only on predicting.

## How to run

Before running `main.py` make sure to generate the training data by running either of `kvae_data/box.py`, `kvae_data/box_gravity.py`, `kvae_data/polygon.py`.

## Dependencies
- `pytorch`
- `pygame`
- `pymunk`
- `matplotlib`
- `numpy`
- `tqdm`
