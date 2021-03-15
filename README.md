# KVAE-LSTM
An implementation of a variational auto-encoder with latent state predictions of non-linear dynamics

The results are relatable to the paper [A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning](https://arxiv.org/abs/1710.05741) (see repo https://github.com/simonkamronn/kvae).
The main difference is that here an LSTM is used instead of a Kalman Filter.
Moreover, the paper discuses other functionality such as data imputation. This repo instead focuses only on predicting.
