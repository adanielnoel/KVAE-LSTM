import math
import sys

import torch
import torch.distributions

import numpy as np
import random


class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


# Adapted from https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    """
    File object to redirect a copy of the print commands to a log file.
    """
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(str(filename), mode)
        self.prev_line = ''

    def write(self, message):
        self.terminal.write(message)
        # The following logic handles carriage return to write over the previous line, so that progress bars do not span a line per update.
        if message[:1] != '\r' and len(message) > 0:
            self.log.write(self.prev_line)
            self.prev_line = message
        elif message[-1:] == '\n':
            self.log.write(message)
            self.prev_line = ''
        else:
            self.prev_line = message

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def __del__(self):
        self.log.write(self.prev_line)


# Borrowed from https://github.com/iffsid/mmvae
def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


# Adapted from https://github.com/iffsid/mmvae
def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    try:
        return torch.distributions.kl_divergence(d1, d2)
    except NotImplementedError:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def gen_func(length, bins, p, smoothing_window=None):
    """
    Generates a random function with regularly-spaced x and a Markov process for y, with optional smoothing.
    :param length: The length in x samples
    :param bins: a list of values to choose y from
    :param p: the probability of changing the current value
    :param smoothing_window: The length of the smoothing window (rolling average)
    :return: a numpy array of y samples
    """
    data = np.zeros(length)
    current_bin = random.choice(bins)
    for i in range(length):
        if random.random() < p:
            current_bin = random.choice(bins)
        data[i] = current_bin

    if smoothing_window:
        cumsum = np.cumsum(np.append(data, np.repeat(data[-1], smoothing_window)))
        data = (cumsum[smoothing_window:] - cumsum[:-smoothing_window]) / smoothing_window

    return data


class Normalizer:
    """
    Utility class to normalize and denormalize data.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def apply(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

