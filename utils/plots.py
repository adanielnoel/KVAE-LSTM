from math import sqrt, ceil

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import numpy as np


#    _____
#   |_   _|
#     | |  _ __ ___   __ _  __ _  ___  ___
#     | | | '_ ` _ \ / _` |/ _` |/ _ \/ __|
#    _| |_| | | | | | (_| | (_| |  __/\__ \
#   |_____|_| |_| |_|\__,_|\__, |\___||___/
#                           __/ |
#                          |___/


def torch_images_to_matplotlib(image: torch.Tensor):
    """
    Prepares a torch tensor for plotting in matplotlib as an image
    :param image: torch.Tensor of shape [channels, height, width] or [batch_size, channels, height, width]
    :return: numpy array of shape [height, width, 3] or [batch_size, height, width, 3]
    """
    np_data = image.detach().cpu().numpy()
    np_data = np.swapaxes(np_data, -3, -1)
    if np_data.shape[-1] == 1:  # If image is grayscale, repeat channel since matplotlib only accepts RGB or RGBA
        np_data = np_data.repeat(3, -1)
    return np_data


def torch_imgs_side_by_side(left: torch.Tensor, right: torch.Tensor, filepath=None):
    """
    Plots images in two columns
    :param left: images tensor of shape [batch_size, channels, height, width] or [channels, height, width]
    :param right: images tensor of shape [batch_size, channels, height, width] or [channels, height, width]
    :param filepath: where to save the image as a PNG. Default None.
    :return: the figure object. Remember to `plt.close()` after using the figure to delete it from memory.
    """
    if left.ndim == 3:
        left = left.unsqueeze(0)
        right = right.unsqueeze(0)
    left = torch_images_to_matplotlib(left)
    right = torch_images_to_matplotlib(right)
    rows = left.shape[0]
    fig = plt.figure(figsize=(8, 4 * rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, 2), axes_pad=0.1)
    for i, row_axes in enumerate(grid.axes_row):
        row_axes[0].imshow(left[i])
        row_axes[1].imshow(right[i])
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    return fig


def torch_img_grid(images: torch.Tensor, filepath=None):
    """
    Plots a grid of images
    :param images: tensor of shape [rows, cols, channels, height, width] or [batch_size, channels, height, width]
    :param filepath: where to save the image as a PNG. Default None.
    :return: the figure object. Remember to `plt.close()` after using the figure to delete it from memory.
    """
    if images.ndim == 5:  # [rows, cols, channels, height, width]
        rows = images.shape[0]
        cols = images.shape[1]
        n_images = cols * rows
    else:  # [images, channels, height, width]
        n_images = images.shape[0]
        rows = int(sqrt(n_images))
        cols = ceil(n_images / rows)
    images = torch_images_to_matplotlib(images.reshape((-1, *images.shape[-3:])))  # [images, height, width, channels]
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1)
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            if k < n_images:
                grid.axes_row[i][j].imshow(images[k])
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    return fig


def torch_img_sequence_compare(sequences1: torch.Tensor, sequences2: torch.Tensor, filepath=None, stacked=False):
    """
    Plots a grid of image comparisons. The comparisons are two horizontal sequences of images either side-by-side or stacked
    :param sequences1: tensor of shape [sequences, images, channels, height, width]
    :param sequences2: tensor of shape [sequences, images, channels, height, width]
    :param filepath: where to save the image as a PNG. Default None.
    :param stacked: whether to stack the comparisons (True) of plot them side-by-side (False). Default False.
    :return: the figure object. Remember to `plt.close()` after using the figure to delete it from memory.
    """
    if stacked:
        assert sequences1.shape == sequences2.shape
    else:
        assert sequences1.shape[0] == sequences2.shape[0]
    # sequences shape: [sequences, images, channels, height, width]
    sequences1 = [torch_images_to_matplotlib(seq) for seq in sequences1]
    sequences2 = [torch_images_to_matplotlib(seq) for seq in sequences2]
    cols = sequences1[0].shape[0] if stacked else (sequences1[0].shape[0] + sequences2[0].shape[0])
    rows = 2 if stacked else 1
    fig = plt.figure(figsize=(cols * 4, len(sequences1) * rows * 4))
    for i in range(len(sequences1)):
        grid = ImageGrid(fig, int(f'{len(sequences1)}{1}{i}'), nrows_ncols=(rows, cols), axes_pad=0.1)
        if stacked:
            for j in range(sequences1[i].shape[0]):
                grid.axes_row[0][j].imshow(sequences1[i][j])
                grid.axes_row[1][j].imshow(sequences2[i][j])
        else:
            for j in range(sequences1[i].shape[0]):
                grid.axes_row[0][j].imshow(sequences1[i][j])
            for j in range(sequences2[i].shape[0]):
                grid.axes_row[0][j + sequences1[0].shape[0]].imshow(sequences2[i][j])
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    return fig


#     _____ _                   _
#    / ____(_)                 | |
#   | (___  _  __ _ _ __   __ _| |___
#    \___ \| |/ _` | '_ \ / _` | / __|
#    ____) | | (_| | | | | (_| | \__ \
#   |_____/|_|\__, |_| |_|\__,_|_|___/
#              __/ |
#             |___/


def torch_var_sequences_compare(sequences1: torch.Tensor, sequences2: torch.Tensor, filepath=None, labels=None):
    rows = sequences1.shape[0]
    fig = plt.figure(figsize=(10, rows * 2))
    x_axis = torch.arange(sequences1.shape[1])
    for row in range(rows):
        ax = fig.add_subplot(int(f'{rows}1{row + 1}'))
        for j in range(sequences1.shape[2]):
            ln = ax.plot(x_axis, sequences1[row, :, j], linestyle='-', label=labels[j] if labels else '')
            ax.plot(x_axis, sequences2[row, :, j], color=ln[0].get_color(), linestyle='--')
        if labels:
            ax.legend()
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    return fig


def plot_image_sequence_predictions(dvae, data: torch.Tensor, steps: int, filepath=None):
    predictions = dvae.predict(data, steps)
    return torch_img_sequence_compare(data[:, -7:], predictions, filepath)


def plot_image_reconstructions(vae, data, filepath=None):
    reconstruction = vae.reconstruct(data)
    return torch_img_grid(reconstruction, filepath)


def plot_state_action_reconstructions(vae, data, filepath=None, labels=None):
    reconstruction = vae.reconstruct(data)
    return torch_var_sequences_compare(data, reconstruction, filepath, labels)


if __name__ == '__main__':
    import torch

    # torch_img_sequence_compare(torch.rand((3, 5, 1, 32, 32)), torch.rand((3, 5, 1, 32, 32)), 'test.png', stacked=True)
    torch_imgs_side_by_side(torch.rand((3, 1, 32, 32)), torch.rand((3, 1, 32, 32)), 'test.png')
    # torch_var_sequences_compare(torch.rand((3, 30, 6)), torch.rand((3, 30, 6)), filepath='../test.png', labels=list('abcdef'))
