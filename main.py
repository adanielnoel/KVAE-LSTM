import datetime
import sys
import os
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from utils.data_loaders import ImageSequencesDataset, LatentSequencesDataset
from kvae_data.kvae_data_loader import NPZ_ImageSequencesLoader
import models
from utils.train_regressor import train_regressor
import utils.objectives as objectives
from utils.utils import Logger
import utils.plots as plots

# EXPERIMENT CONFIGURATION

parser = argparse.ArgumentParser(description='KVAE training with latent LSTM')
# General arguments
parser.add_argument('--experiment_dir', type=str, default=None)
parser.add_argument('--spatial_data_filepath', type=str, default=None)
parser.add_argument('--load_previous', type=bool, default=True)
parser.add_argument('--set_device', type=bool, default=True)
# VAE settings
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--batch_size_vae', type=int, default=16)
parser.add_argument('--epochs_vae', type=int, default=30)
# LSTM settings
parser.add_argument('--dynamic_latent_dim', type=int, default=20)
parser.add_argument('--batch_size_lstm', type=int, default=16)
parser.add_argument('--epochs_lstm', type=int, default=30)

args = parser.parse_args()

experiment_dir = Path(args.experiment_dir or './experiments/spatial_kvae')
spatial_data_filepath = Path(args.spatial_data_filepath or './kvae_data/data/box.npz')
temporal_data_filepath = experiment_dir.joinpath('latent_dataset.pk')
config_spatial = dict(
    model_name='VAE',
    latent_dim=args.latent_dim,
    learning_rate=0.001,
    batch_size=args.batch_size_vae,
    epochs=args.epochs_vae
)
config_temporal = dict(
    model_name='LSTM',
    dynamic_latent_dim=args.dynamic_latent_dim,
    learning_rate=0.001,
    batch_size=args.batch_size_lstm,
    epochs=args.epochs_lstm
)

experiment_dir.mkdir(parents=True, exist_ok=True)  # Create experiment dir if it doesn't exist yet
experiment_dir.joinpath('logs').mkdir(parents=True, exist_ok=True)  # Create logs dir if it doesn't exist yet
runId = datetime.datetime.now().isoformat()
sys.stdout = Logger(f'{experiment_dir}/logs/run_{runId}.txt')  # Redirect print to log file as well as the terminal

#   __      __     ______
#   \ \    / /\   |  ____|
#    \ \  / /  \  | |__
#     \ \/ / /\ \ |  __|
#      \  / ____ \| |____
#       \/_/    \_\______|

# SETUP VAE

spatial_model = models.Image32x32_VAE(latent_dim=config_spatial['latent_dim'], learn_prior=False)
kvae_dataset = ImageSequencesDataset(
    NPZ_ImageSequencesLoader(spatial_data_filepath.absolute()).get_as_tensor(),
    return_sequences=False
)


# EPOCH CALLBACKS VAE

def plot_reconstructions(epoch, model, test_subset, results_dir, **kwargs):
    reconstructions = model.reconstruct(test_subset[:5])
    plots.torch_imgs_side_by_side(test_subset[:5], reconstructions, os.path.join(results_dir, f'reconstruction_{epoch}.png'))
    plt.close()  # Make sure matplotlib does not keep the figure in memory


def plot_generations(epoch, model, results_dir, **kwargs):
    generations = model.generate(20)
    plots.torch_img_grid(generations, os.path.join(results_dir, f'generations_{epoch}.png'))
    plt.close()  # Make sure matplotlib does not keep the figure in memory


# RUN TRAINING VAE

train_regressor(model=spatial_model,
                objective=objectives.neg_elbo_imgs,
                config=config_spatial,
                dataset=kvae_dataset,
                results_dir=experiment_dir,
                epoch_callbacks=[plot_reconstructions, plot_generations],
                set_device=args.set_device,
                load_previous=args.load_previous,
                save_history_length=3)

#    _       _____ _______ __  __
#   | |     / ____|__   __|  \/  |
#   | |    | (___    | |  | \  / |
#   | |     \___ \   | |  | |\/| |
#   | |____ ____) |  | |  | |  | |
#   |______|_____/   |_|  |_|  |_|

# SETUP LSTM

kvae_dataset.return_sequences = True
if args.load_previous and os.path.exists(temporal_data_filepath):
    latent_dataset = LatentSequencesDataset().load(temporal_data_filepath)
else:
    latent_dataset = kvae_dataset.make_latent_dataset(spatial_model)
    latent_dataset.save(temporal_data_filepath)
normalizer = latent_dataset.get_normalizer()

temporal_model = models.Dyn_LSTM(latent_dim=config_spatial['latent_dim'], dynamic_latent_dim=config_temporal['dynamic_latent_dim'], dyn_normalizer=normalizer)


# EPOCH CALLBACKS LSTM

def plot_predictions(epoch, model, results_dir, **kwargs):
    dvae_model = models.DVAE(spatial_model, model)
    predictions = dvae_model.predict(kvae_dataset[:5], 20)
    plots.torch_img_sequence_compare(kvae_dataset[:5][:, -7:], predictions, os.path.join(results_dir, f'predictions_{epoch}.png'))
    plt.close()  # Make sure matplotlib does not keep the figure in memory


# RUN TRAINING LSTM

train_regressor(model=temporal_model,
                objective=objectives.mse_rnn,
                config=config_temporal,
                dataset=latent_dataset,
                results_dir=experiment_dir,
                epoch_callbacks=[plot_predictions],
                set_device=args.set_device,
                load_previous=args.load_previous,
                save_history_length=3)
