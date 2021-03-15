from typing import Union, Callable, List, Dict, Sized
from pathlib import Path
from collections import defaultdict
import os
import sys
import json

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
import tqdm


def train_epoch(model: nn.Module,
                optimizer: Optimizer,
                objective: Callable,
                train_subset: Union[Dataset, Subset],
                statistics: Dict,
                config: Dict,
                device: str,
                **kwargs):
    """
    Runs a training epoch, updating the model parameters after each batch optimization
    :param model: The model to optimize
    :param optimizer: The optimizer object (e.g., an instance of `torch.optim.Adam`)
    :param objective: The objective function. Must accept as parameters the `model`, a data sample, and the `config` dictionary.
    :param train_subset: An instance of `DataLoader` that delivers batched data compatible with the inputs to the `model`
    :param statistics: The dictionary of training statistics
    :param config: The config dictionary, here used to extract the `batch_size`
    :param device: The device to send the data to, usually same as the model for efficiency.
    :return: The average loss per datapoint (not per batch)
    """
    model.train()
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"] if "batch_size" in config.keys() else 1), shuffle=True)
    b_loss = 0
    n_datapoints = 0
    batches = tqdm.tqdm(train_loader, unit='batch', file=sys.stdout, ascii=False)
    batches.set_description('\t\tTraining')
    for i, batch in enumerate(batches):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = objective(model, batch)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        n_datapoints += len(batch)
        batches.set_postfix_str(f'loss={loss.item() / len(batch)}' if i < len(batches) - 1 else f'ave_loss={b_loss / n_datapoints}')
    avg_loss = b_loss / n_datapoints
    statistics['train_losses'].append(avg_loss)
    return avg_loss


def test_mean_loss(model: nn.Module,
                   objective: Callable,
                   test_subset: Union[Dataset, Subset],
                   statistics: Dict,
                   config: Dict,
                   device: str,
                   **kwargs):
    """
    Computes the average loss on a test set
    :param model: A (partially) optimized model
    :param objective: The objective function. Must accept as parameters the `model`, a data sample, and the `config` dictionary.
    :param test_subset: An instance of `DataLoader` that delivers batched data compatible with the inputs to the `model`
    :param statistics: The dictionary of training statistics
    :param config: The config dictionary, here used to extract the `batch_size`
    :param device: The device to send the data to, usually same as the model for efficiency.
    :return: The average loss per datapoint (not per batch)
    """
    model.eval()
    test_loader = DataLoader(test_subset, batch_size=int(config["batch_size"]), shuffle=True)
    b_loss = 0
    n_datapoints = 0
    batches = tqdm.tqdm(test_loader, unit='batch', file=sys.stdout, ascii=False)
    batches.set_description('\t\tTesting ')
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = batch.to(device)
            loss = objective(model, batch)
            b_loss += loss.item()
            n_datapoints += len(batch)
            batches.set_postfix_str(f'loss={loss.item() / len(batch)}' if i < len(batches) - 1 else f'ave_loss={b_loss / n_datapoints}')
    avg_loss = b_loss / n_datapoints
    statistics['test_loss'].append(avg_loss)
    return avg_loss


class ModelSaver:
    """
    Saves a snapshot of the model and optimizer parameters, as well as training statistics.
    It can also manage a rolling history of the last n saves, useful for early stopping.
    If passed `None` as results_dir it becomes a dummy object that does not save anything.
    """

    def __init__(self, results_dir, history_length=1, config=None):
        assert history_length >= 1
        if results_dir is not None:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
        else:
            self.results_dir = ''
        self.model_name = f"{config['model_name']}_" if config and 'model_name' in config.keys() else ''
        self.history_length = history_length

    def previous_saves(self):
        """Returns an ordered list of previous saves as PosixPath (newest is last)"""
        if os.path.exists(self.results_dir):
            return sorted([x for x in Path(self.results_dir).glob(f'{self.model_name}checkpoint_*.pk')], key=lambda s: int(s.name.replace(f'{self.model_name}checkpoint_', '').replace('.pk', '')))
        else:
            return []

    def previous_save_available(self):
        return len(self.previous_saves()) > 0

    def load_latest_save(self, device=None):
        """Returns model state and optimizer state. Check `previous_save_available` first!"""
        return torch.load(str(self.previous_saves()[-1].absolute()), map_location=device)

    def load_stats(self, fallback=None):
        """Tries to load stats from disk. Returns fallback if loading fails, else returns saved statistics"""
        stats_filepath = os.path.join(self.results_dir, f"{self.model_name}statistics.json")
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                stats = json.load(f)
        else:
            stats = fallback
        return stats

    def cleanup_old_saves(self, keep_history_length=True):
        previous_saves = self.previous_saves()
        to_delete = previous_saves[:-self.history_length] if keep_history_length else previous_saves
        for f in to_delete:
            f.unlink()  # deletes the file

    def clean_directory(self, exclude_logs_subdir=True):
        if os.path.exists(self.results_dir):
            for item in Path(self.results_dir).iterdir():
                if exclude_logs_subdir and item.is_dir() and item.absolute().name == 'logs':
                    continue
                else:
                    item.unlink()

    def __call__(self, epoch, model, optimizer, statistics, **kwargs):
        if os.path.exists(self.results_dir):
            filepath = os.path.join(self.results_dir, f'{self.model_name}checkpoint_{epoch}.pk')
            torch.save((model.state_dict(), optimizer.state_dict()), filepath)
            with open(os.path.join(self.results_dir, f"{self.model_name}statistics.json"), 'w') as f:
                json.dump(statistics, f, indent=4)
            self.cleanup_old_saves()


def train_regressor(model: nn.Module,
                    objective: Callable,
                    config: Dict,
                    dataset: Union[Sized, Dataset],
                    results_dir: Union[str, Path] = None,
                    epoch_callbacks: Union[Callable, List[Callable]] = None,
                    set_device=True,
                    load_previous=False,
                    save_history_length=1):
    """
    Canonical regressor neural network training routine.

    Uses the Adam optimizer with a default learning rate of 0.001 (can be provided in `config`)
    Uses a train-test split ratio of 0.8 by default (can be provided in `config`)
    If `results_dir` is provided, model parameters and optimizer parameters are saved at every epoch, together with statistics like losses.
    If `results_dir` is provided and there is a previous save, and `load_previous` is True, it will resume from the previous save.

    :param model: The model to optimize
    :param objective: The objective function. Must accept as parameters the `model`, a data sample, and the `config` dictionary.
    :param config: A dictionary containing meta-parameters
    :param dataset: The dataset to train on. Must be a torch `Dataset` subclass.
    :param results_dir: Directory where to save/load checkpoints, including `model` parameters, `optimizer` parameters, `statistics`, and results from additional tests.
    :param epoch_callbacks: A callable or list of callables, which can accept any of `epoch`, `model`, `optimizer`, `objective`, `statistics`, `test_subset`, `train_subset`, `results_dir`, `model_saver`
    :param set_device: whether to automatically detect if CUDA is enabled and use it, else use CPU. Default True.
    :param load_previous: whether to load fro a previous save or start over, it which case it also deletes the previous saves. For debugging it's better to set this to false. Default False.
    :param save_history_length: how many of the latest checkpoints to keep in the disk.
    :return: A dictionary of lists of train losses, test losses, and epochs.

    .. warnings:: When loading a previous save, parameters are loaded onto the device where they were originally. Loading on a different machine is likely to fail if set_device is False.
    """
    msg = f'Setting up training for "{config["model_name"]}"'
    print('\n' + ''.ljust(len(msg) + 2, '-'))
    print(' ' + msg)

    # ==== Setup defaults ====
    config['learning_rate'] = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
    config['train_split_ratio'] = config['train_split_ratio'] if 'train_split_ratio' in config.keys() else 0.8
    total_epochs = config["epochs"]
    config["batch_size"] = config["batch_size"] if "batch_size" in config.keys() else 1
    model_saver = ModelSaver(results_dir, save_history_length, config)
    default_epoch_callbacks = [train_epoch, test_mean_loss, model_saver]

    epoch_callbacks = epoch_callbacks or []  # Replaces possible None by []
    epoch_callbacks = [epoch_callbacks] if callable(epoch_callbacks) else epoch_callbacks  # Puts possible single callable in a list
    epoch_callbacks = default_epoch_callbacks + epoch_callbacks  # Prepend default callbacks

    # ==== Configure device ====
    device = 'cpu'
    if set_device and torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    print(f'\tDevice: {device}')

    # ==== Configure optimizer ====
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], amsgrad=True)

    # ==== Configure statistics ====
    statistics = defaultdict(lambda: [])

    # ==== Load previous save ====
    if load_previous and model_saver.previous_save_available():
        statistics.update(model_saver.load_stats(fallback=statistics))
        model_state, optimizer_state = model_saver.load_latest_save(device=device if set_device else None)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        print(f'\tLoaded checkpoint from "{model_saver.previous_saves()[-1]}"')
    elif not load_previous and model_saver.previous_save_available():
        model_saver.clean_directory(exclude_logs_subdir=True)
        print(f'\tCleaned previous runs in "{model_saver.results_dir}"')
    else:
        print(f'\tSave results to "{model_saver.results_dir}"')

    # ==== Prepare data ====
    train_length = int(config['train_split_ratio'] * len(dataset))
    train_subset, test_subset = torch.utils.data.random_split(dataset, [train_length, len(dataset) - train_length])

    batch_size = config["batch_size"]
    print(f'\tBatch size: {batch_size}')
    mem_size_dataset = dataset.memory_size() / 1e6 if hasattr(dataset, 'memory_size') else None
    mem_size_batch = mem_size_dataset / len(dataset) * batch_size if mem_size_dataset is not None else None
    print(f'\tDataset size: {len(dataset):,} examples {f"(~{mem_size_dataset:.1f}Mb, ~{mem_size_batch:.2f}Mb/batch)" if hasattr(dataset, "memory_size") else ""}')
    print(f'\tModel size: {sum([p.numel() for p in model.parameters()]):,} parameters (~{sum([p.numel() * p.element_size() for p in model.parameters()]) / 1e6:.2f}Mb)')
    print(''.ljust(len(msg) + 2, '='))
    # ==== Run training ====
    epoch = statistics['epochs'][-1] if len(statistics['epochs']) > 0 else 0
    while epoch < total_epochs:
        # Advance epoch
        epoch += 1  # 1st epoch is nr 1, not 0
        statistics['epochs'].append(epoch)
        print(f'\tEpoch {epoch}/{total_epochs}')

        # Run epoch routines
        for t in epoch_callbacks:
            t(**dict(epoch=epoch,
                     model=model,
                     optimizer=optimizer,
                     objective=objective,
                     statistics=statistics,
                     config=config,
                     test_subset=test_subset,
                     train_subset=train_subset,
                     results_dir=results_dir,
                     model_saver=model_saver,
                     device=device))
    print('\tTraining done')
    print(''.ljust(len(msg) + 2, '='))
    return statistics
