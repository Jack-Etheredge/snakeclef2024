"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import datetime
import random
import shutil
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import matplotlib
from torch import nn as nn

matplotlib.style.use('ggplot')
OUTPUT_DIR = Path('__file__').parent.absolute() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_model(epochs, model, optimizer, criterion, pretrained, model_path):
    """
    Function to save the trained model to disk.
    """

    if not model_path:
        model_path = str(OUTPUT_DIR / f"model_pretrained_{pretrained}.pth")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, model_path)


def checkpoint_model(epochs, model, optimizer, criterion, validation_loss, file_path, add_val_flag=False):
    """
    Function to save the trained model to disk.
    """
    data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'validation_loss': validation_loss,
    }
    if add_val_flag:
        data["val_fine_tuned"] = True
    torch.save(data, file_path)


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, accuracy_plot_path, loss_plot_path):
    """
    Function to save the loss and accuracy plots to disk.
    """

    if not accuracy_plot_path:
        accuracy_plot_path = str(OUTPUT_DIR / f"accuracy_pretrained_{pretrained}.png")

    if not loss_plot_path:
        loss_plot_path = str(OUTPUT_DIR / f"loss_pretrained_{pretrained}.png")

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_path)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_seed(seed=1337):
    print(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)


def copy_config(script_name, experiment_id):
    now_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = Path("model_checkpoints") / experiment_id
    config_from = Path("conf") / "config.yaml"
    config_to = experiment_dir / f"config_{script_name}_{now_dt}.yaml"
    shutil.copyfile(config_from, config_to)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
