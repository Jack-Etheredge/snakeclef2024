"""
modified from these sources:
https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from focal_loss.focal_loss import FocalLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_lr_finder import LRFinder
import time
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from paths import CHECKPOINT_DIR
from closedset_model import build_model, unfreeze_model, update_dropout_rate
from datasets import get_datasets, get_data_loaders
from evaluate import evaluate_experiment
from utils import copy_config, save_plots, checkpoint_model, get_device
from losses import SeesawLoss, SupConLoss, CompositeLoss


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    alternative implementation
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/12
    returns the model parameters for use with weight_decay
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# alternative implementation
# https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/7
# TODO: revise and try
@torch.no_grad()
def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


# Training function.
def train(model, trainloader, optimizer, criterion, max_norm, device='cpu'):
    model.train()
    print(f'Training with device {device}')
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # gradient norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # this is a tunable hparam
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / i + 1
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
@torch.no_grad()
def validate(model, testloader, criterion, device='cpu'):
    model.eval()
    print(f'Validation with device {device}')
    valid_running_loss = 0.0
    valid_running_correct = 0
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / i + 1
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def create_scheduler_and_optimizer(model, lr, weight_decay, lr_scheduler, lr_scheduler_patience):
    parameters = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = optim.AdamW(parameters, lr=lr)
    # make scheduler after the optimizer has been created/modified
    if lr_scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_scheduler_patience)
    else:
        scheduler = None
        print(f"not using lr scheduler, config set to {lr_scheduler}")
    return optimizer, scheduler


def get_progression_params(cfg, epoch):
    """
    Get the progressive learning parameters for the current epoch.
    :param cfg: hydras config object
    :param epoch: current epoch
    :return: image_size, dropout_rate, batch_size
    """
    # progressive learning params
    start_image_size = cfg["progressive-learning"]["start_image_size"]
    end_image_size = cfg["progressive-learning"]["end_image_size"]
    start_dropout = cfg["progressive-learning"]["start_dropout"]
    end_dropout = cfg["progressive-learning"]["end_dropout"]
    start_batch_size = cfg["progressive-learning"]["start_batch_size"]
    end_batch_size = cfg["progressive-learning"]["end_batch_size"]
    progression_epochs = cfg["progressive-learning"]["progression_epochs"]

    epoch_percent = epoch / (progression_epochs - 1)
    image_size = start_image_size + epoch_percent * (end_image_size - start_image_size)
    dropout_rate = start_dropout + epoch_percent * (end_dropout - start_dropout)
    batch_size = start_batch_size + epoch_percent * (end_batch_size - start_batch_size)

    return int(image_size), dropout_rate, int(batch_size)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model_id = cfg["train"]["model_id"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    pretrained = cfg["train"]["pretrained"]
    early_stop_thresh = cfg["train"]["early_stop_thresh"]
    multiclass_loss_function = cfg["train"]["loss_function"]
    use_venom_loss = cfg["train"]["use_venom_loss"]
    use_logitnorm = cfg["train"]["use_logitnorm"]
    max_norm = cfg["train"]["max_norm"]
    dropout_rate = cfg["train"]["dropout_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    use_lr_finder = cfg["train"]["use_lr_finder"]
    fine_tune_after_n_epochs = cfg["train"]["fine_tune_after_n_epochs"]
    lr_scheduler = cfg["train"]["lr_scheduler"]
    lr_scheduler_patience = cfg["train"]["lr_scheduler_patience"]
    n_classes = cfg["train"]["n_classes"]
    train_progressively = cfg["train"]["train_progressively"]
    use_class_weights_venom_loss = cfg["train"]["use_class_weights_venom_loss"]

    for k, v in cfg["train"].items():
        if v == "None" or v == "null":
            url = "https://stackoverflow.com/questions/76567692/hydra-how-to-express-none-in-config-files"
            raise ValueError(f"`{k}` set to 'None' or 'null'. Use `null` for None values in hydra; see {url}")

    experiment_id = cfg["train"]["experiment_id"]
    if experiment_id is None:
        experiment_id = str(datetime.now()).replace(" ", "-")
        cfg["train"]["experiment_id"] = experiment_id

    # use experiment_id to save model checkpoint, graphs, predictions, performance metrics, etc
    experiment_dir = CHECKPOINT_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = str(experiment_dir / f"model.pth")
    accuracy_plot_path = str(experiment_dir / "accuracy_plot.png")
    loss_plot_path = str(experiment_dir / "loss_plot.png")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(str(experiment_dir / "experiment_config.json"), "w") as f:
        json.dump(config_dict, f)
    copy_config("train", experiment_id)

    # torch.autograd.set_detect_anomaly(True)
    if train_progressively:
        image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch=0)
    else:
        train_loader, val_loader, class_weights = create_train_val_loaders(cfg)

    device = get_device()
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(
        model_id=model_id,
        pretrained=pretrained,
        fine_tune=not fine_tune_after_n_epochs,
        num_classes=n_classes,
        dropout_rate=dropout_rate,
    ).to(device)

    # TODO: refine/replace this logic
    resume_from_checkpoint = model_file_path if Path(model_file_path).exists() else None
    if resume_from_checkpoint:
        print(f"resuming from checkpoint: {model_file_path}")
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, scheduler = create_scheduler_and_optimizer(model, lr, weight_decay, lr_scheduler,
                                                              lr_scheduler_patience)
        # Start epoch counter from the checkpoint epoch and set the validation loss so that the model checkpoint isn't
        #   automatically updated on the first epoch after resuming training.
        # There are edge cases where this behavior might not be desired such as fine-tuning with a different
        #   loss function and/or dataset such that the validation loss increases but still represents model improvement.
        best_validation_loss = checkpoint['validation_loss']
        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        if train_progressively:
            image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch=start_epoch)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for group in optimizer.param_groups:
                group["lr"] = lr
            print("loaded optimizer state but replaced lr")
        except Exception as e:
            print(f"unable to load optimizer state due to {e}")
        model.to(device)
    else:
        best_validation_loss = float("inf")
        best_epoch, start_epoch = 0, 0
        optimizer, scheduler = create_scheduler_and_optimizer(model, lr, weight_decay, lr_scheduler,
                                                              lr_scheduler_patience)
        print(f"training new model: {experiment_id}")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    if multiclass_loss_function == "seesaw":
        multiclass_loss = SeesawLoss(num_classes=n_classes, device=device)
    elif multiclass_loss_function == "focal":
        multiclass_loss = FocalLoss(gamma=2.)
    elif multiclass_loss_function == "supcon":
        multiclass_loss = SupConLoss()
    elif multiclass_loss_function == "crossentropy":
        multiclass_loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {multiclass_loss_function}")
    criterion = CompositeLoss(multiclass_loss=multiclass_loss, use_venom_loss=use_venom_loss,
                              use_logitnorm=use_logitnorm,
                              class_weights=class_weights if use_class_weights_venom_loss else None)

    if use_lr_finder:
        optimizer, lr = update_optimizer_lr_finder(criterion, device, model, optimizer, train_loader)

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    unfrozen = fine_tune_after_n_epochs == 0
    for epoch in range(start_epoch, epochs):

        if train_progressively:
            image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch)
            model = update_dropout_rate(model, dropout_rate)

        if (not unfrozen and (epoch + 1 > fine_tune_after_n_epochs)):
            model = unfreeze_model(model)
            print("all layers unfrozen")
            if use_lr_finder:
                print("finding new best LR after unfreezing weights")
                optimizer, lr = update_optimizer_lr_finder(criterion, device, model, optimizer, train_loader)
            else:
                print("manually stepping down LR after unfreeze since LR finder is incompatible with multi-input model")
                lr = cfg["train"]["lr_after_unfreeze"]
            optimizer, scheduler = create_scheduler_and_optimizer(model, lr, weight_decay, lr_scheduler,
                                                                  lr_scheduler_patience)
            unfrozen = True

        print(f"Epoch {epoch + 1} of {epochs}")
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"current learning rate: {curr_lr:.0e}")
        recreate_loader = True
        while recreate_loader:
            try:
                train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, max_norm, device)
                recreate_loader = False
            except Exception as e:
                print("issue with training")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, val_loader, _ = create_train_val_loaders(cfg)

        recreate_loader = True
        while recreate_loader:
            try:
                valid_epoch_loss, valid_epoch_acc = validate(model, val_loader, criterion, device)
                if scheduler is not None:
                    scheduler.step(valid_epoch_loss)
                recreate_loader = False
            except Exception as e:
                print("issue with validation")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, val_loader, _ = create_train_val_loaders(cfg)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss: .3f}, training acc: {train_epoch_acc: .3f}")
        print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")
        # TODO: make a helper that can use loss or accuracy and minimizes or maximizes intelligently
        if valid_epoch_loss < best_validation_loss:
            best_validation_loss = valid_epoch_loss
            best_epoch = epoch
            print("updating best model")
            checkpoint_model(epoch, model, optimizer, criterion, valid_epoch_loss, model_file_path)
            print(">> successfully updated best model <<")
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch + 1}")
            break  # terminate the training loop
        else:
            print(f"model did not improve from best epoch {best_epoch + 1} with loss {best_validation_loss: .3f}")
        print('-' * 50)
        time.sleep(5)

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, accuracy_plot_path=accuracy_plot_path,
               loss_plot_path=loss_plot_path)
    print('TRAINING COMPLETE')

    # print('evaluating')
    # evaluate_experiment(cfg, experiment_id)
    # print('EVALUATION COMPLETE')


def update_optimizer_lr_finder(criterion, device, model, optimizer, train_loader):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    lrs = lr_finder.history["lr"][10:-5]
    losses = lr_finder.history["loss"][10:-5]
    min_grad_idx = (np.gradient(np.array(losses))).argmin()
    lr = lrs[min_grad_idx]
    print(f"Setting LR to {lr} according to LR finder.")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer, lr


def create_train_val_loaders(cfg):
    tcfg = cfg["train"]

    # Load the training and validation datasets.
    closed_dataset_train, closed_dataset_val = get_datasets(cfg, tcfg["pretrained"], tcfg["image_resize"])

    # for each idx, the class weight = total samples / number of samples in that class
    total_samples = len(closed_dataset_train)
    n_classes = len(np.unique(closed_dataset_train.target))
    class_sample_count = np.array([len(np.where(closed_dataset_train.target == t)[0])
                                   for t in np.unique(closed_dataset_train.target)])
    weight_per_class = 1 / class_sample_count
    weight_per_class = weight_per_class * n_classes  # sum 130549.9844
    # weight_per_class = weight_per_class * n_classes / weight_per_class.sum()
    # weight_per_class = total_samples / class_sample_count
    class_weights = torch.tensor(weight_per_class, dtype=torch.float32)  # sum 12304482
    # sum of the weights should be equal to n_classes so that LR doesn't need to be adjusted

    # Load the training and validation data loaders.
    train_loader, val_loader = get_data_loaders(closed_dataset_train, closed_dataset_val, tcfg["batch_size"],
                                                tcfg["num_dataloader_workers"],
                                                balanced_sampler=tcfg["balanced_sampler"],
                                                timeout=tcfg["worker_timeout_s"])
    print(f"Number of training images: {len(closed_dataset_train)}")
    print(f"Number of validation images: {len(closed_dataset_val)}")
    return train_loader, val_loader, class_weights


if __name__ == '__main__':
    train_model()
