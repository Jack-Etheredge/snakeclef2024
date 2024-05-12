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
from datasets import get_datasets, get_data_loaders, get_dataloader_combine_and_balance_datasets, get_openset_datasets
from evaluate import evaluate_experiment
from utils import save_plots, checkpoint_model, copy_config
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
def train(model, trainloader, optimizer, criterion, loss_function_id, max_norm, device='cpu'):
    model.train()
    print(f'Training with device {device}')
    train_running_loss = 0.0
    train_running_correct = 0
    m = nn.Softmax(dim=-1)
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
def validate(model, testloader, criterion, loss_function_id, device='cpu'):
    model.eval()
    print(f'Validation with device {device}')
    valid_running_loss = 0.0
    valid_running_correct = 0
    m = nn.Softmax(dim=-1)
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
    use_poison_loss = cfg["train"]["use_poison_loss"]
    num_dataloader_workers = cfg["train"]["num_dataloader_workers"]
    validation_frac = cfg["train"]["validation_frac"]
    max_norm = cfg["train"]["max_norm"]
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    weight_decay = cfg["train"]["weight_decay"]
    balanced_sampler = cfg["train"]["balanced_sampler"]
    use_lr_finder = cfg["train"]["use_lr_finder"]
    fine_tune_after_n_epochs = cfg["train"]["fine_tune_after_n_epochs"]
    skip_frozen_epochs_load_failed_model = cfg["train"]["skip_frozen_epochs_load_failed_model"]
    lr_scheduler = cfg["train"]["lr_scheduler"]
    lr_scheduler_patience = cfg["train"]["lr_scheduler_patience"]
    include_unknowns = cfg["train"]["include_unknowns"]
    n_classes = cfg["train"]["n_classes"]
    worker_timeout_s = cfg["train"]["worker_timeout_s"]

    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]

    for k, v in cfg["train"].items():
        if v == "None" or v == "null":
            url = "https://stackoverflow.com/questions/76567692/hydra-how-to-express-none-in-config-files"
            raise ValueError(f"`{k}` set to 'None' or 'null'. Use `null` for None values in hydra; see {url}")

    experiment_id = cfg["train"]["experiment_id"]
    if experiment_id is None:
        experiment_id = str(datetime.now())
        cfg["train"]["experiment_id"] = experiment_id

    if balanced_sampler and (oversample or undersample):
        raise ValueError("cannot use balanced sampler with oversample or undersample")

    # use experiment_id to save model checkpoint, graphs, predictions, performance metrics, etc
    experiment_dir = CHECKPOINT_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = str(experiment_dir / f"model.pth")
    accuracy_plot_path = str(experiment_dir / "accuracy_plot.png")
    loss_plot_path = str(experiment_dir / "loss_plot.png")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(str(experiment_dir / "experiment_config.json"), "w") as f:
        json.dump(config_dict, f)
    copy_config("train_progressive", experiment_id)

    image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch=0)

    # torch.autograd.set_detect_anomaly(True)

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(
        model_id=model_id,
        pretrained=pretrained,
        fine_tune=not fine_tune_after_n_epochs or skip_frozen_epochs_load_failed_model,
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
        # update settings based on epoch from checkpoint
        image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch=start_epoch)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]["lr"] = lr
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

    train_loader, val_loader = create_train_val_loaders(cfg, image_size, batch_size)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    if multiclass_loss_function == "seesaw":
        multiclass_loss = SeesawLoss(num_classes=n_classes, device=device)
    elif multiclass_loss_function == "focal":
        multiclass_loss = FocalLoss(gamma=0.7)
    elif multiclass_loss_function == "crossentropy":
        multiclass_loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {multiclass_loss_function}")
    criterion = CompositeLoss(multiclass_loss=multiclass_loss, use_poison_loss=use_poison_loss)

    if use_lr_finder:
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=100, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
        min_grad_idx = (np.gradient(np.array(lr_finder.history["loss"]))).argmin()
        lr = lr_finder.history["lr"][min_grad_idx]
        print(f"Setting LR to {lr} according to LR finder.")
        optimizer.param_groups[0]["lr"] = lr

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    unfrozen = fine_tune_after_n_epochs == 0
    for epoch in range(start_epoch, epochs):

        image_size, dropout_rate, batch_size = get_progression_params(cfg, epoch)
        model = update_dropout_rate(model, dropout_rate)

        if (not unfrozen and (skip_frozen_epochs_load_failed_model or
                              epoch + 1 > fine_tune_after_n_epochs)):
            model = unfreeze_model(model)
            print("all layers unfrozen")
            optimizer, scheduler = create_scheduler_and_optimizer(model, lr, weight_decay, lr_scheduler,
                                                                  lr_scheduler_patience)
            unfrozen = True

        print(
            f"Epoch {epoch + 1} of {epochs}, image_size: {image_size}, dropout: {dropout_rate}, batch size: {batch_size}")
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"current learning rate: {curr_lr:.0e}")
        recreate_loader = True
        while recreate_loader:
            try:
                train_loader, val_loader = create_train_val_loaders(cfg, image_size, batch_size)
                train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                          optimizer, criterion, multiclass_loss_function, max_norm,
                                                          device)
                recreate_loader = False
            except Exception as e:
                print("issue with training")
                print(e)
                print("recreating data loaders and trying again")

        recreate_loader = True
        while recreate_loader:
            try:
                train_loader, val_loader = create_train_val_loaders(cfg, image_size, batch_size)
                valid_epoch_loss, valid_epoch_acc = validate(model, val_loader,
                                                             criterion, multiclass_loss_function, device)
                if scheduler:
                    scheduler.step(valid_epoch_loss)
                recreate_loader = False
            except Exception as e:
                print("issue with validation")
                print(e)
                print("recreating data loaders and trying again")

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

    print('evaluating')
    evaluate_experiment(cfg, experiment_id)
    print('EVALUATION COMPLETE')


def create_train_val_loaders(cfg, image_size, batch_size):
    tcfg = cfg["train"]
    osrcfg = cfg["open-set-recognition"]

    # Load the training and validation datasets.
    closed_dataset_train, closed_dataset_val, _ = get_datasets(tcfg["pretrained"], image_size,
                                                               tcfg["validation_frac"],
                                                               oversample=tcfg["oversample"],
                                                               undersample=tcfg["undersample"],
                                                               oversample_prop=tcfg["oversample_prop"],
                                                               equal_undersampled_val=tcfg[
                                                                   "equal_undersampled_val"],
                                                               include_metadata=tcfg[
                                                                   "use_metadata"])

    if tcfg["include_unknowns"]:
        open_dataset_train, open_dataset_val, _ = get_openset_datasets(pretrained=tcfg["pretrained"],
                                                                       image_size=image_size,
                                                                       n_train=osrcfg["openset_n_train"],
                                                                       n_val=osrcfg["openset_n_val"],
                                                                       training_augs=True,
                                                                       include_metadata=tcfg[
                                                                           "use_metadata"])
        print("[[train]] combining dataloaders and balancing classes")
        train_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_train, open_dataset_train,
                                                                   batch_size=batch_size, unknowns=True,
                                                                   timeout=tcfg["worker_timeout_s"])
        print("[[val]] combining dataloaders and balancing classes")
        val_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_val, open_dataset_val,
                                                                 batch_size=batch_size, unknowns=True,
                                                                 timeout=tcfg["worker_timeout_s"])

        print(f"Number of closed set training images: {len(closed_dataset_train)}")
        print(f"Number of closed set validation images: {len(closed_dataset_val)}")
        print(f"Number of open set training images: {len(open_dataset_train)}")
        print(f"Number of open set validation images: {len(open_dataset_val)}")
    else:
        # Load the training and validation data loaders.
        train_loader, val_loader = get_data_loaders(closed_dataset_train, closed_dataset_val, batch_size,
                                                    tcfg["num_dataloader_workers"],
                                                    balanced_sampler=tcfg["balanced_sampler"],
                                                    timeout=tcfg["worker_timeout_s"])
        print(f"Number of training images: {len(closed_dataset_train)}")
        print(f"Number of validation images: {len(closed_dataset_val)}")
    return train_loader, val_loader


if __name__ == '__main__':
    train_model()
