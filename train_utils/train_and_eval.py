import os
import errno
import torch
from torch import nn
import numpy as np
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_results(confmat, dice, results_file, train_info):
    val_info = str(confmat)
    print(val_info)
    print(f"dice coefficient: {dice:.4f}")
    # write into txt
    with open(results_file, "a") as f:
        f.write(train_info + val_info + "\n\n")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt', trace_func=print, save_model=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_dice = None
        self.early_stop = False
        self.best_score = None
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func
        self.save_model = save_model
    def __call__(self, dice):

        score = dice
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    name = 'out'
    x = inputs
    loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes, header = 'Val:'):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 400, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    loss_weight = None
    idx = 1

    for image, target in metric_logger.log_every(data_loader, print_freq, header):

        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step_update(epoch*len(data_loader)+idx)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        idx += 1

    return metric_logger.meters["loss"].global_avg, lr