import torch
from torch.optim.lr_scheduler import _LRScheduler

class MultiStepLRScheduler(_LRScheduler):
    def __init__(self, optimizer, drop_steps, drop_factor=0.1, last_epoch=-1):
        self.drop_steps = drop_steps
        self.drop_factor = drop_factor
        super(MultiStepLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = 1.0
        for step in self.drop_steps:
            if self.last_epoch >= step:
                factor *= self.drop_factor
        return [base_lr * factor for base_lr in self.base_lrs]

# IN TRAINER:
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
import torch

class CustomSchedulerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs['optimizer']
        scheduler = kwargs['lr_scheduler']
        if scheduler is not None:
            scheduler.step()

# Define the drop steps and drop factor
drop_steps = [5, 10, 15]  # Epochs at which to drop the learning rate
drop_factor = 0.1  # Factor by which to drop the learning rate

scheduler = MultiStepLRScheduler(optimizer, drop_steps, drop_factor)