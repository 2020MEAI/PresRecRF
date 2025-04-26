#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
from gensim.models import Word2Vec
import pandas as pd
import networkx as nx


class Logger(object):
    def __init__(self, filename='Default.log'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'a', encoding='utf-8')
        self.encoding = self.file.encoding

    def write(self, message):
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class PreDataset(torch.utils.data.Dataset):
    def __init__(self, a, b, c, d):
        self.S_array, self.Sd_array, self.T_array, self.H_array = a, b, c, d

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        sdid = self.Sd_array[idx]
        tid = self.T_array[idx]
        hid = self.H_array[idx]
        return sid, sdid, tid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetLung(torch.utils.data.Dataset):
    def __init__(self, a, b, c):
        self.S_array, self.Sd_array, self.H_array = a, b, c

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        sdid = self.Sd_array[idx]
        hid = self.H_array[idx]
        return sid, sdid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetPTM(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.S_array, self.H_array = a, b

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        hid = self.H_array[idx]
        return sid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetDosage(torch.utils.data.Dataset):
    def __init__(self, a, b, c):
        self.S_array, self.H_array, self.H_d_array = a, b, c

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        hid = self.H_array[idx]
        h_did = self.H_d_array[idx]
        return sid, hid, h_did

    def __len__(self):
        return self.S_array.shape[0]
