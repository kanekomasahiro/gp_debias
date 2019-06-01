import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, lr_decay=1, max_grad_norm=None):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()
        self._updateLearningRate()

    def _updateLearningRate(self):
        self.lr = self.lr*self.lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
