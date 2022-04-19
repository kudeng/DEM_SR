# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import sys

import torch
import numpy as np
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from .metrics import tensor_accessment
from .utils import quantize


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.loss = loss
        self.lr = lr
        if self.model_name in ['edsr_net', 'dbpn_net']:
            self.task = 'sr'
        elif self.model_name in ['dkn_net']:
            self.task = 'g_sr'
        else:
            self.task = 'mde_sr'
        self.test_mde = kargs['test_mde']
        self.load_model()
        self.configure_loss()

    def forward(self, *datas):
        return self.model(*datas)

    def training_step(self, batch, batch_idx):
        lr, hr, img, _ = batch
        if self.task == 'sr':
            sr = self(lr)
            loss = self.loss_function(sr, hr)
        elif self.task == 'g_sr':
            sr = self(lr, img)
            loss = self.loss_function(sr, hr)
        elif self.task == 'bid_net':
            mde, sr, sr_2 = self(img, lr)
            loss_sr = self.loss_function(sr, hr)
            loss_de = self.loss_function(mde, hr)
            loss_sr_2 = self.loss_function(F.interpolate(sr_2, scale_factor=2, mode='bicubic'), hr)
            loss = loss_sr + loss_de + loss_sr_2
        else:
            mde, sr = self(img, lr)
            loss_sr = self.loss_function(sr, hr)
            loss_de = self.loss_function(mde, hr)
            loss = loss_sr + loss_de
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, img, _ = batch
        if self.task == 'sr':
            sr = self(lr)
        elif self.task == 'g_sr':
            sr = self(lr, img)
        else:
            mde, sr, *other = self(img, lr)
            if self.test_mde:
                mde_psnr = tensor_accessment(
                    x_pred=mde.cpu().numpy(),
                    x_true=hr.cpu().numpy(),
                    data_range=self.hparams.color_range)
                self.log('mde_psnr', mde_psnr, on_step=False, on_epoch=True, prog_bar=True)

        psnr = tensor_accessment(
            x_pred=sr.cpu().numpy(),
            x_true=hr.cpu().numpy(),
            data_range=self.hparams.color_range)

        self.log('psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)



    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        pass
        # Make the Progress Bar leave there
        # self.print(self.get_progress_bar_dict())

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     if hasattr(self.hparams, 'weight_decay'):
    #         weight_decay = self.hparams.weight_decay
    #     else:
    #         weight_decay = 0
    #     optimizer_sr = torch.optim.Adam(
    #         self.sr_parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
    #     optimizer_de = torch.optim.Adam(
    #         self.de_parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
    #
    #     if self.hparams.lr_scheduler is None:
    #         return optimizer_sr
    #     else:
    #         if self.hparams.lr_scheduler == 'step':
    #             scheduler_sr = lrs.StepLR(optimizer_sr,
    #                                    step_size=self.hparams.lr_decay_steps,
    #                                    gamma=self.hparams.lr_decay_rate)
    #             scheduler_de = lrs.StepLR(optimizer_de,
    #                                    step_size=self.hparams.lr_decay_steps,
    #                                    gamma=self.hparams.lr_decay_rate)
    #         elif self.hparams.lr_scheduler == 'cosine':
    #             scheduler_sr = lrs.CosineAnnealingLR(optimizer_sr,
    #                                               T_max=self.hparams.lr_decay_steps,
    #                                               eta_min=self.hparams.lr_decay_min_lr)
    #             scheduler_de = lrs.CosineAnnealingLR(optimizer_de,
    #                                               T_max=self.hparams.lr_decay_steps,
    #                                               eta_min=self.hparams.lr_decay_min_lr)
    #         else:
    #             raise ValueError('Invalid lr_scheduler type!')
    #         return [optimizer_sr, optimizer_de], [scheduler_sr, scheduler_de]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        # from model.bid_net import BidNet
        # from model.bid_net_v2 import BidNetV2
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        # Model = BidNet
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


    def sr_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            # sys.stdout.write(name)
            _name = name[6:name.index('.', 6)]
            # sys.stdout.write(_name+'#$$#')
            if 'sr_' in _name or 'guidance' in _name:
                yield param


    def de_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            # sys.stdout.write(name)
            _name = name[6:name.index('.', 6)]
            # sys.stdout.write(_name+'#$$#')
            if 'de_' in _name or 'correction' in _name:
                yield param