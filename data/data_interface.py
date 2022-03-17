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
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


class DInterface(pl.LightningDataModule):

    def __init__(self,
                 num_workers=0,
                 dataset='pair_data',
                 datatype='nyuv2',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.scale = kwargs['scale']
        self.datatype = datatype
        self.kwargs = kwargs
        self.patched = kwargs['patched']
        self.patch_size = kwargs['patch_size']
        self.batch_size = kwargs['batch_size']
        # self.setup(kwargs['stage'])
        self.load_data_module()



    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if self.datatype == 'nyuv2':
            if stage == 'fit' or stage is None:
                self.trainset = self.instancialize(
                    hr_dir='data/splited/nyuv2/train/hr',
                    img_dir='data/splited/nyuv2/train/img',
                    hr_size=(640, 480),
                    generate_from_hr=True,
                    scale_factor=self.scale,
                    suffix='png',
                    transform=ToTensor(),
                    target_transform=ToTensor(),
                    guided_transform=ToTensor(),
                    patched=self.patched,
                    patch_size=self.patch_size
                )
                self.valset = self.instancialize(
                    hr_dir='data/splited/nyuv2/val/hr',
                    img_dir='data/splited/nyuv2/val/img',
                    hr_size=(640, 480),
                    generate_from_hr=True,
                    scale_factor=self.scale,
                    suffix='png',
                    transform=ToTensor(),
                    target_transform=ToTensor(),
                    guided_transform=ToTensor(),
                    patched=False
                )
            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testset = self.instancialize(
                    hr_dir='data/splited/nyuv2/test/hr',
                    img_dir='data/splited/nyuv2/test/img',
                    hr_size=(640, 480),
                    generate_from_hr=True,
                    scale_factor=self.scale,
                    suffix='png',
                    transform=ToTensor(),
                    target_transform=ToTensor(),
                    guided_transform=ToTensor(),
                    patched=False
                )
        elif self.datatype == 'moon2':
            if stage == 'fit' or stage is None:
                self.trainset = self.instancialize(
                    hr_dir='data/splited/moon2/train/hr',
                    img_dir='data/splited/moon2/train/img',
                    lr_dir='data/splited/moon2/train/lr',
                    hr_size=(1024, 1024),
                    generate_from_hr=False,
                    scale_factor=self.scale,
                    suffix='tif',
                    transform=None,
                    target_transform=None,
                    guided_transform=ToTensor(),
                    patched=self.patched,
                    patch_size=self.patch_size
                )
                self.valset = self.instancialize(
                    hr_dir='data/splited/moon2/val/hr',
                    img_dir='data/splited/moon2/val/img',
                    lr_dir='data/splited/moon2/val/lr',
                    hr_size=(1024, 1024),
                    generate_from_hr=False,
                    scale_factor=self.scale,
                    suffix='tif',
                    transform=None,
                    target_transform=None,
                    guided_transform=ToTensor(),
                    patched=False
                )
            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testset = self.instancialize(
                    hr_dir='data/splited/moon2/test/hr',
                    img_dir='data/splited/moon2/test/img',
                    lr_dir='data/splited/moon2/test/lr',
                    hr_size=(1024, 1024),
                    generate_from_hr=False,
                    scale_factor=self.scale,
                    suffix='tif',
                    transform=None,
                    target_transform=None,
                    guided_transform=ToTensor(),
                    patched=False
                )
        else:
            print('No such dataset')


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset

        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
