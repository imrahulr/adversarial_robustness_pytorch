import torch

import torchvision
import torchvision.transforms as transforms

import re
import numpy as np

from .semisup import SemiSupervisedDataset
from .semisup import SemiSupervisedSampler


def load_cifar10s(data_dir, use_augmentation=False, aux_take_amount=None, 
                  aux_data_filename='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle', 
                  validation=False):
    """
    Returns semisupervised CIFAR10 train, test datasets and dataloaders (with Tiny Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('cifar10s', 'cifar10', data_dir)
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=True, download=True, 
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=validation)
    test_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=False, download=True, 
                                         transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar10', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.dataset_size = len(self.dataset)
