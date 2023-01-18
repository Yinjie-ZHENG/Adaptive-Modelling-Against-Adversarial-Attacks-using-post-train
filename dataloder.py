# -*- coding:UTF-8 -*-
"""
dataset and  data reading
"""

import os
import sys
from PIL import Image
# import glob
# import json
# import functools
import numpy as np
# import pandas as pd
# from osgeo import gdal
# import albumentations as albu
# from skimage.color import gray2rgb
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
#
#
# from utils.arg_utils import *
# from utils.data_utils import *
# from utils.algorithm_utils import *
from MLclf import MLclf

from autoaug.augmentations import Augmentation
from autoaug.archive import fa_reduced_cifar10,autoaug_paper_cifar10,fa_reduced_imagenet
import autoaug.aug_transforms as aug

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset

from dataset_loder.scoliosis_dataloder import ScoliosisDataset
from autoaug.cutout import Cutout

def training_transforms():
    return transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616]),
            # Cutout()
        #[125.3, 123.0, 113.9],[63.0, 62.1, 66.7]
        ])
def validation_transforms():
    return transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2471, 0.2435, 0.2616]),
        ])

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=transforms.ToTensor()):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt








def load_dataset(data_config):
    if data_config.dataset == 'cifar10':
        training_transform=training_transforms()
        if data_config.autoaug:
            print('auto Augmentation the data !')
            training_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        train_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                     train=True,
                                                     transform=training_transform,
                                                     download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                   train=False,
                                                   transform=validation_transforms(),
                                                   download=True)
        return train_dataset,val_dataset
    elif data_config.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                     train=True,
                                                     transform=training_transforms(),
                                                     download=True)
        val_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                   train=False,
                                                   transform=validation_transforms(),
                                                   download=True)
        return train_dataset, val_dataset

    elif data_config.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root=data_config.data_path,
                                                     train=True,
                                                     transform=transforms.ToTensor(),
                                                     download=True)
        val_dataset = torchvision.datasets.MNIST(root=data_config.data_path,
                                                   train=False,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        return train_dataset, val_dataset

        '''elif data_config.dataset == 'tiny_imagenet':
        train_dataset, validation_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                         seed_value=None, shuffle=True,
                                                                                         transform=None,
                                                                                         #transforms.Compose([
                                                                                         #    transforms.RandomHorizontalFlip(),
                                                                                         #    transforms.ToTensor(),
                                                                                         #]),
                                                                                         save_clf_data=True,
                                                                                         few_shot=False)
        return train_dataset, validation_dataset'''




    elif data_config.dataset == 'tiny_imagenet':
        data_path='/home/yinjie/FYP_/torch/dataset/tiny-imagenet-200'#'/hdd7/yinjie/tiny-imagenet-200'   #'/hdd7/yinjie/tiny-imagenet-200-dropusse'


        train_dataset = TinyImageNet(data_path, train=True, transform=training_transforms())
        val_dataset = TinyImageNet(data_path, train=False, transform=validation_transforms())
        # traindir = data_path + '/train'
        # valdir = data_path + '/val'
        # testdir = data_path + '/test'
        # train_dataset = torchvision.datasets.ImageFolder(traindir,
        #                                                  transforms.Compose([
        #                                                      #transforms.RandomResizedCrop(64),
        #                                                      # transforms.RandomCrop(64, padding=4),
        #                                                      transforms.RandomHorizontalFlip(),
        #                                                      transforms.ToTensor(),
        #                                                       #normalize
        #                                                      ]))
        # val_dataset = torchvision.datasets.ImageFolder(testdir,
        #                                                transforms.Compose([
        #                                                    #transforms.Resize(64),
        #                                                    # transforms.RandomResizedCrop(224),
        #                                                    transforms.ToTensor(),
        #                                                    #normalize
        # ]))
        return train_dataset, val_dataset

    elif data_config.dataset == 'imagenet':
        traindir = data_config.data_path+'/imagenet/train'
        valdir =data_config.data_path+'/imagenet/val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        jittering =aug.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = aug.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                                                         transforms.Compose([
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             jittering, lighting, normalize, ]))
        val_dataset = torchvision.datasets.ImageFolder(valdir,
                                                         transforms.Compose([
                                                             transforms.Resize(256),
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.ToTensor(),
                                                             normalize, ]))
        return train_dataset, val_dataset
    elif data_config.dataset == 'scoliosis':
        # traindir = data_config.data_path + '/train'
        # valdir = data_config.data_path + '/test'
        normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        jittering = aug.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
        lighting = aug.Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize])
        train_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))#fa_reduced_cifar10,autoaug_paper_cifar10,fa_reduced_imagenet
        train_dataset =ScoliosisDataset(data_config.data_path,
                                        transform=train_transforms,#,jittering, lighting,transforms.RandomHorizontalFlip(),
                                        train=True)
        val_dataset = ScoliosisDataset(data_config.data_path,
                                         target_transform=test_transforms,
                                         train=False)
        return train_dataset, val_dataset
    elif data_config.dataset == 'SCUT-FBP5500':
        data_path=data_config.data_path
        trainfile = data_config.label_file + '/train.txt'
        valfile = data_config.label_file + '/test.txt'
        normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])

        train_dataset =FacialAttractionDataset(data_path,trainfile,
                                        transform=transforms.Compose([
                                            transforms.Resize(224),
                                            # transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),normalize]),
                                        )
        val_dataset = FacialAttractionDataset(data_path,valfile ,
                                         transform=transforms.Compose([
                                             transforms.Resize(224),
                                             transforms.ToTensor(),normalize]),
                                         )
        return train_dataset, val_dataset

    elif data_config.dataset == 'sco_fa':
        source_dir=data_config.source_dir
        taget_dir=data_config.taget_dir

        trainfile = data_config.label_file + '/train.txt'
        valfile = data_config.label_file + '/test.txt'

        source_normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])
        target_normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        source_transforms=transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            source_normalize])
        target_transforms =transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            target_normalize])
        # source_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        target_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        train_dataset =ScoandFaDataset(source_dir=source_dir,
                                       taget_dir=taget_dir+'train',
                                       label_file=trainfile,
                                       source_transform=source_transforms,
                                       target_transform=target_transforms
                                        )
        val_dataset = ScoandFaDataset(source_dir=source_dir,
                                       taget_dir=taget_dir+'test',
                                       label_file=valfile,
                                       source_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),source_normalize]),
                                       target_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),target_normalize]),
                                        )
        return train_dataset, val_dataset
    elif data_config.dataset == 'scofa':
        source_dir=data_config.source_dir
        taget_dir=data_config.taget_dir

        source_normalize = transforms.Normalize(mean=[0.22, 0.37, 0.73],
                                         std=[1.61, 1.75, 1.80])
        target_normalize = transforms.Normalize(mean=[0.64, 0.53, 0.43],
                                         std=[0.20, 0.19, 0.19])
        source_transforms=transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            source_normalize])
        target_transforms =transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            target_normalize])
        # source_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        target_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        train_dataset =ScoandFaNshotDataset(source_dir=source_dir+'train',
                                       taget_dir=taget_dir+'train',
                                       source_transform=source_transforms,
                                       target_transform=target_transforms
                                        )
        val_dataset = ScoandFaNshotDataset(source_dir=source_dir+'test',
                                       taget_dir=taget_dir+'test',
                                       source_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),source_normalize]),
                                       target_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),target_normalize]),
                                        )
        return train_dataset, val_dataset
    elif data_config.dataset == 'megaage_asian':
        train_path = data_config.data_path+'train'
        val_path = data_config.data_path+'test'
        trainfile = data_config.label_file + 'train_age.txt'
        valfile = data_config.label_file + 'test_age.txt'
        normalize = transforms.Normalize(mean=[0.54, 0.47, 0.44],
                                          std=[0.29, 0.28, 0.28])

        train_dataset = MegaAsiaAgeDataset(train_path, trainfile,
                                                transform=transforms.Compose([
                                                    # transforms.Resize(224),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize]),
                                                )
        val_dataset = MegaAsiaAgeDataset(val_path, valfile,
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.RandomResizedCrop(224),
                                                  # transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  normalize]),
                                              )
        return train_dataset, val_dataset

    else:
        raise Exception('unknown dataset: {}'.format(data_config.dataset))


