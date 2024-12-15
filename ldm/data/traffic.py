import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import random


class TrafficBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size = None,
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        traffic_npy = np.load(path)

        # traffic_npy = np.array(traffic_npy).astype(np.uint8)
        traffic_npy[:,:,0][traffic_npy[:,:,1] > 1] = 1

        traffic_npy[:,:,0] = (traffic_npy[:,:,0] / 1.0).astype(np.float32)


        example['image'] = traffic_npy

        if 'train' in path:
            textpath = './datasets/traffic/train/' + 'text/' + path.split('/')[-1].split('.')[0] + '.txt'
        else:
            textpath = './datasets/traffic/validation/' + 'text/' + path.split('/')[-1].split('.')[0] + '.txt'
        with open(textpath, "r") as f:
            text = str(f.read().splitlines()[0])
        example['caption'] = text
        return example




class TrafficTrain(TrafficBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TrafficValidation(TrafficBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
