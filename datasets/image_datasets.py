import math
import random
import os
import cv2
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from osgeo import gdal
import scipy.io as scio

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        latent_paths,
        image_paths,
        down_factor = 2,
    ):
        super().__init__()

        self.latent_paths = latent_paths
        self.image_paths= image_paths
        self.latent_data = []
        self.multi_data = []
        self.hyper_data = []
        self.down_factor = 2**(down_factor)
        self.resolution = resolution //self.down_factor
        self.multi_path = os.listdir(os.path.join(image_paths,'multi'))

        for img_path in self.multi_path:
            dataset = gdal.Open(os.path.join(image_paths, 'hyper', img_path))
            if dataset is None:
                print("cannot open %s" % img_path)
                exit(-1)
            im_data = dataset.ReadAsArray()
            self.hyper_data.append(im_data.astype('float32') / 4095)

            multi_data = cv2.imread(os.path.join(image_paths,'multi',img_path)).astype('float32')
            self.multi_data.append(multi_data.transpose(2, 0, 1) / 127.5 - 1)
            latent_data = scio.loadmat(os.path.join(latent_paths,img_path.replace('.tif', '.mat')))
            self.latent_data.append(latent_data['code'].astype('float32'))

    def __len__(self):
        return len(self.multi_path)

    def __getitem__(self, idx): # get item

        latent_data = self.latent_data[idx]
        hs, ws, h, w = self.randomcrop(latent_data, self.resolution)
        latentc = latent_data[:, hs:hs + h, ws:ws + w]
        hs = hs * self.down_factor
        ws = ws * self.down_factor
        h = h * self.down_factor
        w = w * self.down_factor
        multi_imgc = self.multi_data[idx][:, hs:hs + h, ws:ws + w]
        hyper_c = self.hyper_data[idx][:, hs:hs + h, ws:ws + w]
        s = self.multi_path[idx]#.split('.')[0]
        return  latentc,multi_imgc,hyper_c,s

    def randomcrop(self, img, crop_size):
        c, h, w = img.shape
        th = tw = crop_size
        if w == tw and h == th:
            return 0, 0, h, w
        if w < tw or h < th:
            print('the size of the image is not enough for crop!')
            exit(-1)
        i = random.randint(0, h - th - 1)
        j = random.randint(0, w - tw - 1)
        return i, j, th, tw