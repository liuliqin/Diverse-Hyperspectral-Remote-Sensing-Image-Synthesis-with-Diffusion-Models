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


def load_data(
    *,
    data_dir,
    code_dir,
    batch_size,
    image_size,
    deterministic=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(os.path.join(data_dir,'multi'))

    dataset = ImageDataset(
        image_size,
        all_files,
        code_dir,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif","tif",".mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        code_dir,
        down_factor = 2,
    ):
        super().__init__()
        self.local_images=image_paths
        self.latent_data = []
        self.multi_data = []
        self.down_factor = 2**(down_factor)
        self.resolution = resolution//self.down_factor

        for i, img_path in enumerate(self.local_images):
            latent_data = scio.loadmat(img_path.replace(os.path.split(img_path)[0],code_dir).replace('.tif','.mat'))
            multi_data =  cv2.imread(img_path).astype('float32')
            self.multi_data.append(multi_data.transpose(2, 0, 1) / 127.5 - 1)
            self.latent_data.append(latent_data['code'].astype('float32')/0.24)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx): # get item

        latent_data = self.latent_data[idx]
        hs, ws, h, w = self.randomcrop(latent_data, self.resolution)
        latentc = latent_data[:, hs:hs + h, ws:ws + w]
        hs = hs * self.down_factor
        ws = ws * self.down_factor
        h = h * self.down_factor
        w = w * self.down_factor
        multi_imgc = self.multi_data[idx][:, hs:hs + h, ws:ws + w]
        s = self.local_images[idx].split('\\')[-1]
        return  latentc,multi_imgc,s

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