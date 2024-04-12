
import os
from osgeo import gdal
import random
from torch.utils.data import Dataset
import torch
from torchvision import transforms as transforms
import blobfile as bf
import cv2

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif","tif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
class ImageDataset(Dataset):
    def __init__(self, folder,crop_size):
        self.folder = folder
        # self.path = os.listdir(folder)
        self.resolution=crop_size
        self.hyper_data = []
        self.multi_data = []
        self.local_images = _list_image_files_recursively(os.path.join(folder, 'hyper'))
        #hyper_path = self.folder+'/'+self.hyper_path[index]
        for i, img_path in enumerate(self.local_images):
            dataset = gdal.Open(img_path)
            if dataset is None:
                print("cannot open %s" % img_path)
                exit(-1)
            im_data = dataset.ReadAsArray()
            self.hyper_data.append(im_data.astype('float32')/4095)

            multi_img = cv2.imread(img_path.replace("hyper", "multi")).astype('float32')
            self.multi_data.append(multi_img.transpose(2, 0, 1) / 127.5 - 1)


    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        hyper_img = self.hyper_data[idx]
        hs, ws, h, w = self.randomcrop(hyper_img, self.resolution)
        hyper_imgc = hyper_img[0:48, hs:hs + h, ws:ws + w]
        multi_imgc = self.multi_data[idx][:, hs:hs + h, ws:ws + w]
        # abun_c = self.abun_data[idx][:, hs:hs + h, ws:ws + w]
        s = self.local_images[idx].split('\\')[-1]
        return  hyper_imgc,multi_imgc,s

    def randomcrop(self,img,crop_size):
        c,h,w= img.shape
        th= tw = crop_size
        if w == tw and h == th:
            return 0,0,h,w
        if w < tw or h < th:
            print('the size of the image is not enough for crop!')
            exit(-1)
        i = random.randint(0, h - th-10)
        j = random.randint(0, w - tw-10)
        return i,j,th,tw






