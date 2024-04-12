import numpy as np
import cv2
import math
from visdom import Visdom

import sys
import time
import datetime
import torch.nn as nn
import torch
from osgeo import gdal
import json


def tensor2image(tensor):
    # imtensor=tensor[0]
    # if imtensor.size()[0]== 3:
    #     imbinar = imtensor
    # elif imtensor.size()[0]<3:
    #     imbinar=np.tile(imtensor[0],(3,1,1))
    # else:
    #     imbinar = imtensor[[35, 18, 8], :, :]
    # image = 127.5 * (imbinar.cpu().float().numpy() + 1.0)
    imtensor = tensor[0]
    if imtensor.dim()<3:
        imbinar = imtensor.repeat(3, 1, 1)
        image = 42 * (imbinar.cpu().float().numpy())
    elif imtensor.dim()== 3 and imtensor.size()[0] == 3:
        imbinar = imtensor
        image = 127.5 * (imbinar.cpu().float().numpy()+1.0)
    else:
        imbinar = imtensor[[35, 18, 8], :, :]
        image = 127.5 * (imbinar.cpu().float().numpy() + 1.0)
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, window_name = 'main'):
        self.viz = Visdom(env=window_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            # print(tensor.requires_grad)
            if image_name not in self.image_windows:

                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.detach()), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.detach()), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



#def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    # test_images = G(x_)
    #
    # size_figure_grid = 3
    # fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    # for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for i in range(x_.size()[0]):
    #     ax[i, 0].cla()
    #     ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 1].cla()
    #     ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 2].cla()
    #     ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)
    #
    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    #
    # if save:
    #     plt.savefig(path)
    #
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

    def save(self,file_path='log.log',env_name='main'):
        # Visdom.save()
        self.create_log_at(file_path,env_name)

    def create_log_at(self,file_path, current_env, new_env=None):
        new_env = current_env if new_env is None else new_env
        vis = Visdom(env=current_env)

        data = json.loads(vis.get_window_data())
        if len(data) == 0:
            print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
            return

        file = open(file_path, 'w+')
        for datapoint in data.values():
            output = {
                'win': datapoint['id'],
                'eid': new_env,
                'opts': {}
            }

            if datapoint['type'] != "plot":
                output['data'] = [{'content': datapoint['content'], 'type': datapoint['type']}]
                if datapoint['height'] is not None:
                    output['opts']['height'] = datapoint['height']
                if datapoint['width'] is not None:
                    output['opts']['width'] = datapoint['width']
            else:
                output['data'] = datapoint['content']["data"]
                output['layout'] = datapoint['content']["layout"]

            to_write = json.dumps(["events", output])
            file.write(to_write + '\n')
        file.close()

def write_img(filename, im_data):#im_geotrans, im_proj,
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    #dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    #dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def batch_PSNR(im_true, im_fake, data_range=4095):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2. * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)
