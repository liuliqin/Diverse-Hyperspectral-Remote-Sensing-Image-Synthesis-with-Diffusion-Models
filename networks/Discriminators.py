import torch
import torch.nn as nn
import torch.nn.functional as F
import random,math
from .weight_initial import normal_init


class spa_discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, input_nc=6):
        super(spa_discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        # x = torch.cat([input, label], 1)

        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        # x = self.conv5(x)
        return x

class spe_discriminator(nn.Module):
    def __init__(self,input_nc=150,inter=64,d=128):
        super(spe_discriminator, self).__init__()
        self.inter=inter
        self.fc_1=nn.Linear(input_nc,d)
        self.fc_2=nn.Linear(d,2 * d)
        # self.fc_2_bn=nn.BatchNorm1d(2 * d)
        self.fc_3 = nn.Linear(2 * d, 4 * d)
        # self.fc_3_bn=nn.BatchNorm1d(4 * d)
        self.fc_4=nn.Linear(4 * d,8 * d)
        # self.fc_4_bn=nn.BatchNorm1d(8 * d)
        self.fc_5 = nn.Linear(8 * d, 4 * d)
        # self.fc_5_bn = nn.BatchNorm1d(4 * d)
        self.fc_6=nn.Linear(4 * d,1)
    def forward(self,input_real,result):
        location_h = random.randrange(self.inter)
        location_w = random.randrange(self.inter)
        numbers=int(math.ceil(input_real.shape[2]/(self.inter)))
        real=torch.zeros(numbers,numbers,input_real.shape[0]).cuda()
        pre =torch.zeros(numbers,numbers,result.shape[0]).cuda()
        for h in range(numbers):
            for w in range(numbers):
                loc_h=location_h+self.inter*h
                loc_w=location_w+self.inter*w
                real_spectral=input_real[:,:,loc_h,loc_w]
                pre_spectral=result[:,:,loc_h,loc_w]
                spe2=F.leaky_relu(self.fc_2(F.leaky_relu((self.fc_1(real_spectral)))))
                spe3=F.leaky_relu(self.fc_3(spe2))
                spe4 = F.leaky_relu(self.fc_4(spe3))
                spe5 = F.leaky_relu(self.fc_5(spe4))
                real[h, w,:] = torch.sigmoid(F.leaky_relu(self.fc_6(spe5))).squeeze()
                pre2=F.leaky_relu(self.fc_2(F.leaky_relu((self.fc_1(pre_spectral)))))
                pre3=F.leaky_relu(self.fc_3(pre2))
                pre4 = F.leaky_relu(self.fc_4(pre3))
                pre5 = F.leaky_relu(self.fc_5(pre4))
                pre[h, w,:] = torch.sigmoid(F.leaky_relu(self.fc_6(pre5))).squeeze()
        return real.permute(2,0,1),pre.permute(2,0,1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)