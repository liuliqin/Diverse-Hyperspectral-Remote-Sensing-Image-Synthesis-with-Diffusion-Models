import torch
import torch.nn as nn
from torch.autograd import Variable
from .ssim_loss import MS_SSIM
from utils.util import rgb_to_hsv
import random
class GenerationLoss():
    def __init__(self,opt,hyper_channel,weight_band, close_list=None):
        self.hyper_channel = hyper_channel
        self.device = opt.device
        self.close_list = close_list
        self.loss = {}
        self.L1_loss = nn.L1Loss().to(opt.device)
        self.BCE_loss = nn.BCELoss().to(opt.device)
        if opt.L1_lambda > 0:
            self.loss['G_l1_loss'] = 0
            self.L1_lambda =opt.L1_lambda
            self.weight_l1 = opt.weight_l1
            if opt.weight_l1:
                weight_band = weight_band / torch.mean(weight_band)
                # self.weight_band = weight_band[:,self.close_list,:,:]
        if opt.Cos_lambda > 0:
            self.Cos_loss = nn.CosineSimilarity(dim=1).to(opt.device)
            self.loss['G_cos_loss'] = 0
            self.Cos_lambda = opt.Cos_lambda
        if opt.Ssim_lambda > 0:
            self.SSIM_loss = MS_SSIM(max_val = 1,channel = self.hyper_channel)
            # self.SSIM_loss_vn = MS_SSIM(max_val=1, channel=50)
            self.loss['G_ssim_loss'] = 0
            self.Ssim_lambda = opt.Ssim_lambda
        if opt.Hsv_lambda > 0:
            self.loss['G_hsv_loss'] = 0
            self.Hsv_lambda = opt.Hsv_lambda
        if opt.Abun_reg_lambda > 0:
            self.loss['Abun_regular_loss'] = 0
            self.Abun_reg_lambda = opt.Abun_reg_lambda
            self.abun_num = opt.abun_num
        if opt.Res_reg_lambda > 0:
            self.loss['R_regular_loss'] = 0
            self.Res_reg_lambda = opt.Res_reg_lambda
        if opt.D_start_epoch < opt.train_epoch:
            self.loss['G_spa_loss'] = 0
            self.loss['D_spa_real_loss'] = 0
            self.loss['D_spa_fake_loss'] = 0
            self.loss['D_spa_loss'] = 0
            if opt.spe_lambda > 0:
                self.loss['G_spe_loss'] = 0
                self.loss['D_spe_real_loss'] = 0
                self.loss['D_spe_fake_loss'] = 0
                self.loss['D_spe_loss'] = 0
                self.spe_lambda = opt.spe_lambda
            self.loss['D_loss'] = 0
        self.loss['G_train_loss'] = 0
        self.loss['Embed_loss'] = 0

    def calcul_generation_loss_iter(self,y_,G_result,embed_loss):
        # if not self.close_list is None:
        #     y_ = y_[self.close_list,:,:]
        #     G_result = G_result_old[self.close_list,:,:].clone
        # else:
        #     G_result = G_result_old
        self.loss['Embed_loss'] = embed_loss.clone()
        # self.loss['G_train_loss'] +=embed_loss
        self.loss['G_train_loss'] = embed_loss
        if self.loss.__contains__('G_l1_loss'):
            if self.weight_l1:
                self.loss['G_l1_loss'] = self.L1_loss(self.weight_band*G_result,self.weight_band*y_)
            else:
                self.loss['G_l1_loss'] = self.L1_loss(G_result, y_)
            self.loss['G_train_loss'] += self.L1_lambda * self.loss['G_l1_loss']
        if self.loss.__contains__('G_cos_loss'):
            cos_all_band = 1 - torch.mean(self.Cos_loss(G_result,y_))
            # cos_vn_band = 1 - torch.mean(self.Cos_loss(G_result[:,:50,:,:],y_[:,:50,:,:]))
            self.loss['G_cos_loss'] = cos_all_band #+ cos_vn_band
            self.loss['G_train_loss'] += self.Cos_lambda * self.loss['G_cos_loss']
        if self.loss.__contains__('G_ssim_loss'):
            ssim_all_band = 1- self.SSIM_loss(y_,G_result)
            self.loss['G_ssim_loss'] = ssim_all_band
            self.loss['G_train_loss'] += self.Ssim_lambda * self.loss['G_ssim_loss']
        if self.loss.__contains__('G_hsv_loss'):
            channel = [23,12,5]
            hsv_real = rgb_to_hsv(y_[:,channel,:,:]/16)
            hsv_result = rgb_to_hsv(G_result[:,channel,:,:]/16)
            self.loss['G_hsv_loss'] = self.L1_loss(hsv_result,hsv_real)
            self.loss['G_train_loss'] += self.Hsv_lambda * self.loss['G_hsv_loss']


    def calcul_D_loss_iter(self,D_real_result,D_pre_result,spe_pre_real,spe_pre_result):
        self.loss['D_spa_real_loss'] = self.BCE_loss(D_real_result, Variable(torch.ones(D_pre_result.size()).to(self.device)))
        self.loss['D_spa_fake_loss'] = self.BCE_loss(D_pre_result, Variable(torch.zeros(D_pre_result.size()).to(self.device)))
        self.loss['D_spa_loss'] = (self.loss['D_spa_real_loss'] + self.loss['D_spa_fake_loss']) * 0.5
        self.loss['D_spe_real_loss'] = self.BCE_loss(spe_pre_real, Variable(torch.ones(spe_pre_real.size()).to(self.device)))
        self.loss['D_spe_fake_loss'] = self.BCE_loss(spe_pre_result, Variable(torch.zeros(spe_pre_real.size()).to(self.device)))
        self.loss['D_spe_loss'] = (self.loss['D_spe_real_loss'] + self.loss['D_spe_fake_loss']) * 0.5
        self.loss['D_loss'] = self.loss['D_spa_loss'] + self.spe_lambda * self.loss['D_spe_loss']


    def calcul_G_adver_loss_iter(self,D_pre_result,spe_pre_result):
        self.loss['G_spa_loss'] = self.BCE_loss(D_pre_result, Variable(torch.ones(D_pre_result.size()).to(self.device)))
        self.loss['G_spe_loss'] = self.BCE_loss(spe_pre_result, Variable(torch.ones(spe_pre_result.size()).to(self.device)))
        self.loss['G_train_loss'] += self.loss['G_spa_loss']+self.spe_lambda *self.loss['G_spe_loss']


