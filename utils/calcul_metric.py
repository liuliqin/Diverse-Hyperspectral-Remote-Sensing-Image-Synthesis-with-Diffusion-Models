import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

class Cal():
    def __init__(self,num_range=4095):
        self.num_range=num_range
        self.cal_num=0
        self.sum_mssim = 0
        self.sum_psnr = 0
        self.sum_sam = 0
        self.sum_rmse = 0
        self.sum_mrae = 0
    def cal_ssim(self,img1,img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            ssims = []
            for i in range(img1.shape[0]):
                ssims.append(self.ssim(img1[i, :, :], img2[i, :, :]))  # 改
            return np.array(ssims).mean()
        else:
            raise ValueError('Wrong input image dimensions.')
    def ssim(self,img1,img2):

        C1 = (0.01 * self.num_range) ** 2
        C2 = (0.03 * self.num_range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    def cal_psnr(self,img1,img2):
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.psnr(img1, img2)
        elif img1.ndim == 3:
            sum_psnr = 0
            for i in range(img1.shape[0]):
                this_psnr = self.psnr(img1[i, :, :], img2[i, :, :])
                sum_psnr += this_psnr
        return sum_psnr / img1.shape[0]
    def psnr(self,img1,img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(self.num_range *1.0 / math.sqrt(mse))
    def cal_rmse_mrae_sam(self,img1,img2):# img1 result img2 label
        channel, height, width = img2.shape
        sum_se = 0
        sum_mrae = 0
        sum_sam = 0
        for i in range(0, height):
            for j in range(0, width):
                sum_se += np.sum((img1[:, i, j] - img2[:, i, j]) ** 2)
                A = img2[:, i, j]
                sum_mrae += np.sum(abs(img1[:, i, j] - img2[:, i, j]) / (img2[:, i, j] + 1))
                spe_res = img1[:, i, j].reshape(1, -1)
                spe_lab = img2[:, i, j].reshape(1, -1)
                sum_sam += math.acos(cosine_similarity(spe_lab, spe_res))

        rmse = (sum_se / (height * width * channel)) ** 0.5
        mrae = sum_mrae / (height * width * channel)
        sam = sum_sam / (height * width)
        return rmse,mrae,sam
    def SID(self,x, y):
        p = np.zeros_like(x, dtype=np.float)
        q = np.zeros_like(y, dtype=np.float)
        Sid = 0
        for i in range(len(x)):
            p[i] = x[i] / np.sum(x)
            q[i] = y[i] / np.sum(y)
        for j in range(len(x)):
            Sid += p[j] * np.log10(p[j] / q[j]) + q[j] * np.log10(q[j] / p[j])
        return Sid
    def SAM(self,x, y):# 计算SAM
        s = np.sum(np.dot(x, y))
        t = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
        th = np.arccos(s / t)
        # print(s,t)
        return th
    def calcul_one_img(self,result,label):

        mssim = self.cal_ssim(result,label)
        mpsnr = self.cal_psnr(result,label)
        rmse, mrae, sam = self.cal_rmse_mrae_sam(result,label)
        self.sum_mssim+=mssim
        self.sum_psnr+=mpsnr
        self.sum_rmse+=rmse
        self.sum_mrae+=mrae
        self.sum_sam+=sam
        self.cal_num+=1
        return rmse, mrae,sam,mssim,mpsnr
    def calcul_mean(self):
        return self.sum_rmse/self.cal_num,self.sum_mrae/self.cal_num,self.sum_sam/self.cal_num,self.sum_mssim/self.cal_num,self.sum_psnr/self.cal_num
