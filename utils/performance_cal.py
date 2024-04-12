import os
import gdal
import argparse
import glob
from metric.calcul_metric import Cal
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import math
def read_tif(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        print("文件%s无法打开" % filename)
        exit(-1)
    im_data = dataset.ReadAsArray()
    return im_data
if __name__=='__main__':
    parser = argparse.ArgumentParser()#hyperspetral image generation SOTA\FMNetforSSR-master\TestLog\140
    #../../R2HGAN/mini_aviris_2202_results\20220624\test_results_120\test
    #hyperspetral image generation SOTA\HSRnet-main\TestLog\85
    parser.add_argument('--cal_path', default=r'../../hyperspetral image generation SOTA\pytoch-hscnn-plus-master\TestLog\80', help='cal path')
    parser.add_argument('--data_path',default=r'../data/mini_aviris_2202/test')
    opt = parser.parse_args()#../results/mini_aviris_2202\20220621 woband\test_results_201
    print(opt)

    data_path= opt.data_path
    real_path=os.path.join(data_path,'*.tif')#找到原图
    real_img_name=glob.glob(real_path)

    f=open(opt.cal_path+'/performance_new.txt','w')
    close_list =list(range(103)) + list(range(114, 151)) + list(range(168, 224))
    calculator = Cal(65535,list(range(196)))#close_list
    for img_name in real_img_name:
        f.write(img_name)
        label = read_tif(img_name).astype('float64')#n*h*w
        label = label[close_list,:,:]
        result_path = opt.cal_path+'/'+img_name.split('\\')[-1]
        result = read_tif(result_path).astype('float64')
        rmse, mrae, sam, mssim, mpsnr=calculator.calcul_one_img(result,label)

        #调用函数实现SSIM
        f.write('   rmse='+str(rmse)+'   mrae='+str(mrae)+'   sam='+str(sam)+'  mssim='+str(mssim) + '   mpsnr='+str(mpsnr))
        f.write('\n')
        mean_rmse, mean_mrae, mean_sam, mean_ssim, mean_psnr=calculator.calcul_mean()
        print(calculator.cal_num)

    f.write('图像平均 \n  rmse='+str(mean_rmse)+'  mrae='+str(mean_mrae)+'  sam='+str(mean_sam)+'  ssim='+str(mean_ssim)+'  psnr='+str(mean_psnr))
    f.write('\n')
    f.close()


