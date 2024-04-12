The code for a diverse hyperspectral remote sensing image generation method with diffusion models for paper "Diverse Hyperspectral Remote Sensing Image Synthesis With Diffusion Models". 

Please cite the following paper:
@article{liu2023diverse,
  title={Diverse Hyperspectral Remote Sensing Image Synthesis With Diffusion Models},
  author={Liu, Liqin and Chen, Bowen and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--16},
  year={2023},
  publisher={IEEE}
}

The method consists of two stages: one is the conditional VQGAN and the other is the diffusion process in latent space.

Conduct the diverse hyperspectral generation by running generation_infer.py, with the options setting in options/base_options.py and options/infer_options.py.

When you need to train the conditional VQGAN, python generation_train.py, with the options setting in options/base_options.py and options/train_options.py.

After the training of the conditional VQGAN, python generation_test.py, with options/test_options.py, setting the 'test_flag' as 'train_mini' and 'test_mini' respectively, to get the latent code of each image for training process of the latent diffusion model.

When training the latent diffusion model, python image_train.py.

When testing the latent code generation results, python image_samples.py.

The conditional VQGAN model is saved in results/20230407_latent16, the diffusion model is saved in results/20230517_atten. The model can be downloaded with the link:
https://pan.baidu.com/s/1jTqSYe09iP3u3BoaCD6NpA?pwd=qktb 

The generation results are saved in results/20230517_atten/004000, and different results of multi-times running is saved in 004000_0,004000_1,...,004000_9, etc.

The grss_dfc_2018 dataset can be downloaded with the link:
https://pan.baidu.com/s/1q3CVQgeQaUxZlLZgOVHYsA?pwd=qu4x 




