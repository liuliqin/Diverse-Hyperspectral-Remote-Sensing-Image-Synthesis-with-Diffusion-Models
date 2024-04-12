import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.quantize import VectorQuantizer2 as VectorQuantizer
from networks.vqmodules import Encoder,Decoder
from interface.generation_loss import GenerationLoss
import h5py
import scipy.io as scio
from torch.autograd import Variable

class VQModel(nn.Module):
    def __init__(self,bright,spe_lib,
                 embed_dim = 256, embed_num =1024, d=128,z_channel=8,abun_dim=345,
                 hyper_channel=48,down_factor=2,resolution=256):
        super(VQModel, self).__init__()
        self.bright = bright
        self.spe_lib = spe_lib

        self.encoder = Encoder(ch=d, out_ch=z_channel,in_channels=hyper_channel,down_factor=down_factor,resolution=resolution)

        self.decoder = Decoder(ch=d, out_ch=abun_dim,in_channels=z_channel,bright=bright,down_factor = down_factor,resolution=resolution)

        self.quantizer = VectorQuantizer(embed_num, embed_dim, beta=0.25)

        self.quant_conv = nn.Conv2d(z_channel,embed_dim,1)
        self.post_quant_conv = nn.Conv2d(embed_dim,z_channel,1)
    def _hyper_from_abun(self,pos,bright):
        abun = F.softmax(pos,dim =1)
        [b,_,h,w]=abun.size()
        [c,_] = bright.size()
        hyper = (bright * torch.matmul(self.spe_lib,
                                   torch.reshape(abun.permute(1, 0, 2, 3), (abun.shape[1], -1)))).reshape(c,b,h,w).permute(1, 0, 2, 3)
        return hyper

    def encode(self,x):
        h =self.encoder(x)
        h=self.quant_conv(h)
        quant, emb_loss,info =self.quantizer(h)
        return quant,emb_loss,info
    def decode(self,quant,cond):
        quant = self.post_quant_conv(quant)
        pos,bright = self.decoder(quant,cond)
        dec = self._hyper_from_abun(pos,bright)
        return dec,F.softmax(pos,dim =1)
    # def decode_code(self,code_b):
    #     quant_b = self.quantizer.embed_code()
    def forward(self,input,cond):
        quant,emd_loss,_ = self.encode(input)
        dec,_= self.decode(quant,cond)
        return dec,emd_loss
    # def zero_grad(self):
    #     self.encoder.zero_grad()
    #     self.decoder.zero_grad()
    #     self.post_quant_conv.zero_grad()
    #     self.quant_conv.zero_grad()
    #     self.quantizer.zero_grad()
    # # def training_loss(self,real,dec,emd_loss):
    # def train_one_epoch(self,train_loader):
    #     num_iter = 0
    #     for hyper_,multi_ in train_loader:
    #         hyper_,multi_=Variable(hyper_,multi_)
    #         self.zero_grad()
    #         dec_hyper, emd_loss =self.forward(hyper_,multi_)




    # def train_loop(self,train_loader,start_epoch,train_epoch,test_loader = None):
    #     print('training start!')
    #     for epoch in range(start_epoch+1,train_epoch):
    #         self.train_one_epoch(train_loader)

if __name__=="__main__":
    data = h5py.File('../spe_grss_345.mat')
    spe_lib = torch.tensor(data['spe_grss']).float().cuda()
    bright = torch.tensor(scio.loadmat('../bright.mat')['bright']).float().cuda().t()
    vqmodel = VQModel(bright,spe_lib,embed_dim = 256, embed_num =1024, d=128,z_channel=8,abun_dim=345,
                 hyper_channel=48,down_factor=2,resolution=256).cuda()
    batch = 2
    channel = 48
    H = W = 256
    input = torch.randn(batch, channel, H, W).cuda()
    cond = torch.randn(batch, 3, H, W).cuda()
    dec = vqmodel(input,cond)
