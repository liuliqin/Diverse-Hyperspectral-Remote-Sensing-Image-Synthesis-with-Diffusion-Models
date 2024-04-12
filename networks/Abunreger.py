import torch.nn as nn
import torch.nn.functional as F
from .weight_initial import normal_init

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class Imageencoder(nn.Module):
    def __init__(self,d = 128,input_nc = 48,output_nc = 8,ResBlock = ResBlock):
        super(Imageencoder, self).__init__()
        self.inchannel = d,
        self.input_layer = nn.Conv2d(input_nc, d, 3, 1, 1)
        self.layer_e1 = self.make_layer(ResBlock, d, 2, stride= 2)
        self.layer_e1_bn = nn.BatchNorm2d(d)
        self.layer_e2 = self.make_layer(ResBlock, d * 2, 2, stride= 2)
        self.layer_e2_bn = nn.BatchNorm2d(2 * d)
        self.layer_e3 = self.make_layer(ResBlock, d * 4, 2, stride= 2)
        self.layer_e3_bn = nn.BatchNorm2d(4 * d)
        self.layer_e4 = self.make_layer(ResBlock, d * 8, 2, stride= 2)

        self.layer_d4 = nn.Conv2d(d * 8, d * 4, 3, 1, 1)
        self.layer_d4_bn = nn.BatchNorm2d(d * 4)
        self.layer_d3 = nn.Conv2d(d * 4, d * 2, 3, 1, 1)
        self.layer_d3_bn = nn.BatchNorm2d(d * 2)
        self.layer_d2 = nn.Conv2d(d * 2, d ,3, 1, 1)
        self.out_layer = nn.Conv2d(d, output_nc, 3, 1, 1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self,input):
        encode1 = self.input_layer(input)
        encode2 = F.leaky_relu(self.layer_e1_bn(self.layer_e1(encode1, 0.2)))
        encode3 = F.leaky_relu(self.layer_e2_bn(self.layer_e2(encode2, 0.2)))
        encode4 = F.leaky_relu(self.layer_e3_bn(self.layer_e3(encode3, 0.2)))
        encode5 = F.leaky_relu(self.layer_e4_bn(self.layer_e4(encode4, 0.2)))
        up4 = F.interpolate(encode5, scale_factor=2, mode='bilinear', align_corners=False)
        decode4 = F.leaky_relu(self.layer_d4_bn(self.layer_d4(up4+encode4)),0.2)
        up3 = F.interpolate(decode4, scale_factor=2, mode='bilinear', align_corners=False)
        decode3 = F.leaky_relu(self.layer_d3_bn(self.layer_d3(up3+encode3)),0.2)
        decode2 = F.leaky_relu(self.layer_d2(decode3),0.2)
        out = self.out_layer(decode2)
        return out

class Abundecoder(nn.Module):
    # initializers
    def __init__(self,bright,d=128,input_nc=8,output_nc=345,ResBlock=ResBlock):
        super(Abundecoder, self).__init__()
        self.inchannel=d
        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc,d,3,1,1)
        self.layer1 = self.make_layer(ResBlock, d, 2, stride=2)
        self.layer1_bn = nn.BatchNorm2d(d)
        self.layer2 = self.make_layer(ResBlock, d * 2, 2, stride=1)
        self.layer2_bn = nn.BatchNorm2d(2 * d)
        self.layer3 = self.make_layer(ResBlock, d * 4, 2, stride=1)
        self.layer3_bn = nn.BatchNorm2d(4 * d)
        self.layer4 = self.make_layer(ResBlock, d * 2, 2, stride=2)
        self.layer4_bn = nn.BatchNorm2d(2 * d)
        self.layer5 = self.make_layer(ResBlock, d * 1, 2, stride=1)
        self.layer5_bn = nn.BatchNorm2d(1 * d)
        self.layer6 = self.make_layer(ResBlock, 8, 2, stride=1)



        # saa= np.random.rand(224)*5.
        # c=torch.tensor(saa).float()
        # self.bright= nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(c,0),2),3).cuda(),requires_grad=True)
        self.bright= nn.Parameter(bright,requires_grad=True)

        # Unet decoder
        self.deconv6 = nn.Conv2d(d * 8, d * 8, 3, 1, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.Conv2d(d * 8, d * 8, 3, 1, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.Conv2d(d * 8, d * 4, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.Conv2d(d * 4, d * 2, 3, 1, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.Conv2d(d * 2, d, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(d)
        self.deconv1 = nn.Conv2d(d, d, 3, 1, 1)
        self.deconv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, output_nc, kernel_size=3, stride=1, padding=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        encode1 = self.conv1(input)
        encode2 = self.layer1_bn(self.layer1(F.leaky_relu(encode1, 0.2)))
        encode3 = self.layer2_bn(self.layer2(F.leaky_relu(encode2, 0.2)))
        encode4 = self.layer3_bn(self.layer3(F.leaky_relu(encode3, 0.2)))
        encode5 = self.layer4_bn(self.layer4(F.leaky_relu(encode4, 0.2)))
        encode6 = self.layer5_bn(self.layer5(F.leaky_relu(encode5, 0.2)))
        encode7 = self.layer6(F.leaky_relu(encode6, 0.2))

        up7=F.interpolate(F.leaky_relu(encode7, 0.2),scale_factor=2,mode='bilinear',align_corners=False)
        decode6 = F.leaky_relu(self.deconv6_bn(self.deconv6(up7)),0.2)
        up6=F.interpolate(encode6 + decode6,scale_factor=2,mode='bilinear',align_corners=False)
        decode5 = F.leaky_relu(self.deconv5_bn(self.deconv5(up6)),0.2)
        up5 = F.interpolate(encode5 + decode5, scale_factor=2, mode='bilinear',align_corners=False)
        decode4 = F.leaky_relu(self.deconv4_bn(self.deconv4(up5)),0.2)
        up4 = F.interpolate(encode4 + decode4, scale_factor=2, mode='bilinear',align_corners=False)
        decode3 = F.leaky_relu(self.deconv3_bn(self.deconv3(up4)), 0.2)
        up3 = F.interpolate(encode3 + decode3, scale_factor=2, mode='bilinear',align_corners=False)
        decode2= F.leaky_relu(self.deconv2_bn(self.deconv2(up3)), 0.2)
        up2 = F.interpolate(encode2 + decode2, scale_factor=2, mode='bilinear',align_corners=False)
        decode1 = F.leaky_relu(self.deconv1_bn(self.deconv1(up2)), 0.2)
        pos = self.conv2(F.leaky_relu(encode1 + decode1, 0.2))

        # abun = F.softmax(pos)
        # feature_name = eval('decode'+str(self.feature_id))

        return pos,F.relu(self.bright) #,feature_name