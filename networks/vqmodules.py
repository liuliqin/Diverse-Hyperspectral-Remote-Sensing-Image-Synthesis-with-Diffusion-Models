import torch
import torch.nn as nn
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), down_factor = 2,num_res_blocks = 2,
                 attn_resolutions=(16,8), dropout=0.0, resamp_with_conv=False, in_channels,
                 resolution,**ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.down_factor = down_factor


        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.out = nn.ModuleList()

        for layer_id in range(down_factor-1,self.num_resolutions)[::-1]:
            block_in = ch_mult[layer_id] * self.ch
            block_out = in_ch_mult[layer_id] * self.ch
            block=nn.ModuleList()
            block.append(ResnetBlock(in_channels = block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
            if (layer_id != self.num_resolutions-1) and (layer_id != down_factor-1):
                block.append(Upsample(in_channels=block_out,with_conv=resamp_with_conv))
            self.out.append(block)

        self.norm_out = Normalize(block_out)
        self.conv_out = torch.nn.Conv2d(block_out,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        hs = [h]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)
                hs.append(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        hs.append(h)
        # end
        for block in self.out:
            h = h+hs.pop()
            for sub_block in block:
                if isinstance(sub_block,ResnetBlock):
                    h = sub_block(h, temb)
                else:
                    h = sub_block(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels,bright,cond_channels=3,ch_mult=(1,2,4,8), down_factor = 2,
                 num_res_blocks= 2, dropout = 0.0, resamp_with_conv = False,
                 resolution, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = down_factor #len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        curr_res = resolution // 2**(down_factor) # 64*64
        self.z_shape = (1,in_channels,curr_res,curr_res)

        # z to block_in
        self.conv_cond = torch.nn.Conv2d(cond_channels,ch,kernel_size=3,stride=1,padding=1)

        self.cond_block = nn.ModuleList()
        for i_level in range(down_factor):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            block = nn.ModuleList()
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out

            block.append(Downsample(block_out, resamp_with_conv))
            self.cond_block.append(block)

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.input_block = nn.ModuleList()
        for i_level in range(down_factor):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            block = nn.ModuleList()
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out
            self.input_block.append(block)

        # middle
        # block_out = 2* block_out
        self.mid =nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=(block_out*2),
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_out)
        self.mid.block_2 = ResnetBlock(in_channels=block_out,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.out_block=nn.ModuleList()
        for i_level in reversed(range(down_factor)):
            block_in = ch * ch_mult[i_level]
            block_out = ch * in_ch_mult[i_level]
            block = nn.ModuleList()
            for i in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out
            if curr_res!=self.resolution:
                block.append(Upsample(block_out,resamp_with_conv))
                curr_res = curr_res * 2
            self.out_block.append(block)
        #
        #
        #
        # # upsampling
        # self.up = nn.ModuleList()
        # for i_level in reversed(range(self.num_resolutions)):# 1,0
        #     block = nn.ModuleList()
        #     attn = nn.ModuleList()
        #     block_out = ch*ch_mult[i_level]
        #     for i_block in range(self.num_res_blocks+1):
        #         block.append(ResnetBlock(in_channels=block_in,
        #                                  out_channels=block_out,
        #                                  temb_channels=self.temb_ch,
        #                                  dropout=dropout))
        #         block_in = block_out
        #         if curr_res in attn_resolutions:
        #             attn.append(AttnBlock(block_in))
        #     up = nn.Module()
        #     up.block = block
        #     up.attn = attn
        #     if i_level != 0:
        #         up.upsample = Upsample(block_in, resamp_with_conv)
        #         curr_res = curr_res * 2
        #     self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_out)
        self.conv_out = torch.nn.Conv2d(block_out,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.bright= nn.Parameter(bright,requires_grad=True)

    def forward(self, z,cond):
        # #assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        for block in self.input_block:
            for subblock in block:
                if isinstance(subblock,ResnetBlock):
                    h=subblock(h,temb)
                else:
                    h=subblock(h)
        h_z = h  # 64*64*2d

        h = self.conv_cond(cond)
        hs = [h]
        for block in self.cond_block:
            for subblock in block:
                if isinstance(subblock,ResnetBlock):
                    h=subblock(h,temb)
                else:
                    h=subblock(h)
            hs.append(h)
        h = torch.cat([h,h_z],1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for block in self.out_block:
            h = h + hs.pop()
            for subblock in block:
                if isinstance(subblock,ResnetBlock):
                    h=subblock(h,temb)
                else:
                    h=subblock(h)
        # middle
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h,F.relu(self.bright)


if __name__=="__main__":
    batch =2
    channel = 48
    H = W =256
    input = torch.randn(batch, channel, H, W).cuda()

    encoder = Encoder(ch=128, out_ch=8,in_channels=48,down_factor=2,resolution=256)
    encoder = encoder.cuda()
    out = encoder(input)
    cond_ch =3
    c = torch.randn(batch,cond_ch,H,W).cuda()
    decoder = Decoder(ch=128, out_ch=345,down_factor = 2,in_channels=8,resolution=256)
    decoder = decoder.cuda()
    pos,bright = decoder(out,c)

