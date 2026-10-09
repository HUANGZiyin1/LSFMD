import torch
import torch.nn as nn
import torch.nn.functional as F
from octconv import *
import common

# ==========
# Spatio-temporal deformable fusion module
# ==========



class HAWT(nn.Module):
    def __init__(self, in_channel=3, in_channel_long = 5,nf=48, nb=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(HAWT, self).__init__()
        # u-shape backbone
        self.conv_local= nn.Sequential(
            nn.Conv2d(in_channel, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
        )

        self. MSRB1 = MSRB(nf)
        self. MSRB2 = MSRB(nf)
        self. MSRB3 = MSRB(nf)

        self.conv_global= nn.Sequential(
            nn.Conv3d(1, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ResN1 = ResBlock(nf)
        self.ResN2 = ResBlock(nf)
        self.ResN3 = ResBlock(nf)

        self.conv_trans = nn.MaxPool3d((in_channel_long, 1, 1), stride=1, padding=0)
        self.conv_trans1= nn.MaxPool3d((in_channel_long,1,1),stride=1,padding=0)
        self.conv_trans2 = nn.MaxPool3d((in_channel_long, 1, 1), stride=1, padding=0)
        self.conv_trans3 = nn.MaxPool3d((in_channel_long, 1, 1), stride=1, padding=0)

        self.conv = nn.Sequential(
            nn.Conv2d(nf*2, nf, 1, padding=1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 5, padding=5//2),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(nf*2, nf, 1, padding=1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf*2, nf, 1, padding=1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )

        self.LA= CALayer(nf,16)
        self.LA1= CALayer(nf,16)
        self.LA2= CALayer(nf,16)
        self.GA= CALayer(nf*3,16)
        self.sig = nn.Sigmoid()
        self.fusion= nn.Sequential(
            nn.Conv2d(nf*3, nf, 1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.bott= nn.Sequential(
            nn.Conv2d(nf*3, nf, 1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.qenet = QE(nf,nf)


    def forward(self, inputs, input_long, TFrm):

        input3d = input_long.unsqueeze(1)
        # Feature Extraction

        out_local = self.conv_local(inputs)#1,32,128,128
        out_1 = self.MSRB1(out_local)#1,32,128,128
        out_2 = self.MSRB2(out_1)#1,32,128,128
        out_3 = self.MSRB3(out_2)#1,32,128,128

        out_global = self.conv_global(input3d) # 1,32,3,128,128
        og = self.conv_trans(out_global)
        og = og.squeeze(2)
        olg = self.conv(torch.cat([out_local,og],1))
        la_olg = self.LA(olg)
        out_g1 = self.ResN1(out_global)  # 1,32,5,128,128
        out_s1 = self.conv_trans1(out_g1)  #1, 32, 1, 128, 128
        out_s1 = out_s1.squeeze(2)     #1,32,128,128
        olg1 = self.conv1(torch.cat([out_1,out_s1],1))
        la_olg1 = self.LA1(olg1)
        out_g2 = self.ResN2(out_g1)      #1, 32, 3, 128, 128
        out_s2 = self.conv_trans2(out_g2)  #1, 32, 1, 128, 128
        out_s2 = out_s2.squeeze(2)     #1,32,128,128
        olg2 = self.conv2(torch.cat([out_2,out_s2],1))
        la_olg2 = self.LA2(olg2)

        olc = torch.cat([la_olg,la_olg1,la_olg2],1)
        oc = self.sig(self.GA(torch.cat([olg,olg1,olg2],1)))
        oc = oc*olc
        oc = self.fusion(oc)
        out_g3 = self.ResN3(out_g2)
        out_g3 = self.conv_trans3(out_g3) #1,32,1,128,128
        out_g3 = out_g3.squeeze(2) #1,32,128,128
        o = torch.cat([out_3,out_g3,oc],1)#[1, 256, 128, 128]
        o = self.bott(o) #1,32,128,128
        o = self.qenet(o,TFrm) #1,1,128,128
        o = TFrm+o

        return o

class HFB(nn.Module):
    def __init__(self, ins,nf,alpha):
        super(HFB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )
        self.c1 = Conv_ACT(ins, nf*2, 3,padding=1, alpha_in=alpha)
        self.conv_ot = nn.Sequential(
            nn.Conv2d(nf*2, nf, 1, padding=1//2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, TFrm):
        res = self.conv_bt(x)
        x1_h, x1_l = self.c1(TFrm)
        out = torch.cat([res,x1_h],1)
        out = self.conv_ot(out)
        out = x+out
        return out,x1_h

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self,):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 3, padding=3 // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 5, padding=5 // 2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 1, padding=1 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_temp = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_sum = torch.cat([x1,x2],1)
        x_sum = self.conv3(x_sum)
        out = self.sigmoid(x_sum)
        out = out*x_temp
        return out

class QE(nn.Module):
    def __init__(self, ins,n_feats):
        super(QE, self).__init__()
        self.swin_layer1 = nn.Sequential(
            nn.Conv2d(ins, n_feats, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )
        self.ada_cnn1 =Conv_ACT(1, n_feats*2, 3,padding=1, alpha_in=0)
        self.swin_layer2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )
        self.ada_cnn2 = Conv_ACT(n_feats*2, n_feats*2, 3,padding=1)


        self.swin_layer4 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=3//2),
            nn.ReLU(inplace=True),
        )
        self.ada_cnn4 = Conv_ACT(n_feats*2, n_feats, 3,padding=1, alpha_out=0)

        self.fusion = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, 1, 3, padding=3//2),
        )

    def forward(self, x, Tfrm):
        x = self.swin_layer1(x)
        x1_h, x1_l = self.ada_cnn1(Tfrm)
        out = x+x1_h

        out = self.swin_layer2(out)
        x2_h, x2_l = self.ada_cnn2((x1_h, x1_l))
        out = out+x2_h


        out = self.swin_layer4(out)
        x4_h, x4_l = self.ada_cnn4((x2_h,x2_l))
        out = torch.cat([out,x4_h],1)
        out = self.fusion(out)
        return out

class RSB(nn.Module):
    def __init__(self, channel):
        super(RSB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=3 // 2)
        )
        self.SA = SpatialAttention()

    def forward(self, x):
        res = self.conv_bt(x)
        res = self.SA(res)
        res = x+res
        return res

# ==========
# Quality enhancement module
# ==========
class unet(nn.Module):
    def __init__(self, in_nc, nf, nb, base_ks=3):
        super(unet,self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, inputs):
        nb = self.nb
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel//2, 1, padding=1//2),
            nn.ReLU(inplace=True),
        )
        self.CAlayer = CALayer(channel//2,16)

    def forward(self, x):
        res = self.conv_bt(x)
        res = self.CAlayer(res)
        return res

class MSRB(nn.Module):
    def __init__(self, nf, k1 = 3, k2 = 5):
        super(MSRB, self).__init__()

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k1, padding=k1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*2, k1, padding=k1//2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(nf, nf, k2, padding=k2//2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*2, k2, padding=k2//2),
            nn.ReLU(inplace=True),
        )

        self.confusion = nn.Conv2d(nf * 4, nf, 1, padding=0, stride=1)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(input_1)
        output_5_1 = self.conv_5_1(input_1)
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv3d(channel, channel, 3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 3, padding=3 // 2),
        )

    def forward(self, x):
        res = self.conv_bt(x)
        res = res+x
        return res



if __name__ == '__main__':
    model = HAWT()
    net_input = torch.randn(1, 3, 128, 128)
    net_input_long = torch.randn(1, 5, 128, 128)
    TFrm = torch.randn(1, 1, 128, 128)
    net_output= model(net_input,net_input_long,TFrm)
    print(net_output.shape)