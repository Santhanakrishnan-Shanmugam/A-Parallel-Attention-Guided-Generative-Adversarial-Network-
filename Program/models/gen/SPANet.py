import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models.models_utils import weights_init, print_network
from src.attention_util import AttentionGate, ChannelGate, Flatten
# import common

###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

def newCovo2d(in_channel,out_channel):
    return nn.Conv2d(in_channels=64,out_channels=32,kernel_size = 1,
                    stride =1, padding=0,bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        
    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)


class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels,self.out_channels)
        self.relu1 = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask

###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet,self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(3,32),
            nn.ReLU(True)
            )
                
        self.SAM1 = SAM(32,32,1)
        self.res_block1 = Bottleneck(32,32)
        self.res_block2 = Bottleneck(32,32)
        self.res_block3 = Bottleneck(32,32)
        self.res_block4 = Bottleneck(32,32)
        self.res_block5 = Bottleneck(32,32)
        self.res_block6 = Bottleneck(32,32)
        self.res_block7 = Bottleneck(32,32)
        self.res_block8 = Bottleneck(32,32)
        self.res_block9 = Bottleneck(32,32)
        self.res_block10 = Bottleneck(32,32)
        self.res_block11 = Bottleneck(32,32)
        self.res_block12 = Bottleneck(32,32)
        self.res_block13 = Bottleneck(32,32)
        self.res_block14 = Bottleneck(32,32)
        self.res_block15 = Bottleneck(32,32)
        self.res_block16 = Bottleneck(32,32)
        self.res_block17 = Bottleneck(32,32)
        self.conv_out = nn.Sequential(
            conv3x3(32,3)
        )
        self.cbam1 = CBAM(gate_channels = 32, kernel_size = 3, reduction_ratio = 16, pool_types = ['avg', 'max'], no_spatial = False, bam = False, num_layers = 1, bn = False, dilation_conv_num = 2, dilation_val = 4)
        self.cord1=CoordAtt(inp=32, oup=32, reduction=32)
        
        self.convo1= nn.Sequential(
            newCovo2d(64,32),
            nn.ReLU(True)
            )
        #block 2
        self.cbam2 = CBAM(gate_channels = 32, kernel_size = 3, reduction_ratio = 16, pool_types = ['avg', 'max'], no_spatial = False, bam = False, num_layers = 1, bn = False, dilation_conv_num = 2, dilation_val = 4)
        self.cord2=CoordAtt(inp=32, oup=32, reduction=32)
        
        self.convo2= nn.Sequential(
            newCovo2d(64,32),
            nn.ReLU(True)
            )
         #block 3
        self.cbam3 = CBAM(gate_channels = 32, kernel_size = 3, reduction_ratio = 16, pool_types = ['avg', 'max'], no_spatial = False, bam = False, num_layers = 1, bn = False, dilation_conv_num = 2, dilation_val = 4)
        self.cord3=CoordAtt(inp=32, oup=32, reduction=32)
        
        self.convo3= nn.Sequential(
            newCovo2d(64,32),
            nn.ReLU(True)
            )
   

    def forward(self, x):

        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
        
        
        out_cbam1=self.cbam1(out)#output for cbam
        out_cord1=self.cord1(out)
        out_combaine1= torch.cat((out_cbam1, out_cord1), dim=1)#concatination
        outnewcovo1=self.convo1(out_combaine1)#newConvolution
        
        out_cbam2=self.cbam2(outnewcovo1)#output for cbam
        out_cord2=self.cord2(outnewcovo1)
        out_combaine2= torch.cat((out_cbam2, out_cord2), dim=1)#concatination
        outnewcovo2=self.convo2(out_combaine2)
        
        out_cbam3=self.cbam3(outnewcovo2)#output for cbam
        out_cord3=self.cord3(outnewcovo2)
        out_combaine3= torch.cat((out_cbam3, out_cord3), dim=1)#concatination
        outnewcovo3=self.convo3(out_combaine3)
        
        out=outnewcovo3+out
        
        
        Attention1 = self.SAM1(out) 
        out = F.relu(self.res_block4(out) * Attention1  + out)
        out = F.relu(self.res_block5(out) * Attention1  + out)
        out = F.relu(self.res_block6(out) * Attention1  + out)
        
        Attention2 = self.SAM1(out) 
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        
        Attention3 = self.SAM1(out) 
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        
        Attention4 = self.SAM1(out) 
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)
        
        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
       
        out = self.conv_out(out)
        

        return Attention4 , out

class Generator(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.gen = nn.Sequential(OrderedDict([('gen', SPANet())]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)


# ### Coordinate  starts here

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
         return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
         return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# ### Coordinate  ends here

class SpatialGate(nn.Module):
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            "gate_s_bn_reduce0", nn.BatchNorm2d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                "gate_s_bn_di_%d" % i, nn.BatchNorm2d(gate_channel // reduction_ratio)
            )
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.ReLU())
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)



class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        kernel_size=3,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
        bam=False,
        num_layers=1,
        bn=False,
        dilation_conv_num=2,
        dilation_val=4,
    ):
        super(CBAM, self).__init__()
        self.bam = bam
        self.no_spatial = no_spatial
        if self.bam:
            self.dilatedGate = SpatialGate(
                gate_channels, reduction_ratio, dilation_conv_num, dilation_val
            )
            self.ChannelGate = ChannelGate(
                gate_channels,
                reduction_ratio,
                pool_types,
                bam=self.bam,
                num_layers=num_layers,
                bn=bn,
            )
        else:
            self.ChannelGate = ChannelGate(
                gate_channels, reduction_ratio, pool_types
            )
            if not no_spatial:
                self.SpatialGate = AttentionGate(kernel_size)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.bam:
            if not self.no_spatial:
                x_out = self.SpatialGate(x_out)
            return x_out
        else:
            att = 1 + F.sigmoid(self.ChannelGate(x) * self.dilatedGate(x))
            return att * x
#CBAM Ends here


