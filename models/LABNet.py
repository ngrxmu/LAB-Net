import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import time
import kornia
import numpy as np


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True, groups = 1, norm = 'in', nonlinear = 'leakyrelu'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1)) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups = groups, bias = bias, dilation = dilation)
        self.norm = norm
        self.nonlinear = nonlinear
        
        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine = False)
        else:
            self.normalization = None
          
        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = None
          
    def forward(self, x): 
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
            out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, k, domain=4, nonlinear = 'leakyrelu'):
      super(ChannelAttention, self).__init__()
      self.channels = channels
      self.k = k
      self.nonlinear = nonlinear
      
      self.linear1 = nn.Linear(channels, channels//k)
      self.linear2 = nn.Linear(channels//k, channels)

      if domain == 8:
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        kernel_value = np.array([kernel] * channels, dtype='float32')
      else:
        kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        kernel_value = np.array([kernel] * channels, dtype='float32')

      kernel_value = kernel_value.reshape((channels, 1, 3, 3))
      self.convop = nn.Conv2d(channels, channels, 3, bias=False, groups=channels)
      self.convop.weight.data = torch.from_numpy(kernel_value)
      self.convop.weight.data.requires_grad = False

      if nonlinear == 'relu':
          self.activation = nn.ReLU(inplace = True)
      elif nonlinear == 'leakyrelu':
          self.activation = nn.LeakyReLU(0.2)
      elif nonlinear == 'PReLU':
          self.activation = nn.PReLU()
      else:
          raise ValueError
      
    def attention(self, x):
      N, C, H, W = x.size()
      feature = self.convop(x)
      std = feature.std(dim = (2, 3))
      out = self.activation(self.linear1(std))
      out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)
      
      return out.mul(x)
      
    def forward(self, x):
      return self.attention(x)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(SpatialAttentionBlock, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.conv_s = ConvLayer(in_channels, channels * 3 // 2, 3, 1, 1, norm='None')
        self.conv_ns = ConvLayer(in_channels, channels * 3 // 2, 3, 1, 1, norm='None')
        self.convfinal = ConvLayer(channels * 5 // 2, channels, 1, 1, 1, norm='None')

    def forward(self, input, mask, mask_d, down_w, down_h):
        size = input.size()
        input_t = torch.nn.functional.interpolate(input, (down_w, down_h))
        mask = torch.nn.functional.interpolate(mask, (down_w, down_h))
        mask_d = torch.nn.functional.interpolate(mask_d, (down_w, down_h))
        q_s = (self.conv_s(input_t)).reshape((size[0], self.channels * 3 // 2, -1))
        k_ns = (self.conv_ns(input_t)).reshape((size[0], self.channels * 3 // 2, -1))

        for i in range(size[0]):
            CMat = F.softmax(torch.mm(q_s[i][(mask[i].reshape((1, -1)).expand_as(q_s[i]) == 1)].reshape((self.channels * 3 // 2, -1)).T, k_ns[i][(mask_d[i]-mask[i]).reshape((1, -1)).expand_as(k_ns[i]) == 1].reshape((self.channels * 3 // 2, -1))) / torch.sqrt(torch.tensor(self.channels * 3 // 2)), dim=1)
            k_ns[i][mask[i].reshape((1, -1)).expand_as(k_ns[i]) == 1] = torch.mm(CMat, k_ns[i][(mask_d[i]-mask[i]).reshape((1, -1)).expand_as(k_ns[i]) == 1].reshape((self.channels * 3 // 2, -1)).T).permute(1, 0).reshape((-1))

        k_ns = k_ns.reshape((size[0], self.channels * 3 // 2, down_w, down_h))
        k_ns = torch.nn.functional.interpolate(k_ns, (size[2], size[3]))
        output = self.convfinal(torch.cat([input, k_ns], dim=1))

        return output


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super(BasicBlock, self).__init__()

        self.conv11 = ConvLayer(in_channels, channels // 2, 3, 1, 1, norm='None')
        self.conv12 = ConvLayer(in_channels, channels, 3, 1, 4, norm='None')
        self.conv13 = ConvLayer(in_channels, channels * 3 // 2, 3, 1, 16, norm='None')
        self.conv14 = ConvLayer(channels*3, channels, 1, 1, 1, norm='None')

        self.conv21 = ConvLayer(channels, channels // 2, 3, 1, 2, norm='None')
        self.conv22 = ConvLayer(channels, channels, 3, 1, 8, norm='None')
        self.conv23 = ConvLayer(channels, channels * 3 // 2, 3, 1, 32, norm='None')
        self.conv24 = ConvLayer(channels*3, channels, 1, 1, 1, norm='None')

        self.conv31 = ConvLayer(channels, channels // 2, 3, 1, 4, norm='None')
        self.conv32 = ConvLayer(channels, channels, 3, 1, 16, norm='None')
        self.conv33 = ConvLayer(channels, channels * 3 // 2, 3, 1, 64, norm='None')
        self.conv34 = ConvLayer(channels*3, channels, 1, 1, 1, norm='None')

        self.conv41 = ConvLayer(channels*3, channels, 1, 1, 1, norm='None')
        self.conv42 = ConvLayer(channels, out_channels, 3, 1, 1, norm='None')
        self.channel_attention = ChannelAttention(channels * 3, 8, domain=4)

    def forward(self, input):
        x = self.conv14(torch.cat([self.conv11(input), self.conv12(input), self.conv13(input)], dim=1))

        y = self.conv24(torch.cat([self.conv21(x), self.conv22(x), self.conv23(x)], dim=1))

        z = self.conv34(torch.cat([self.conv31(y), self.conv32(y), self.conv33(y)], dim=1))

        out_fea = self.channel_attention(torch.cat([x, y, z], dim=1))
        output = self.conv42(self.conv41(out_fea)) 

        return output

class LABNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super(LABNet, self).__init__()

        self.convst = ConvLayer(in_channels, out_channels, 3, 1, 1, norm='None')

        self.basicblock1 = BasicBlock(out_channels+1, channels, channels)
        self.basicblock2 = BasicBlock(out_channels+1, channels, channels)
        self.conv12_l = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.basicblock3 = BasicBlock(channels, channels, channels)
        self.conv12_ab = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.basicblock4 = BasicBlock(channels, channels, channels)
        self.conv34_l = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.spaatt1 = SpatialAttentionBlock(channels, channels)
        self.basicblock5 = BasicBlock(channels, channels, channels)
        self.conv34_ab = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.spaatt2 = SpatialAttentionBlock(channels, channels)
        self.basicblock6 = BasicBlock(channels, channels, channels)
        self.conv56_l = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.spaatt3 = SpatialAttentionBlock(channels, channels)
        self.basicblock7 = BasicBlock(channels, 1, channels)
        self.conv56_ab = ConvLayer(channels * 2, channels, 1, 1, 1, norm='None')
        self.spaatt4 = SpatialAttentionBlock(channels, channels)
        self.basicblock8 = BasicBlock(channels, 2, channels)
        self.convfinal = ConvLayer(3, 3, 3, 1, 1, norm='None')

    def forward(self, input_lab, maskgt, mask_d, down_w, down_h):

        image = self.convst(input_lab)

        image1 = self.basicblock1(torch.cat([image, maskgt], dim=1))
        image2 = self.basicblock2(torch.cat([image, maskgt], dim=1))
        image12_l = self.conv12_l(torch.cat([image1, image2], dim=1))
        image12_ab = self.conv12_ab(torch.cat([image1, image2], dim=1))
        image3 = self.basicblock3(image12_l)
        image4 = self.basicblock4(image12_ab)
        image34_l = self.conv34_l(torch.cat([image3, image4], dim=1))
        image34_l = self.spaatt1(image34_l, maskgt, mask_d, down_w, down_h)
        image34_ab = self.conv34_ab(torch.cat([image3, image4], dim=1))
        image34_ab = self.spaatt2(image34_ab, maskgt, mask_d, down_w, down_h)
        image5 = self.basicblock5(image34_l)
        image6 = self.basicblock6(image34_ab)
        image56_l = self.conv56_l(torch.cat([image5, image6], dim=1))
        image56_l = self.spaatt3(image56_l, maskgt, mask_d, down_w, down_h)
        image56_ab = self.conv56_ab(torch.cat([image5, image6], dim=1))
        image56_ab = self.spaatt4(image56_ab, maskgt, mask_d, down_w, down_h)
        image7 = self.basicblock7(image56_l)
        image8 = self.basicblock8(image56_ab)
        out = self.convfinal(torch.cat([image7, image8], dim=1)+input_lab)

        return out
