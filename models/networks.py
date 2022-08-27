import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .LABNet import *

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'shadow_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70000,90000,13200], gamma=0.3)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init_weight=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_weight:
        init_weights(net, init_type, gain=init_gain)
    return net

def define_G(netG,init_type='normal', init_gain=0.02, gpu_ids=[],channels=32):
    net = None
    init_weight = True

    if netG == 'LABNet':
        net = LABNet(in_channels = 3, out_channels = 3, channels = channels)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, init_weight=init_weight)

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:3])
        self.enc_2 = nn.Sequential(*vgg16.features[3:8])
        self.enc_3 = nn.Sequential(*vgg16.features[8:13])
        self.enc_4 = nn.Sequential(*vgg16.features[13:20])
        self.enc_5 = nn.Sequential(*vgg16.features[20:27])

        # fix the encoder
        for i in range(5):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.extrator = VGG16FeatureExtractor()
        self.coef = [1.0, 2.6, 4.8, 3.7, 5.6, 0.15]
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        x_out = self.extrator(x)
        y_out = self.extrator(y)
        loss = 0
        for i in range(len(x_out)):
            loss += self.l1(x_out[i], y_out[i])/self.coef[i]
        return loss

class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target):
        _, cin, _, _ = pred.shape
        _, cout, _, _ = target.shape
        assert cin == 3 and cout == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)
        kx = kx.repeat((cout, 1, 1, 1))
        ky = ky.repeat((cout, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        loss = (
            nn.L1Loss(reduction=self.reduction)
            (pred_grad_x, target_grad_x) +
            nn.L1Loss(reduction=self.reduction)
            (pred_grad_y, target_grad_y))
        return loss * self.loss_weight
