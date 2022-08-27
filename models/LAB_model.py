import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
import random
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from PIL import ImageOps,Image
from torchgeometry.losses import SSIM
import numpy as np
import math
import kornia


class LABModel(BaseModel):
    def name(self):
        return 'LABNet Module'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='none')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G', 'perceptual', 'grad', 'l2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_img', 'final', 'shadow_mask', 'mask_disc', 'input_gt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.gpu_ids,channels=32)
        #image pixel value range
        self.range_img = (0, 1)
        self.loss_per_epoch = 0
        self.ratio = 1.0
        self.dilation_K = torch.ones(11, 11).cuda()
        self.down_w = opt.down_w
        self.down_h = opt.down_h

        if self.isTrain:
            self.criterionPerceptual = networks.PerceptualLoss().to(self.device)
            self.GradLoss = networks.GradientLoss().to(self.device)
            self.L2loss = torch.nn.MSELoss().to(self.device)
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, train=False):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.input_gt = input['C'].to(self.device)
        self.input_lab = input['A_lab'].to(self.device)
        self.gt_lab = input['C_lab'].to(self.device)        
        self.shadow_mask = (self.shadow_mask>0.5).type(torch.float)
        self.mask_d = kornia.morphology.dilation(self.shadow_mask, self.dilation_K)
  
    def forward(self):
        self.final = self.netG(self.input_lab, self.shadow_mask, self.mask_d, self.down_w, self.down_h)

    def backward_G(self):
        self.loss_perceptual = self.criterionPerceptual(self.final , self.gt_lab)
        self.loss_grad = self.GradLoss(self.final, self.gt_lab)
        self.loss_l2 = self.L2loss(self.final, self.gt_lab)
        self.loss_G = (self.loss_perceptual + 10 * self.loss_grad + 100 * self.loss_l2)
        self.loss_G.backward()
        self.loss_per_epoch += self.loss_G.item()

    def optimize_parameters(self):
        self.forward()
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def get_current_visuals(self):
        nim = self.input_img.shape[0]
        all =[]
        for i in range(0,min(nim,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:], range_img=self.range_img)
                        row.append(im)           
            row=tuple(row)
            row = np.hstack(row)
            all.append(row)      
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])  
    
    def get_prediction(self,input,is_origin=False):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.gt = input['C'].to(self.device)
        self.input_lab = input['A_lab'].to(self.device)
        self.gt_lab = input['C_lab'].to(self.device)    
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)
        self.mask_d = kornia.morphology.dilation(self.shadow_mask, self.dilation_K)
        self.forward()
        self.shadow_free = self.final
        RES = dict()
        if is_origin:
            RES['final'] = self.shadow_free
            RES['input'] = self.input_img
            RES['gt'] = input['C']
            RES['mask'] = self.shadow_mask

        else:
            RES['final']= util.tensor2im(self.shadow_free,scale=0, range_img=self.range_img)
            RES['input'] = util.tensor2im(self.input_img, scale=0, range_img=self.range_img)
            RES['gt'] = util.tensor2im(input['C'], scale=0, range_img=self.range_img)
            RES['mask'] = util.tensor2im(self.shadow_mask, scale=0, range_img=self.range_img)

        return  RES

    def reset_loss(self):
        self.loss_per_epoch = 0
