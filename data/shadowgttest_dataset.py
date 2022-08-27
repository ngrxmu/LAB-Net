import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageChops
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
import cv2 as cv


class ShadowgttestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        print(self.dir_B)
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        print(self.dir_A)
        self.A_paths,self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=opt.norm_mean,
                                               std = opt.norm_std)]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])
     
    def __getitem__(self,index):
        birdy = {}
        A_path = self.A_paths[index % self.A_size]
        imname = self.imname[index % self.A_size]

        
        B_path = os.path.join(self.dir_B,imname.replace('.jpg','.png'))
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B,imname)
        A_img = Image.open(A_path).convert('RGB')
        A_lab = Image.fromarray(cv.cvtColor(np.array(A_img), cv.COLOR_RGB2LAB))
    
        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
        if os.path.isfile(B_path): 
            B_img = Image.open(B_path)
        else:
            print('MASK NOT FOUND : %s'%(B_path))
            B_img = Image.fromarray(np.zeros((int(w),int(h)),dtype = np.float),mode='L')
        
        C_img = Image.open(os.path.join(self.dir_C,imname)).convert('RGB')
        C_lab = Image.fromarray(cv.cvtColor(np.array(C_img), cv.COLOR_RGB2LAB))
        birdy['C'] = C_img
        birdy['C_lab'] = C_lab  
       
        loadSize = self.opt.loadSize

        neww = self.opt.size_w
        newh = self.opt.size_h
        
        birdy['A'] = A_img
        birdy['B'] = B_img
        birdy['A_lab'] = A_lab
        

        for k,im in birdy.items():
            birdy[k] = im.resize((neww, newh),Image.NEAREST)
        
        w = birdy['A'].size[0]
        h = birdy['A'].size[1]
        for i in ['A', 'C', 'A_lab', 'C_lab']:
            birdy[i] = self.transformA(birdy[i])
        birdy['B'] = self.transformB(birdy['B'])
        birdy['imname'] = imname
        birdy['w'] = ow
        birdy['h'] = oh
        birdy['A_paths'] = A_path
        birdy['B_baths'] = B_path
        return birdy

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'ShadowgttestDataset'
