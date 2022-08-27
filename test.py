from collections import OrderedDict
from options.train_options import TrainOptions
from options.test_options import  TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import visdom
from util.util import sdmkdir
from util import util
import time
import os
from skimage.color import rgb2lab
import numpy as np
import torch
import cv2 as cv
import tqdm


test_opt = TestOptions().parse()
model = create_model(test_opt)
model.setup(test_opt)

test_data_loader = CreateDataLoader(test_opt)
test_set = test_data_loader.load_data()
test_save_path = os.path.join(test_opt.checkpoints_dir, test_opt.resroot)

if not os.path.isdir(test_save_path):
    os.makedirs(test_save_path)

model.eval()
idx = 0

for i, data in enumerate(test_set):
  with torch.no_grad():
    idx += 1
    visuals = model.get_prediction(data, is_origin=True)
    im_name = data['imname'][0].split('.')[0]

    pred = cv.cvtColor(np.array(util.tensor2im(visuals['final'], scale=0)), cv.COLOR_LAB2RGB)
    mask = util.tensor2im(visuals['mask'], scale=0)
    gt = util.tensor2im(visuals['gt'], scale=0)
    ori = util.tensor2im(visuals['input'], scale=0)

    util.save_image(pred, os.path.join(test_save_path, im_name+'pred.png'))
    print(idx)

print('test end!')

