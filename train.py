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
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


opt = TrainOptions().parse()
writer = SummaryWriter(log_dir=opt.name)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)
model.setup(opt)
#model.register_mask_hook()
visualizer = Visualizer(opt)

total_steps = 0
best_loss = np.inf


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    model.epoch = epoch
    torch.cuda.empty_cache()

    model.train()
    for i, data in enumerate(tqdm(dataset)):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.cepoch=epoch
        model.optimize_parameters()

        ##############Visualization block
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch)
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size

            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, opt, errors)
        ###################################
        writer.add_scalar('percep loss', model.loss_perceptual, epoch*(dataset_size//5)+ i)
        writer.add_scalar('grad loss', model.loss_grad, epoch*(dataset_size//5)+ i)
        writer.add_scalar('l2 loss', model.loss_l2, epoch*(dataset_size//5)+ i)
        #writer.add_scalar('mask loss', model.mask_loss, epoch*(dataset_size//5)+ i)

    if epoch %  5 == 0:
        model.save_networks('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)
    
    if model.loss_per_epoch < best_loss:
        print('Update model at epoch {} from {:.2f} to {:.2f}'.format(epoch, best_loss/len(dataset)*5, model.loss_per_epoch/len(dataset)*5))
        model.save_networks('best')
        best_loss = model.loss_per_epoch

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
    #model.update_cons_coef(opt.niter, opt.niter + opt.niter_decay)
    model.reset_loss()
