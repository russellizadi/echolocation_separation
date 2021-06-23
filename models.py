import os
import math
import collections
import time
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage
import skimage.segmentation
import skimage.morphology
import skimage.measure
import utils as ut
import scipy


def separate(x, args):
    """
    The main function for separation
    :param x: input
    :param args: config arguments
    :returns y: the separated sources
    """

    # path model file and figures folder
    path_pth = args.path_pth
    path_figures = args.path_figures
    
    # directories
    ut.make_dir(args.path_sources_)
    ut.make_dir(args.path_figures)
    
    # train
    if not os.path.isfile(path_pth):
        # loss folder
        args.path_loss = args.path_model
        ut.make_dir(args.path_loss)
        
        # initialize model 
        model = DenseUnet(args).to(args.device)
        args.logger.info("model {} is born".format(args.name_model))
        
        # info: number of params
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args.logger.info("number of model's parameters: {}".format(num_params))
        
        # info: model layers
        args.logger.info("model modules:")
        for name, module in model.named_modules():
            args.logger.info(name)
        
        # list train and test files
        lst_path_mixtures = ut.lst_path_endswith(args.path_mixtures, '.pkl')
        lst_path_train, lst_path_valid = ut.fold(lst_path_mixtures, args)
        
        # loader train
        args.lst_path_mixtures = lst_path_train
        dataset_train = Dataset(args)
        loader_train = data.DataLoader(dataset_train, batch_size=args.len_batch, shuffle=args.shuffle)

        # loader test
        args.lst_path_mixtures = lst_path_valid
        dataset_valid = Dataset(args)
        loader_valid = data.DataLoader(dataset_valid, batch_size=args.len_batch, shuffle=args.shuffle)

        # start training
        str_time = time.time()
        args.logger.info("training started")

        lst_train = []
        lst_valid = []

        for epoch in range(args.num_epochs):
            lst_train_b = []
            opt = optim.Adam(model.parameters(), lr = fun_lr(epoch, args))

            for inputs, outputs in loader_train:
                opt.zero_grad()
                outputs_ = model(inputs)
                loss = fun_loss(outputs_, outputs, args)
                loss.backward()
                opt.step()
                lst_train_b.append(loss.item())

            loss_train = np.mean(lst_train_b)
            args.logger.info("epoch: {:3d}, train loss: {:7f}".format(epoch, loss_train))
            lst_train.append(loss_train)

            with torch.no_grad():
                lst_valid_b = []
                for inputs, outputs in loader_valid:
                    outputs_ = model(inputs)
                    loss = fun_loss(outputs_, outputs, args)
                    lst_valid_b.append(loss.item())

                loss_valid = np.mean(lst_valid_b)
                args.logger.info("epoch: {:3d}, valid loss: {:7f}".format(epoch, loss_valid))
                lst_valid.append(loss_valid)

        args.logger.info("training ended")

        end_time = time.time()
        hours, rem = divmod(end_time-str_time, 3600)
        minutes, seconds = divmod(rem, 60)
        args.logger.info('training time:{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

        torch.save(model.state_dict(), path_pth)
        args.logger.info('model saved')

        # save loss
        loss = {'train': lst_train, 'valid': lst_valid}
        args.path_pkl = os.path.join(args.path_loss, '{}.pkl'.format(args.name_model))
        ut.save_pkl(loss, args)

        # plot loss vs epoch
        args.path_fig = os.path.join(args.path_loss, '{}.png'.format(args.name_model))
        ut.plot_train_valid(lst_train, lst_valid, args)

    # test
    args.model = DenseUnet(args).to(args.device)

    with torch.no_grad():
        args.model.load_state_dict(torch.load(path_pth))
        args.model.eval()

    args.logger.info("model {} loaded".format(args.name_model))

    num_params = sum(p.numel() for p in args.model.parameters() if p.requires_grad)
    # args.logger.info("number of model's parameters: {}".format(num_params))

    x = detect(x, args)

    x = segment(x, args)

    args.path_pkl = os.path.join(args.path_sources_, "{}.pkl".format(x.name))
    ut.save_pkl(x, args)
    
    args.path_fig = os.path.join(args.path_figures, "{}.png".format(x.name))

    if hasattr(args, 'path_figures') and hasattr(args, 'path_fig'):
        ut.plot_mixture(x, args)
    return x

def fun_loss(y_, y, args):
    a, b, c, d = args.lst_reg
    box = F.max_pool2d(y[:,:3], 7)
    box_ = F.avg_pool2d(y_[:, :3], 7)

    loss_c = a*bce(box_[:, 0], box[:, 0])
    loss_h = b*mse(torch.mul(box[:, 0], torch.sqrt(box_[:, 1])), torch.sqrt(box[:, 1]))
    loss_w = c*mse(torch.mul(box[:, 0], torch.sqrt(box_[:, 2])), torch.sqrt(box[:, 2]))

    loss_m = d*(bce(y_[:, 3], y[:, 3]))

    return loss_c + loss_h + loss_w + loss_m

def fun_lr(epoch, args):
    return args.str_lr + epoch*(args.end_lr-args.str_lr)/args.num_epochs

def bce(y_, y):
    return F.binary_cross_entropy(y_, y)

def mse(y_, y):
    return F.mse_loss(y_, y)

def detect(x, args):
    F, T = x.image.shape

    # loader detection
    input = torch.tensor(x.image, dtype=torch.float).to(args.device).unsqueeze(0)
    input = torch.cat((input, input)).unsqueeze(0)

    output_ = args.model(input)

    output_ = torch.cat((output_.cpu(), torch.zeros([1, 4, F, T-output_.shape[-1]])), dim=3)

    x.box_ = np.zeros([3, F, T])
    x.boxed_ = np.empty([0, F, T])

    box_ = output_[0][:3]

    box_ = box_.cpu().data.numpy()

    cd = skimage.measure.block_reduce(box_[0], (7,7), np.max)
    cu = scipy.ndimage.zoom(cd, 7, order=0)[:F, :T]
    c_ = (box_[0] == cu).astype(int)
    c_ = np.multiply(box_[0], c_)

    hd = skimage.measure.block_reduce(box_[1], (7,7), np.mean)
    h_ = scipy.ndimage.zoom(hd, 7, order=0)[:F, :T]

    wd = skimage.measure.block_reduce(box_[2], (7,7), np.mean)
    w_ = scipy.ndimage.zoom(wd, 7, order=0)[:F, :T]


    centers_ = np.where(c_ > args.thr_det)
    heights_ = h_[centers_]
    widths_ = w_[centers_]
    lst_area = np.multiply(heights_, widths_)

    ind = np.argsort(lst_area)[::-1]

    lst_boxes = []

    for i in ind:
        if lst_area[i] < 2000:
            continue
        is_source = True
        box_i = np.zeros_like(x.image)
        lh_ = max(int(centers_[0][i] - heights_[i]//2)-10, 0)
        lw_ = max(int(centers_[1][i] - widths_[i]//2)-10, 0)
        rh_ = min(lh_ + int(heights_[i]) + 10, F)
        rw_ = min(lw_ + int(widths_[i]) + 10, T)

        h_ = int(rh_ - lh_)
        w_ = int(rw_ - lw_)
        box_i[lh_:rh_, lw_:rw_] = 1
        for box_j in lst_boxes:

            # intersection
            box_i_and_j = np.zeros_like(x.image)
            box_i_and_j[(box_i==1) & (box_j==1)] = 1
            intersection = np.sum(box_i_and_j)

            # union
            box_i_or_j = np.zeros_like(x.image)
            box_i_or_j[(box_i==1) | (box_j==1)] = 1
            union = np.sum(box_i_or_j)

            iou = intersection/union if union else 0

            if iou > args.thr_iou_det:
                is_source = False

                break
        if is_source:
            x.box_[:, centers_[0][i], centers_[1][i]] = [1, h_, w_]
            boxed_ = np.array([np.multiply(x.image, box_i)])
            x.boxed_ = np.append(x.boxed_, boxed_, axis=0)
            lst_boxes.append(box_i)

    return x

def segment(x, args):
    F, T = x.image.shape
    x.mask_ = np.zeros([0, F, T])
    masks_ = np.ones([F, T])
    x.box_ = np.zeros_like(x.box_)
    for i_boxed_, boxed_ in enumerate(x.boxed_):
        box_ = np.where(boxed_ ==0, 0, 1)
        boxed_ = ut.resize(boxed_)
        boxed_ = ut.normalize(boxed_)
        input = torch.cat((
            torch.tensor(x.image,
                dtype=torch.float).unsqueeze(0),
            torch.tensor(boxed_,
                dtype=torch.float).unsqueeze(0))).unsqueeze(0).to(args.device)
        output_ = args.model(input)
        output_ = torch.cat((output_.cpu(), torch.zeros([1, 4, F, T-output_.shape[-1]])), dim=3)

        # first of batch
        mask_ = output_[0][3].data.numpy()
        mask_ = ut.normalize(mask_)

        mask_ = ut.segment(mask_, box_, args)

        if np.sum(mask_) > 0:
            mask_i = mask_
            is_source = True
            for j, mask_j in enumerate(x.mask_):
                # intersection
                mask_i_and_j = np.zeros_like(x.image)
                mask_i_and_j[(mask_i==1) & (mask_j==1)] = 1
                intersection = np.sum(mask_i_and_j)
                # union
                mask_i_or_j = np.zeros_like(x.image)
                mask_i_or_j[(mask_i==1) | (mask_j==1)] = 1
                union = np.sum(mask_i_or_j)

                iou = intersection/union if union else 0
                if iou > args.thr_iou_seg:
                    x.mask_[j] = mask_i_or_j
                    is_source = False
            if is_source:
                x.mask_ = np.append(x.mask_, np.array([mask_]), axis=0)
                l = np.where(mask_)
                h = np.max(l[0]) - np.min (l[0]) + 1
                w = np.max(l[1]) - np.min (l[1]) + 1
                c = [np.min(l[0])+h//2, np.min(l[1])+w//2]
                x.box_[:, c[0], c[1]] = [1, h, w]
    return x

class _Bottleneck(nn.Sequential):
    def __init__(self, args):
        super(_Bottleneck, self).__init__()

        # hyperparams
        self.droprate = args.droprate

        # layers
        args.i_layer += 1
        self.num_inputs = args.num_inputs
        self.num_outputs = 4 * args.rate_growth
        self.add_module('norm{}'.format(args.i_layer), nn.BatchNorm2d(self.num_inputs))
        self.add_module('relu{}'.format(args.i_layer), nn.ReLU(inplace=True))
        self.add_module('conv{}'.format(args.i_layer), nn.Conv2d(self.num_inputs, self.num_outputs,
                                    kernel_size=1, stride=1, bias=False))

        args.i_layer += 1
        self.num_inputs = self.num_outputs
        self.num_outputs = args.rate_growth
        self.add_module('norm{}'.format(args.i_layer), nn.BatchNorm2d(self.num_inputs))
        self.add_module('relu{}'.format(args.i_layer), nn.ReLU(inplace=True))
        self.add_module('conv{}'.format(args.i_layer), nn.Conv2d(self.num_inputs, self.num_outputs,
                                    kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        y = super(_Bottleneck, self).forward(x)
        if self.droprate > 0:
            y = F.dropout(y, p=self.droprate,training=self.training)
        y = torch.cat([x, y], 1)
        return y

class _Block(nn.Sequential):
    def __init__(self, args):
        super(_Block, self).__init__()
        self.num_inputs = args.num_inputs

        # layers
        for i_bottleneck in range(args.num_bottlenecks):

            # update args bottleneck
            args.num_inputs = self.num_inputs + i_bottleneck*args.rate_growth
            double = _Bottleneck(args)
            self.add_module('bottleneck{}'.format(i_bottleneck+1), double)

class _Downsample(nn.Sequential):
    def __init__(self, args):
        super(_Downsample, self).__init__()

        # layers
        args.i_layer +=1
        self.num_inputs = args.num_inputs
        self.num_outputs = args.num_outputs
        self.add_module('norm{}'.format(args.i_layer), nn.BatchNorm2d(self.num_inputs))
        self.add_module('relu{}'.format(args.i_layer), nn.ReLU(inplace=True))
        self.add_module('conv{}'.format(args.i_layer), nn.Conv2d(self.num_inputs, self.num_outputs,
                                    kernel_size=1, stride=1, bias=False))
        self.add_module('pool{}'.format(args.i_layer), nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        y = super(_Downsample, self).forward(x)
        return y

class _Upsample(nn.Sequential):
    def __init__(self, args):
        super(_Upsample, self).__init__()

        # layers
        args.i_layer +=1
        self.num_inputs = args.num_inputs
        self.add_module('norm{}'.format(args.i_layer), nn.BatchNorm2d(args.num_inputs))
        self.add_module('relu{}'.format(args.i_layer), nn.ReLU(inplace=True))
        self.add_module('convt{}'.format(args.i_layer), nn.ConvTranspose2d(args.num_inputs, args.num_outputs,
                                    kernel_size=2, stride=2, bias=False))

    def forward(self, x):
        y = super(_Upsample, self).forward(x)
        return y

class DenseUnet(nn.Module):
    def __init__(self, args):
        super(DenseUnet, self).__init__()
        # hyperparams
        self.num_inputs = 2
        self.num_outputs = 2 * args.rate_growth

        # first layer
        args.i_layer = 0

        lst_num_hiddens = []

        # encoder
        lst_encoder = []
        args.num_inputs = self.num_inputs
        args.num_outputs = self.num_outputs
        for i_block, num_bottlenecks in enumerate(args.lst_num_bottlenecks):
            lst_encoder.append(('downsample{}'.format(i_block+1), _Downsample(args)))
            lst_num_hiddens.append(args.num_outputs)
            args.num_inputs = args.num_outputs
            args.num_bottlenecks = num_bottlenecks
            lst_encoder.append(('block{}'.format(i_block+1), _Block(args)))
            args.num_inputs = args.num_outputs + num_bottlenecks*args.rate_growth
            args.num_outputs = int(math.floor(args.num_inputs*args.rate_reduction))
        self.encoder = nn.Sequential(collections.OrderedDict(lst_encoder))

        num_blocks = i_block + 1

        # decoder
        lst_decoder = []
        for i_block, num_bottlenecks in reversed(list(enumerate(args.lst_num_bottlenecks))):
            args.num_inputs = args.num_inputs + lst_num_hiddens[i_block]
            args.num_outputs = int(math.floor(args.num_inputs*args.rate_reduction))
            lst_decoder.append(('usample{}'.format(i_block + 1), _Upsample(args)))

            args.num_bottlenecks = num_bottlenecks
            args.num_inputs = args.num_outputs
            lst_decoder.append(('block{}'.format(i_block + 1), _Block(args)))
            args.num_inputs = args.num_outputs + num_bottlenecks*args.rate_growth
        self.decoder = nn.Sequential(collections.OrderedDict(lst_decoder))

        args.i_layer +=1

        lst_output = [('linear{}'.format(args.i_layer), nn.Linear(args.num_inputs, 4))]
        self.output = nn.Sequential(collections.OrderedDict(lst_output))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        lst_hidden = []
        for i_module, module in enumerate(self.encoder):
            y = module(y)
            if i_module%2 == 0:
                lst_hidden.append(y)
        hidden = lst_hidden[-1]
        for i_module, module in enumerate(self.decoder):
            if i_module%2 == 0:
                h = lst_hidden[-i_module//2-1]
                y = torch.cat([y, h], 1)
            y = module(y)

        y = torch.transpose(y, 1, 3)
        y = self.output(y)
        y = torch.transpose(y, 1, 3)

        y[:, 0] = torch.sigmoid(y[:, 0])

        y[:, 3] = torch.sigmoid(y[:, 3])

        y = torch.relu(y)

        return y 

class Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.lst_path_mixtures = args.lst_path_mixtures
        self.num_sources = args.lst_num_sources[0]
        self.len_dataset = len(self.lst_path_mixtures)*self.num_sources

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, i):
        i_mixture = i//self.num_sources
        i_source = i%self.num_sources

        path_file = self.lst_path_mixtures[i_mixture]
        x = ut.load_pkl(path_file)
        input = torch.tensor(x.image, dtype=torch.float).to(self.args.device).unsqueeze(0)
        boxed_i = x.boxed[i_source]

        boxed_i = ut.resize(x.boxed[i_source])
        boxed_i = ut.normalize(boxed_i)

        input = torch.cat((input, torch.tensor(boxed_i,
                        dtype=torch.float).to(self.args.device).unsqueeze(0)))

        output = torch.tensor(x.box, dtype=torch.float).to(self.args.device)
        mask_i = x.mask[i_source]

        mask_i = ut.resize(x.mask[i_source])
        mask_i = ut.normalize(mask_i)
        mask_i = np.where(mask_i > .5, 1, 0)

        output = torch.cat((output, torch.tensor(mask_i,
                        dtype=torch.float).to(self.args.device).unsqueeze(0)))
        return input, output
