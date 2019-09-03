#!/usr/bin/env python
import os, sys
import pickle
import numpy as np
import librosa
import models
import utils as ut


# path setting
lst_args = sys.argv[1:]
assert len(lst_args)==1, "Not a valid argument"
_, name_suffix = lst_args[0].split('.')
assert name_suffix == 'ini', "Not a valid config"
path_setting = lst_args[0]
assert os.path.exists(path_setting), "Config doesn't exist!"

def main():
    # settings
    args = ut.settings(path_setting)
    
    # path figures
    args.path_figures = os.path.join('results', args.name_setting, 'figures')
    ut.make_dir(args.path_figures)
    
    # path model folder
    args.path_model = os.path.join('results', args.name_setting, 'models')
    print(args.path_model)
    ut.make_dir(args.path_model)
    
    # path pth model
    args.path_pth = os.path.join('results', args.name_model, 'models', "{}_{}.pth".format(args.name_model, args.i_fold))

    print(args.path_pth)

    # list all the inputs
    lst_path_mixtures = ut.lst_path_endswith(args.path_mixtures, args.name_format)
    
    # len inputs
    len_mixture = int(args.len_hop*(args.len_input-1))
    
    print(lst_path_mixtures)
    len_mixtures = 0
    num_sources_ = 0
    for path_mixture in lst_path_mixtures:
        mixture = ut.read(path_mixture, args)
        len_mixtures += len(mixture)
        frames = librosa.util.frame(mixture, len_mixture, len_mixture//2).T
        for i_frame, frame in enumerate(frames):
            x = args.dataset
            x.amplitude = frame
            x.name = '{}_{}'.format(path_mixture.split('/')[-1].split('.')[0], i_frame)
            x.image = ut.image(x.amplitude, args)
            x = models.separate(x, args)
            num_sources_ += x.mask_.shape[0]
            # plot 
            args.path_fig = os.path.join(args.path_figures, "{}.png".format(x.name))
            ut.plot_mixture(x, args)
            print(num_sources_)
        args.logger.info('number of sources: {}'.format(num_sources_))

if __name__ == "__main__":
    main()
