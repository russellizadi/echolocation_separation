#!/usr/bin/env python
import os
import shutil
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import torch
import librosa
import soundfile
import utils as ut
import scipy.ndimage as ndimage

class Dataset:
    """
    make a dataset object
    """
    def __init__(self, args):
        args.path_dataset = os.path.join(args.path_datasets, args.name_dataset)

        args.path_sources = os.path.join(args.path_results, 'sources')
        ut.make_dir(args.path_sources)

        args.path_noises = os.path.join(args.path_results, 'noises')
        ut.make_dir(args.path_noises)

    def sources(self, args):

        args.logger.info("Extracting sources started!")

        lst_path_files = ut.lst_path_endswith(args.path_dataset, args.name_format)

        args.logger.info("{} files found.".format(len(lst_path_files)))

        len_source = int(args.len_hop*(args.len_input-1))
        len_hop = int(.9*len_source)

        num_sources = 0
        num_noises = 0

        #
        for i_file, path_file in enumerate(lst_path_files):
            # read
            amplitude = ut.read(path_file, args)
            len_amplitude = len(amplitude)
            num_frames = 1 + int((len_amplitude - len_source) / len_hop)
            args.logger.info("File: {}, length: {}, number of frames: {}".format(i_file, len_amplitude, num_frames))

            # frames
            i_frame = 0
            while True:
                frame = np.arange(i_frame*len_hop, i_frame*len_hop + len_source)
                if frame[-1] > len(amplitude):
                    break
                self.amplitude = amplitude[frame]
                self.image = ut.image(self.amplitude, args)
                self_i = ut.labels(self, args)
                for j_self, self_j in enumerate(self_i):
                    # source
                    if not self_j.is_noise and num_sources < args.num_sources_max:
                        self = self_j
                        self.name = '{}_{}_{}'.format(path_file.split('/')[-1].split('.')[0], i_frame, j_self)

                        args.path_pkl = os.path.join(args.path_sources, self.name + '.pkl')
                        ut.save_pkl(self, args)
                        num_sources += 1

                        # plot
                        if 1:
                            path_figures = os.path.join(args.path_sources, 'figures')
                            ut.make_dir(path_figures)
                            args.path_fig = os.path.join(path_figures, self.name + '.png')
                            ut.plot_source(self, args)


                    # noise
                    elif num_noises < num_sources and self_j.is_noise:

                        self.name = '{}_{}'.format(path_file.split('/')[-1].split('.')[0], i_frame)

                        args.path_pkl = os.path.join(args.path_noises, self.name + '.pkl')
                        ut.save_pkl(self, args)
                        num_noises += 1

                        # plot
                        if num_noises < 10:
                            path_figures = os.path.join(args.path_noises, 'figures')
                            ut.make_dir(path_figures)
                            args.path_fig = os.path.join(path_figures, self.name + '.png')
                            ut.plot_image(self.image, args)
                        break
                i_frame += 1
                args.logger.info("frame: {}, number of sources: {}, and noises: {}".format(i_frame, num_sources, num_noises))
                if num_noises >= args.num_sources_max:
                    break
            args.logger.info("total number of sources: {}".format(num_sources))
        args.logger.info("extracting sources ended.")

class NIPS4Bplus(Dataset):
    def __init__(self, args):
        super(NIPS4Bplus, self).__init__(args)

class BelleBats(Dataset):
    def __init__(self, args):
        super(BelleBats, self).__init__(args)

class BirdVox_full_night(Dataset):
    def __init__(self, args):
        super(BirdVox_full_night, self).__init__(args)

class Bottlenose_Fremantle(Dataset):
    def __init__(self, args):
        super(Bottlenose_Fremantle, self).__init__(args)
