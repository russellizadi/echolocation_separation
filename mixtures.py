#!/usr/bin/env python
import os, sys
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
    # path sources
    args.path_sources = os.path.join('results', args.name_sources, 'sources')
    args.path_noises = os.path.join('results', args.name_sources, 'noises')

    # save mixtures
    num_mixtures_train = int(4/5*args.num_mixtures)
    num_sources_train = args.lst_num_sources[0]
    num_mixtures_test = int(1/5*args.num_mixtures)
    num_sources_test = args.lst_num_sources[1]

    for i_fold in range(args.num_folds):
        args.i_fold = i_fold

        args.path_train = os.path.join(args.path_results, 'train{}'.format(i_fold))
        ut.make_dir(args.path_train)
        args.path_test = os.path.join(args.path_results, 'test{}'.format(i_fold))
        ut.make_dir(args.path_test)

        lst_path_sources_train, lst_path_sources_test = ut.fold(ut.lst_path_endswith(args.path_sources, '.pkl'), args)
        lst_path_noises_train, lst_path_noises_test = ut.fold(ut.lst_path_endswith(args.path_noises, '.pkl'), args)

        args.lst_path_sources = lst_path_sources_train
        args.lst_path_noises = lst_path_noises_train
        args.num_mixtures = num_mixtures_train
        args.num_sources = num_sources_train
        args.path_mixtures = args.path_train
        args.path_figures = os.path.join(args.path_train, 'figures')
        args.logger.info("mixing train fold {} started".format(i_fold))
        ut.mixtures(args)
        args.logger.info("train mixtures of fold {} saved".format(i_fold))

        args.lst_path_sources = lst_path_sources_test
        args.lst_path_noises = lst_path_noises_test
        args.num_mixtures = num_mixtures_test
        args.num_sources = num_sources_test
        args.path_mixtures = args.path_test
        args.path_figures = os.path.join(args.path_test, 'figures')
        args.logger.info("mixing test fold {} started".format(i_fold))
        ut.mixtures(args)
        args.logger.info("test mixtures of fold {} saved".format(i_fold))

if __name__ == "__main__":
    main()
