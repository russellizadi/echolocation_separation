#!/usr/bin/env python
import os, sys
import utils as ut
import models 

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
    lst_metric = []
    lst_name_metric = ['mdFs', 'mdFNRs', 'mdFb', 'mdTPRb', 'mdFNRb', 'msFs', 'msFNRs', 'msFb', 'msTPRb', 'msFPRb']

    for i_fold in range(args.num_folds):
        # if not trained
        args.path_mixtures = os.path.join('results', args.name_mixtures, 'train{}'.format(i_fold))
        args.i_fold = i_fold
        args.name_model = "{}_{}".format(args.name_setting, i_fold)
        args.path_model = os.path.join('results', args.name_setting)
        args.path_pth = os.path.join(args.path_model, 'models', "{}.pth".format(args.name_model))

        args.path_figures = os.path.join(args.path_model, 'figures')
        path_mixtures = os.path.join('results', args.name_mixtures, 'test{}'.format(i_fold))
        lst_path_mixtures = ut.lst_path_endswith(path_mixtures, '.pkl')
        num_sources_ = 0
        assert len(lst_path_mixtures) != 0, "No test file exists"
        for path_mixture in lst_path_mixtures:
            x = ut.load_pkl(path_mixture)
            x = models.separate(x, args)
            num_sources_ += x.mask_.shape[0]
        args.path_model = os.path.join(args.path_models, args.name_model)
        args.path_mixtures = args.path_model

        # evaluate
        metric = ut.evaluate(args)
        args.logger.info("Metrics of the fold: {}".format(i_fold))
        try:
            # JASA
            args.logger.info("{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(metric['mdFs'], metric['mdFNRs'], metric['mdFb'], metric['mdTPRb'], metric['mdFNRb'], metric['msFs'], metric['msFNRs'], metric['msFb'], metric['msTPRb'], metric['msFNRb']))
            args.logger.info("{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(metric['mdFs'], metric['mdFNRs'], metric['mdFb'], metric['mdTPRb'], metric['mdFNRb'], metric['msFs'], metric['msFNRs'], metric['msFb'], metric['msTPRb'], metric['msFNRb']))
            
            for key in lst_name_metric:
                args.logger.info("{}: {:.3f}".format(key, metric[key]))

            for key, value in metric.items():
                if key.startswith('m'):
                    args.logger.info("{}: {:.3f}".format(key, value))
            lst_metric.append(metric)
        except KeyError:
            args.logger.info("All the metrics don't exist!")

    # JASA
    args.logger.info("Average over folds:")
    for key in lst_name_metric:
        if len(lst_metric) > 0:
            metric = 0
            for i in range(len(lst_metric)):
                metric += lst_metric[i][key]
            metric = metric/len(lst_metric)
            args.logger.info("{}: {:.3f}".format(key, metric))

if __name__ == "__main__":
    main()
