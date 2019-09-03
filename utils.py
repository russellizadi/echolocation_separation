import os
import shutil
import logging
import configparser
import pickle
import numpy as np
import scipy
import scipy.ndimage as ndimage
from scipy.stats import multivariate_normal
import skimage.morphology as morphology
import skimage
import skimage.segmentation
import skimage.morphology
import skimage.measure
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import torch
import librosa
import soundfile
import datasets

def lst_path_endswith(path, suffix):
    """List files ending with the suffix in the path
    """
    lst_path = []
    for root, dir, file in os.walk(path):
        lst_path += [os.path.join(root, p) for p in file if p.endswith(suffix) and root==path]
    lst_path.sort()
    return lst_path

def make_rm_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_pkl(x, args):
    with open(args.path_pkl, 'wb') as file:
        pickle.dump(x, file, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def lst_dir_endswith(path, suffix):
    lst_path = []
    for root, dir, file in os.walk(path):
        lst_path += [root.split('/')[-1] for p in file if p.endswith(suffix)]
    lst_path = list(set(lst_path))
    return lst_path

def fold(lst_path, args):
    len_lst = len(lst_path)
    i_str = int(args.i_fold*len(lst_path)//5)
    i_end = i_str + len(lst_path)//5 +1
    lst_path_test = lst_path[i_str:i_end]
    lst_path_train = [path for path in lst_path if path not in lst_path_test]
    return lst_path_train, lst_path_test

def active(x, args):
    y = np.array(x >= args.thr_act, dtype=np.float32)
    return y



def settings(path):
    """
    read setting files .ini
    :params path: the path to the setting file .ini
    :return args: the arguments object
    """
    # name arguments object 
    name = path.split('/')[-1].split('.')[0]
    args = type(name, (), {})()
    args.name_setting = name
    # read setting
    config = configparser.ConfigParser()
    config.read(path)
    for section in config.sections():
        for arg in config[section]:
            value = eval(config[section][arg])
            setattr(args, arg, value)
    # path setting
    args.path_setting = path
    # path results and makedir
    args.path_results = os.path.join('results', args.name_setting)
    make_dir(args.path_results)
    # path models and makedir
    args.path_models = os.path.join(args.path_results, 'models')
    make_dir(args.path_models)
    # rest
    args.logger = logger(args)
    args.dataset = getattr(datasets, args.name_dataset)(args)
    args.device = torch.device('cuda:{}'.format(args.i_cuda) if torch.cuda.is_available() else 'cpu')
    args.logger.info('settings is read!')
    return args

def logger(args):
    args.path_logger = os.path.join(args.path_results, args.name_setting + '.log')
    logging.basicConfig(filename=args.path_logger,
        filemode='w',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO)
    return logging.getLogger(args.name_setting)

def normalize(x):
    if np.min(x) == np.max(x):
        return np.zeros_like(x)
    x -= np.min(x)
    x /= np.max(x)
    return x

def read(path, args):
    endswith = path.split('.')[1]
    if endswith=='wav':
        amplitude, _ = librosa.load(path, sr=args.f_sampling)
    elif endswith=='flac':
        amplitude, _ = soundfile.read(path)
    return amplitude

def labels(x, args):
    m = np.zeros_like(x.image)
    len_frames_t = [int(x.image.shape[1]/pow(2, i)) for i in range(0, 4)]
    len_frames_f = [int(x.image.shape[0]/pow(2, i)) for i in range(0, 1)]
    for len_frame_t in len_frames_t:
        for len_frame_f in len_frames_f:
            i_frames_t = librosa.util.frame(np.arange(x.image.shape[1]), len_frame_t, len_frame_t//2).T
            i_frames_f = librosa.util.frame(np.arange(x.image.shape[0]), len_frame_f, len_frame_f//2).T
            for i_frame_t in i_frames_t:
                for i_frame_f in i_frames_f:
                    x_windowed = x.image[np.ix_(i_frame_f, i_frame_t)]
                    for args.thr_act in np.linspace(.4, .8, 5):
                        m_windowed = active(x_windowed, args)
                        m_windowed = morphology.remove_small_objects(m_windowed.astype(np.bool), min_size=20)
                        l, n = ndimage.label(m_windowed)
                        if n==1:
                            l = np.where(m_windowed==1)
                            h = np.max(l[0]) - np.min (l[0]) + 1
                            w = np.max(l[1]) - np.min (l[1]) + 1
                            if ((w*h > 2000)
                                and (np.min(l[1]) > 1)
                                and (np.max(l[1]) < m_windowed.shape[1]-1)
                                and (np.min(l[0]) > 1)
                                and (np.max(l[0]) < m_windowed.shape[0]-1)):
                                m[np.ix_(i_frame_f, i_frame_t)] += m_windowed.astype(np.float)
                                break
    m = np.clip(m, 0, 1)
    if m.any():
        l, n = ndimage.label(m)
        for i in range(n):
            ls = np.where(l==(i+1))
            # mask
            m_i = np.zeros_like(m)
            m_i[ls] = 1
            if np.sum(np.multiply(x.image, m_i))/np.sum(x.image) > .2:
                x.mask = m_i
                x.masked = np.multiply(x.image, m_i)
                # box
                b_i = np.zeros([3, m.shape[0], m.shape[1]])
                h = np.max(ls[0]) - np.min (ls[0]) + 1
                w = np.max(ls[1]) - np.min (ls[1]) + 1
                c = [np.min(ls[0])+h//2, np.min(ls[1])+w//2]
                b_i[:, c[0], c[1]] = [1, h, w]
                x.box = b_i
                b_i = np.zeros_like(m)
                b_i[np.min(ls[0]):np.max(ls[0]), np.min(ls[1]):np.max(ls[1])] = 1
                x.boxed = np.multiply(x.image, b_i)
                x.is_noise = False
                yield x
    else:
        x.is_noise = True
        yield x

def box(x):
    F, T = x[0].shape
    c = np.where(x[0])
    h = x[1][c]
    w = x[2][c]
    b = np.zeros_like(x[0])
    hd = int(max(0, c[0]-h//2+1))
    hu = int(min(F, c[0]+h//2+1))
    wd = int(max(0, c[1]-w//2+1))
    wu = int(min(T, c[1]+w//2+1))
    b[hd:hu, wd:wu] = 1
    return b

def mixer(mix, x, args):
    snr = np.random.uniform(low=-args.snr_mixing, high=args.snr_mixing)
    #fac = np.random.rand(1)[0]*args.snr_mixing
    mix.image += x.image*(10**(snr/10))
    mix.box += x.box
    mix.mask = np.append(mix.mask, np.array([x.mask]), axis=0)
    boxed = np.multiply(box(x.box), mix.image)
    mix.boxed = np.append(mix.boxed, np.array([boxed]), axis=0)
    return mix

def mixtures(args):
    assert len(args.lst_path_sources) >= args.num_sources, "Not enough sources to mix!"
    for i_mixture in range(args.num_mixtures):

        path_noise = np.random.choice(args.lst_path_noises)
        mixture = load_pkl(path_noise)
        F, T = mixture.image.shape

        mixture.box = np.zeros([3, F, T])
        mixture.mask = np.empty([0, F, T])
        mixture.boxed = np.empty([0, F, T])

        masks = np.zeros([1, F, T])
        num_sources = 0
        while num_sources < args.num_sources:
            path_source = np.random.choice(args.lst_path_sources)
            source = load_pkl(path_source)
            do_mix = True
            for mask in masks:
                # intersection
                mask_i_and_j = np.zeros_like(mixture.image)
                mask_i_and_j[(mask==1) & (source.mask==1)] = 1
                intersection = np.sum(mask_i_and_j)

                if intersection > 10:
                    do_mix = False
                    break

            if do_mix:
                mixture = mixer(mixture, source, args)
                masks = np.append(masks, np.array([source.mask]), axis=0)
                num_sources += 1
        mixture.image = normalize(mixture.image)

        # save the mixture
        mixture.name = str(i_mixture)
        args.path_pkl = os.path.join(args.path_mixtures, mixture.name + '.pkl')

        save_pkl(mixture, args)

        # plot the mixture
        if hasattr(args, 'path_figures') and i_mixture < 100:
            make_dir(args.path_figures)
            args.path_fig = os.path.join(args.path_figures, mixture.name + '.png')
            plot_mixture(mixture, args)

    args.logger.info("number of mixtures: {}".format(args.num_mixtures))

def segment(m, b, args):
    F, T = m.shape

    # active
    args.thr_act = args.thr_seg

    y = active(m, args)

    # largest
    label, n = ndimage.label(y)
    if np.max(label) == 0: return 0
    inds = np.argsort([region.area for region in skimage.measure.regionprops(label)])
    l = np.where(label==inds[-1]+1)
    h = np.max(l[0]) - np.min(l[0])
    w = np.max(l[1]) - np.min(l[1])
    y = np.where(label==inds[-1]+1, 1, 0)
    if (h/F < .8 or w/T < .8): return 0

    y = ndimage.binary_fill_holes(y)
    p = skimage.measure.perimeter(y)
    if p > 700: return 0

    # closing
    y = skimage.morphology.closing(y, np.ones([10, 10]))

    l = np.where(b > 0)
    h = np.max(l[0]) - np.min(l[0])
    w = np.max(l[1]) - np.min(l[1])
    m_ = scipy.misc.imresize(m, size=[h, w]).astype(float)
    y = np.zeros_like(m)
    y[np.min(l[0]):np.max(l[0]), np.min(l[1]):np.max(l[1])] = m_
    y = normalize(y.astype(float))

    args.thr_act = args.thr_seg
    y = active(y, args)
    y = ndimage.binary_fill_holes(y)
    label, n = ndimage.label(y)
    if np.max(label) == 0: return 0

    inds = np.argsort([region.area for region in skimage.measure.regionprops(label)])
    y = np.where(label==inds[-1]+1, 1, 0)

    return y

def resize(x):
    F, T = x.shape
    l = np.where(x > 0)
    box = x[np.min(l[0]):np.max(l[0]), np.min(l[1]):np.max(l[1])]
    y = scipy.misc.imresize(box, size=[F, T]).astype(float)
    return y

def unresize(x, b):
    l = np.where(b > 0)
    h = np.max(l[0]) - np.min(l[0])
    w = np.max(l[1]) - np.min(l[1])
    m = scipy.misc.imresize(x, size=[h, w]).astype(float)
    y = np.zeros_like(x)
    y[np.min(l[0]):np.max(l[0]), np.min(l[1]):np.max(l[1])] = m
    y = normalize(y)
    return y

def save_mask(args):
    args.path_pkl = os.path.join(args.path_results, 'mask.pkl')
    if not os.path.isfile(args.path_pkl):
        lst_path_mixtures = lst_path_endswith(args.path_mixtures, '.pkl')
        avg_mask = np.zeros([args.dim_input, args.len_input])

        for path_mixture in lst_path_mixtures:
            x = load_pkl(path_mixture)
            for mask in x.mask:
                mask = resize(mask)
                mask = normalize(mask)
                avg_mask += mask

        avg_mask = normalize(avg_mask)
        save_pkl(avg_mask, args)

def evaluate(args):
    """
    Parameters:
        args

    Returns
    """
    metric = {}
    lst_path_mixtures = lst_path_endswith(args.path_mixtures, '.pkl')
    for path_mixture in lst_path_mixtures:
        x = load_pkl(path_mixture)
        metric = metric_update(x, metric, args)
    return metric

def metric_update(x, metric, args):

    box = x.box
    centers = np.where(box[0] == 1)
    num_sources = len(box[0][centers])
    heights = box[1][centers]
    widths = box[2][centers]

    box_ = x.box_
    centers_ = np.where(box_[0] == 1)
    num_sources_ = len(box_[0][centers_])
    heights_ = box_[1][centers_]
    widths_ = box_[2][centers_]

    intersection = np.zeros([num_sources, num_sources_])
    union = np.zeros([num_sources, num_sources_])
    ground_truth = np.zeros([num_sources, num_sources_])
    prediction = np.zeros([num_sources, num_sources_])

    for i in range(num_sources):
        box_i = np.zeros_like(x.image)
        ch = int(centers[0][i] - heights[i]//2)
        cw = int(centers[1][i] - widths[i]//2)
        box_i[ch:ch+int(heights[i]), cw:cw+int(widths[i])] = 1
        for j in range(num_sources_):
            box_j = np.zeros_like(x.image)
            ch_ = int(centers_[0][j] - heights_[j]//2)
            cw_ = int(centers_[1][j] - widths_[j]//2)
            box_j[ch_:ch_+int(heights_[j]), cw_:cw_+int(widths_[j])] = 1

            # fill out the intersection matrix
            box_i_and_j = np.zeros_like(x.image)
            box_i_and_j[(box_i==1) & (box_j==1)] = 1
            intersection[i, j] = np.sum(box_i_and_j)

            # fill out the union matrix
            box_i_or_j = np.zeros_like(x.image)
            box_i_or_j[(box_i==1) | (box_j==1)] = 1
            union[i, j] = np.sum(box_i_or_j)

            # fill out ground_truth and prediction
            ground_truth[i, :] = np.sum(box_i)
            prediction[:, j] = np.sum(box_j)

    iou = np.divide(intersection, union)
    a = np.copy(iou)
    if iou.size != 0:
        a[a < np.max(a, axis=1, keepdims=True)] = 0
        a[a < np.max(a, axis=0, keepdims=True)] = 0
        a = (a > args.thr_iou).astype(int)
    # number of boxes with good IoU
    TP = np.sum(a)
    FN = num_sources - TP
    FP = num_sources_ - TP

    metric.setdefault('dTPs',[]).append(TP)
    metric.setdefault('dFNs',[]).append(FN)
    metric.setdefault('dFPs',[]).append(FP)
    metric.setdefault('dTPRs',[]).append(TP/(TP+FN))
    metric.setdefault('dFNRs',[]).append(FN/(FN+TP))
    metric.setdefault('dPs',[]).append(TP/(TP+FP) if TP+FP else 0)
    metric.setdefault('dRs',[]).append(TP/(TP+FN))
    metric.setdefault('dFs',[]).append(2*TP/(2*TP+FP+FN))

    metric['mdTPRs'] = np.mean(metric['dTPRs'])
    metric['mdFNRs'] = np.mean(metric['dFNRs'])
    metric['mdPs'] = np.mean(metric['dPs'])
    metric['mdRs'] = np.mean(metric['dRs'])
    metric['mdFs'] = np.mean(metric['dFs'])

    # accuracy of boxes with good IoU
    if np.sum(a) > 0:
        TP = np.sum(np.multiply(intersection, a))
        FP = np.sum(np.multiply(prediction, a)) - TP
        FN = np.sum(np.multiply(ground_truth, a)) - TP
        TN = max(0, np.prod(x.image.shape) - np.sum(np.multiply(union, a)))

        metric.setdefault('dTPb',[]).append(TP)
        metric.setdefault('dFPb',[]).append(FP)
        metric.setdefault('dFNb',[]).append(FN)
        metric.setdefault('dTNb',[]).append(TN)
        metric.setdefault('dTPRb',[]).append(TP/(TP+FN))
        metric.setdefault('dFNRb',[]).append(FN/(FN+TP))
        metric.setdefault('dFPRb',[]).append(FP/(FP+TN))
        metric.setdefault('dTNRb',[]).append(TN/(TN+FP))
        metric.setdefault('dPb',[]).append(TP/(TP+FP) if TP+FP else 0)
        metric.setdefault('dRb',[]).append(TP/(TP+FN))
        metric.setdefault('dFb',[]).append(2*TP/(2*TP+FP+FN))

        metric['mdTPRb'] = np.mean(metric['dTPRb'])
        metric['mdFNRb'] = np.mean(metric['dFNRb'])
        metric['mdFPRb'] = np.mean(metric['dFPRb'])
        metric['mdTNRb'] = np.mean(metric['dTNRb'])
        metric['mdPb'] = np.mean(metric['dPb'])
        metric['mdRb'] = np.mean(metric['dRb'])
        metric['mdFb'] = np.mean(metric['dFb'])

    mask = x.mask
    num_sources = len(mask)

    mask_ = x.mask_
    num_sources_ = len(mask_)

    intersection = np.zeros([num_sources, num_sources_])
    union = np.zeros([num_sources, num_sources_])
    ground_truth = np.zeros([num_sources, num_sources_])
    prediction = np.zeros([num_sources, num_sources_])

    for i in range(num_sources):
        for j in range(num_sources_):
            # fill out the intersection matrix
            mask_i_and_j = np.zeros_like(x.image)
            mask_i_and_j[(mask[i]==1) & (mask_[j]==1)] = 1
            intersection[i, j] = np.sum(mask_i_and_j)

            # fill out the union matrix
            mask_i_or_j = np.zeros_like(x.image)
            mask_i_or_j[(mask[i]==1) | (mask_[j]==1)] = 1
            union[i, j] = np.sum(mask_i_or_j)

            # fill out ground_truth and prediction
            ground_truth[i, :] = np.sum(mask[i])
            prediction[:, j] = np.sum(mask_[j])

    iou = np.divide(intersection, union)
    a = np.copy(iou)
    if iou.size != 0:
        a[a < np.max(a, axis=1, keepdims=True)] = 0
        a[a < np.max(a, axis=0, keepdims=True)] = 0
        a = (a > args.thr_iou).astype(int)

    # number of masks with good IoU
    TP = np.sum(a)
    FN = num_sources - TP
    FP = num_sources_ - TP

    metric.setdefault('sTPs',[]).append(TP)
    metric.setdefault('sFNs',[]).append(FN)
    metric.setdefault('sFPs',[]).append(FP)
    metric.setdefault('sTPRs',[]).append(TP/(TP+FN))
    metric.setdefault('sFNRs',[]).append(FN/(FN+TP))
    metric.setdefault('sPs',[]).append(TP/(TP+FP) if TP+FP else 0)
    metric.setdefault('sRs',[]).append(TP/(TP+FN))
    metric.setdefault('sFs',[]).append(2*TP/(2*TP+FP+FN))

    metric['msTPRs'] = np.mean(metric['sTPRs'])
    metric['msFNRs'] = np.mean(metric['sFNRs'])
    metric['msPs'] = np.mean(metric['sPs'])
    metric['msRs'] = np.mean(metric['sRs'])
    metric['msFs'] = np.mean(metric['sFs'])

    # accuracy of boxes with good IoU
    if np.sum(a) > 0:
        TP = np.sum(np.multiply(intersection, a))
        FP = np.sum(np.multiply(prediction, a)) - TP
        FN = np.sum(np.multiply(ground_truth, a)) - TP
        TN = np.prod(x.image.shape) - np.sum(np.multiply(union, a))

        metric.setdefault('sTPb',[]).append(TP)
        metric.setdefault('sFPb',[]).append(FP)
        metric.setdefault('sFNb',[]).append(FN)
        metric.setdefault('sTNb',[]).append(TN)
        metric.setdefault('sTPRb',[]).append(TP/(TP+FN))
        metric.setdefault('sFNRb',[]).append(FN/(FN+TP))
        metric.setdefault('sFPRb',[]).append(FP/(FP+TN))
        metric.setdefault('sTNRb',[]).append(TN/(TN+FP))
        metric.setdefault('sPb',[]).append(TP/(TP+FP) if TP+FP else 0)
        metric.setdefault('sRb',[]).append(TP/(TP+FN))
        metric.setdefault('sFb',[]).append(2*TP/(2*TP+FP+FN))

        metric['msTPRb'] = np.mean(metric['sTPRb'])
        metric['msFNRb'] = np.mean(metric['sFNRb'])
        metric['msFPRb'] = np.mean(metric['sFPRb'])
        metric['msTNRb'] = np.mean(metric['sTNRb'])
        metric['msPb'] = np.mean(metric['sPb'])
        metric['msRb'] = np.mean(metric['sRb'])
        metric['msFb'] = np.mean(metric['sFb'])
    return metric

def image(x, args):
    # stft
    y = librosa.stft(x,
        n_fft=args.num_fft,
        hop_length=args.len_hop,
        win_length=args.len_win)
    # mag
    y = np.abs(y)
    # power
    y = y**2
    # Mel
    y = librosa.feature.melspectrogram(y=None,
        S=y,
        sr=args.f_sampling,
        n_fft=args.num_fft,
        n_mels=args.dim_input,
        htk=True,
        fmin=args.lst_f_range[0],
        fmax=args.lst_f_range[1])
    # pcen
    y = librosa.pcen(y,
        sr=args.f_sampling,
        hop_length=args.len_hop)
    #y = librosa.power_to_db(y)
    # normalize
    y = normalize(y)
    return y

def plot_image(x, args):
    """Plot and save spectrogram
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect(.6)
    im = ax.imshow(x, origin='lower', cmap=cm.Blues)
    fig.colorbar(im, ax=ax)
    plt.xlabel('Time', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.yticks(fontsize=20)
    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()

def plot_images(x, args):
    """Plot and save list of images
    """
    fig, axes = plt.subplots(len(x), 1, figsize=(16*len(x), 16))

    for i_ax, ax in enumerate(axes):
        im = ax.imshow(x[i_ax], origin='lower', cmap=cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Frequency', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()


def plot_source(x, args):
    num_subplots = 5
    fig, axes = plt.subplots(num_subplots, 1, figsize=(16*num_subplots, 16))

    # image
    im = axes[0].imshow(x.image, origin='lower', cmap='Blues')
    fig.colorbar(im, ax=axes[0], pad=.005)

    # box
    im = axes[1].imshow(x.image, origin='lower', cmap='Blues')
    fig.colorbar(im, ax=axes[1], pad=.005)
    c = np.where(x.box[0] == 1)
    c1 = c[0]
    c2 = c[1]
    h = x.box[1, c1, c2]
    w = x.box[2, c1, c2]
    c1 = c1 - h // 2
    c2 = c2 - w // 2
    rect = patches.Rectangle((c2,c1), w, h,linewidth=1, edgecolor=tuple(np.random.rand(3)),facecolor='none')
    axes[1].add_patch(rect)

    # mask
    im = axes[2].imshow(x.image, origin='lower', cmap='Blues')
    fig.colorbar(im, ax=axes[2], pad=.005)
    mask = x.mask
    mask[mask == 0] = np.nan
    color = [tuple(np.random.rand(3)), (1, 1, 1)]
    cmap = colors.LinearSegmentedColormap.from_list('random_binary_cmap', color, N=2)
    axes[2].imshow(mask, cmap = cmap, origin='lower', alpha=.5)

    # boxed
    im = axes[3].imshow(x.boxed, origin='lower', cmap=cm.Blues)
    fig.colorbar(im, ax=axes[3], pad=.005)

    # masked
    im = axes[4].imshow(x.masked, origin='lower', cmap=cm.Blues)
    fig.colorbar(im, ax=axes[4], pad=.005)

    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()


def plot_mixture(x, args):

    num_subplots = 1
    if hasattr(x, 'box'):
        num_subplots += 1
    if hasattr(x, 'mask'):
        num_subplots += 1
    if hasattr(x, 'box_'):
        num_subplots += 1
    if hasattr(x, 'mask_'):
        num_subplots += 1

    fig, axes = plt.subplots(num_subplots, 1, figsize=(16*num_subplots, 16))

    # image
    i_ax = 0
    ax = axes[i_ax] if num_subplots > 1 else axes
    im = ax.imshow(x.image, origin='lower', cmap=cm.Blues)
    fig.colorbar(im, ax=ax, pad=.005)

    # box
    if hasattr(x, 'box'):
        i_ax += 1
        ax = axes[i_ax]
        im = ax.imshow(x.image, origin='lower', cmap=cm.Blues)
        fig.colorbar(im, ax=ax, pad=.005)
        centers = np.where(x.box[0] == 1)
        for c in range(len(centers[0])):
            c1 = centers[0][c]
            c2 = centers[1][c]
            h = x.box[1, c1, c2]
            w = x.box[2, c1, c2]
            c1 = c1 - h // 2
            c2 = c2 - w // 2
            rect = patches.Rectangle((c2,c1), w, h,linewidth=1, edgecolor=tuple(np.random.rand(3)),facecolor='none')
            ax.add_patch(rect)

    # mask
    if hasattr(x, 'mask'):
        i_ax += 1
        ax = axes[i_ax]
        im = ax.imshow(x.image, origin='lower', cmap='Blues')
        fig.colorbar(im, ax=ax, pad=.005)
        for m in x.mask:
            mask = np.copy(m)
            mask[mask == 0] = np.nan
            color = [tuple(np.random.rand(3)), (1, 1, 1)]
            cmap = colors.LinearSegmentedColormap.from_list('random_binary_cmap', color, N=2)
            ax.imshow(mask, cmap = cmap, origin='lower', alpha=.5)

    # box_
    if hasattr(x, 'box_'):
        i_ax += 1
        ax = axes[i_ax]
        im = ax.imshow(x.image, origin='lower', cmap='Blues')
        fig.colorbar(im, ax=ax, pad=.005)
        centers = np.where(x.box_[0] == 1)
        for c in range(len(centers[0])):
            c1 = centers[0][c]
            c2 = centers[1][c]
            h = x.box_[1, c1, c2]
            w = x.box_[2, c1, c2]
            c1 = c1 - h // 2
            c2 = c2 - w // 2
            rect = patches.Rectangle((c2,c1), w, h,linewidth=1, edgecolor=tuple(np.random.rand(3)),facecolor='none')
            ax.add_patch(rect)

    # mask_
    if hasattr(x, 'mask_'):
        i_ax += 1
        ax = axes[i_ax]
        im = ax.imshow(x.image, origin='lower', cmap='Blues')
        fig.colorbar(im, ax=ax, pad=.005)
        for m in x.mask_:
            mask = np.copy(m)
            mask[mask == 0] = np.nan
            color = [tuple(np.random.rand(3)), (1, 1, 1)]
            cmap = colors.LinearSegmentedColormap.from_list('random_binary_cmap', color, N=2)
            ax.imshow(mask, cmap = cmap, origin='lower', alpha=.5)

    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()


def plot_train_valid(train, valid, args):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect(.6)

    epochs = np.arange(len(train))+1
    ax.plot(epochs, train, 'b', label='train')
    ax.plot(epochs, valid, 'r', label='valid')

    plt.legend()

    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()
