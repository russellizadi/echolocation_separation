import numpy as np 
import pickle
import librosa
import utils as ut

args = lambda: None

# pkl
# args.path_pkl = "results/mixtures_1000_2000_10/train0/40.pkl"
# x = ut.load_pkl(args.path_pkl)

# wav
args.path_wav = "mixtures/8.wav"
args.fig_pad = .01
args.fig_shrink = .5
args.fig_fraction = .05
args.f_sampling = 256e3
args.lst_f_range = [20e3, 60e3]
args.num_fft = 1024
args.len_hop = 32
args.len_win = 256
args.len_image = 256
args.dim_input = 128
args.len_frame = int(args.len_hop*(args.len_image-1))

x, _ = librosa.load(args.path_wav, args.f_sampling)
np_frames = librosa.util.frame(x, args.len_frame, args.len_frame//2).T

for ind_frame, frame in enumerate(np_frames):
    image = ut.image(frame, args)
    args.path_fig = "results/figures_jasa/8{0}.pdf".format(ind_frame)
    ut.plot_jasa(image, args)

