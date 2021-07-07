# Echolocation Separation

For each of the following experiments, there is one python file `.py`. Associated with each script, a config file `.ini` is available in the `settings` folder. Change the requiered config before running scripts. All results associated with each experiment will be in a folder inside the `results` folder with the same name as the config file.

# Steps to extract binary mask from mixtures
1.  Activate the environment:
```bash
source setup.sh
```
2. Extract the sources into a csv file:
```
python sources.py
```

#### Finding sources for a folder of recordings
Before running the `sources.py` script, change the following variables in the `CFG_SOURCE` in the `settings` folder. `path_dataset` should contain a folder in name `name_dataset`. The masking algorithm runs over all the `wav` or `flac` files inside the folder `path-datasets/name_dataset`. Set the `num_sources_max` to number of needed sources.

To extract individual masks, run the following command:  
``` bash
python sources.py CFG_SOURCE
```
#### Mixing sources
After finding sources, change the `CFG_MIXTURE` in the `settings` folder. `name_sources` should be the name of the config file used in extracting sources. Set the `num_mixtures` to the number of needed mixtures. 

To randomly mix all the sources, run the following command:
```bash
python mixtures.py CFG_MIXTURE
```

#### Training a model
If you have a pretrained model, you can skip this part. After mixing sources, change the `CFG_MODEL` in the `settings` folder and set `name_mixtures` to the appropriate folder name containing train and test mixtures. You can set the number of folds through the `num_folds` variable. Set `num_epochs` and `len_batch` variables for the number of epochs and the batch size.  

To train and test a model, run the following command:
```bash
python train.py CFG_MODEL
```

#### Using a separation model
If you have a pretrained model at `results` folder, set the variable `name_model` at the config file `CFG_SEPARATE`. Also, you may need to set the `i_fold` if you have multiple pretrained models with the same config. The variables `thr_det` and `thr_seg` are detection and segmentation thresholds respectively.  

To separate all the files with the suffix `name_format` and the sampling rate of `f_sampling` in the folder `path_mixtures`, with the run the following command:
```bash
python separate.py CFG_SEPARATE
```

#### Monitoring the log
For each the above experiments, there is a log file in the experiment folder within the `results` folder. 
