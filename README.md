# Echolocation Separation

For each of the following experiments, there is one python file `.py`. Associated with each script, a config file `.ini` is available in the `settings` folder. Change the requiered config before running scripts. All results associated with each experiment will be in a folder inside the `results` folder with the same name as the config file.

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
#### Using the detection model

#### Training and evaluating a model
If you have a pretrained model, you can skip this part. After mixing sources, change the `CFG_EVALUATE` in the `settings` folder and set `name_mixtures` to the appropriate folder name containing train and test mixtures. You can set the number of folds through the `num_folds` variable.

To train and test a model, run the following command:
```

```
