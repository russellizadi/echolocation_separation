#!/usr/bin/env python
import os
import utils as ut

# path setting
name_script, name_suffix = os.path.basename(__file__).split('.')
assert name_suffix == 'py', "Not a Python script!"
PATH_SETTING = 'settings/{}.ini'.format(name_script)

def main():

    # settings
    args = ut.settings(PATH_SETTING)

    # save sources
    args.dataset.sources(args)

if __name__ == "__main__":
    main()
