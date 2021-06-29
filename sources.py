#!/usr/bin/env python
import os, sys
import utils as ut

# path setting
lst_args = sys.argv[1:]
if len(lst_args) == 0:
    lst_args = ["settings/sources.ini"]
assert len(lst_args)==1, "Not a valid argument"
_, name_suffix = lst_args[0].split('.')
assert name_suffix == 'ini', "Not a valid config"
path_setting = lst_args[0]
assert os.path.exists(path_setting), "Config doesn't exists!"

def main():

    # settings
    args = ut.settings(path_setting)

    # save sources
    args.dataset.sources(args)

if __name__ == "__main__":
    main()
