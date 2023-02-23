#Written by Chad Wiley 
#This is used for getting the 2d data with all ensemble members
#doing the stats and making the dataset into tf ds

import xarray as xr
import numpy as xr
import glob
from data_2d_helper import *

#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num

if run_num == 0:
    #do 2017-2018
    examples_path_2017 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/2017-2018/00*')
    labels_path_2017 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/2017-2018/00*')

    examples_path_2017.sort()
    labels_path_2017.sort()

    extract_ex_data(examples_path_2017,'2017-2018')
    extract_label_data(labels_path_2017, '2017-2018')

if run_num == 1:
    #do 2019
    examples_path_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/2019/00*')
    labels_path_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/2019/00*')

    examples_path_2019.sort()
    labels_path_2019.sort()

    extract_ex_data(examples_path_2019,'2019')
    extract_label_data(labels_path_2019, '2019')

if run_num == 2:
    #do 2020
    examples_path_2020_train = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/000*')
    labels_path_2020_train = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/000*')

    examples_path_2020_test = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/001*')
    labels_path_2020_test = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/001*')

    examples_path_2020_train.sort()
    labels_path_2020_train.sort()

    examples_path_2020_test.sort()
    labels_path_2020_test.sort()


    extract_ex_data(examples_path_2020_train,'2020_train')
    extract_label_data(labels_path_2020_train, '2020_train')

    extract_ex_data(examples_path_2020_test,'2020_test')
    extract_label_data(labels_path_2020_test, '2020_test')

if run_num == 3:
    #do 2021
    examples_path_2021_train = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/000*')
    labels_path_2021_train = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/000*')

    examples_path_2021_test = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/001*')
    labels_path_2021_test = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/001*')

    examples_path_2021_train.sort()
    labels_path_2021_train.sort()

    examples_path_2021_test.sort()
    labels_path_2021_test.sort()

    extract_ex_data(examples_path_2021_train,'2021_train')
    extract_label_data(labels_path_2021_train, '2021_train')

    extract_ex_data(examples_path_2021_test,'2021_test')
    extract_label_data(labels_path_2021_test, '2021_test')

print('done')

