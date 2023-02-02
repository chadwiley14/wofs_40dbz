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
    examples_path_2017 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/2017-2018/00*')
    labels_path_2017 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/2017-2018/00*')

    examples_path_2017.sort()
    labels_path_2017.sort()

    extract_ex_data(examples_path_2017,'2017-2018')
    extract_label_data(labels_path_2017, '2017-2018')

if run_num == 1:
    #do 2019
    examples_path_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/2019/00*')
    labels_path_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/2019/00*')

    examples_path_2019.sort()
    labels_path_2019.sort()

    extract_ex_data(examples_path_2019,'2019')
    extract_label_data(labels_path_2019, '2019')

if run_num == 2:
    #do 2020
    examples_path_2020 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/00*')
    labels_path_2020 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/labels/00*')

    examples_path_2020.sort()
    labels_path_2020.sort()

    extract_ex_data(examples_path_2020,'2020')
    extract_label_data(labels_path_2020, '2020')

if run_num == 3:
    #do 2021
    examples_path_2021 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/00*')
    labels_path_2021 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/labels/00*')

    examples_path_2021.sort()
    labels_path_2021.sort()

    extract_ex_data(examples_path_2021,'2021')
    extract_label_data(labels_path_2021, '2021')

print('done')

