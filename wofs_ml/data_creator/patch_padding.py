import xarray as xr
import numpy as np
import glob
from patch_padding_helper import *


#load in things for examples, done in a for loop since 2021-2019 are 300x300
#and 2018/2017 domain size is 250x250. This requires two different patch
#sizes. For 2021-2019, a 5x5 quilt of 64x64, and a 4x4 quilt of 64x64
#for 2018/2017

parser = create_parser() #from data_manipulation.py
args = parser.parse_args()

run_num = args.run_num

if run_num == 0:
    #Do ENS
    ENS = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/ENS_*')

    ENS.sort()
    print('starting ENS')
    ens_patches= break_into_patches(ENS)
    print(np.shape(ens_patches))

    save_ens(ens_patches)
    print('done ENS')

elif run_num == 1:
    #do SVR
    SVR = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/SVR_*')
    SVR.sort()

    print('starting SVR')
    svr_patches= break_into_patches(SVR)
    print(np.shape(svr_patches))

    save_svr(svr_patches)
    print('done SVR')

elif run_num == 2:
    #do init
    init = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/init_*')
    init.sort()

    print('starting Init')
    init_patches = break_into_patches(init)
    print(np.shape(init_patches))

    save_init(init_patches)
    print('done init')

elif run_num == 3:
    #do val
    val = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/val_*')
    val.sort() 

    print('starting Val')
    val_patches = break_into_patches(val)
    print(np.shape(val_patches))

    save_val(val_patches)
    print('done val')

elif run_num == 4:
    #do lat-lon
    lat_lon = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/lat_*')
    lat_lon.sort()

    print('starting lat-lon')
    lat_lon_patches = (break_into_patches(lat_lon))
    print(np.shape(lat_lon_patches))
    print()

    save_lat_lon(lat_lon_patches)
    print('done lat-lons')

elif run_num == 5:
    #save examples and labels
    ens_patches1 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/ens_patches.nc')
    svr_patches1 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/svr_patches.nc')
    init_patches1 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/init_patches.nc')
    val_patches1 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/val_patches.nc')

    print('merging examples...')
    examples = xr.merge([ens_patches1,svr_patches1,init_patches1])
    print(examples)
    print('merging labels...')
    labels = val_patches1.drop_vars('dz_cress')
    print(labels)

    print('saving examples...')
    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/examples.nc')
    print('saving labels...')
    labels.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/labels.nc')
    print('saved examples and labels')

elif run_num == 6:
    #make probs into patches 2017
    probs_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_2017_06.nc')

    make_wofs_patches(probs_2017,'2017')
elif run_num == 7:
    #make probs into patches 2017
    probs_2018 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_2018_06.nc')

    make_wofs_patches(probs_2018,'2018')
elif run_num == 8:
    #make probs into patches 2017
    probs_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_2019_06.nc')

    make_wofs_patches(probs_2019,'2019')

elif run_num == 9:
    #make probs into patches 2017
    probs_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_2020_06.nc')

    make_wofs_patches(probs_2020,'2020')

elif run_num == 10:
    #make probs into patches 2017
    probs_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_2021_06.nc')

    make_wofs_patches(probs_2021,'2021')

print('Done')



