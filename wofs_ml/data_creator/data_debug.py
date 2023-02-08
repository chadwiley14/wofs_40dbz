import glob
import xarray as xr
import numpy as np
#this is to figure out which data set is not correct

#steps in label procress
labels_patcher = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/labels/0000.nc')
labels_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2017-2018.nc')
labels_fix = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2017-2018_fix.nc')

print('From patching:')
print(np.max(labels_patcher['dz_cress']))
#print(labels_patcher)
print('---------------------')

print('Patches in one place:')
print(np.max(labels_2020['dz_binary']))
#print(labels_2020)
print('---------------------')

print('Patches with fix')
print(np.max(labels_fix['dz_binary']))
#print(labels_fix)
print('---------------------')


