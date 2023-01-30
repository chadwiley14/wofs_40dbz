import glob
import xarray as xr
import numpy as np
#this is to figure out which data set is not correct

labels = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_full/labels_20*')
examples = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples_full/*examples.nc')

labels.sort()
examples.sort()

for n in range(np.size(labels)):
    print('loading in examples : %s'%examples[n])
    print('loading in label : %s'%labels[n])
    cur_examples = xr.load_dataset(examples[n])
    cur_labels = xr.load_dataset(labels[n])

    print('------------------------------------')
    print(cur_examples)
    print('------------------------------------')
    print(cur_labels)
    print('------------------------------------')
    print('------------------------------------')
    print('------------------------------------')
    print()

