import glob 
import tensorflow as tf 
import xarray as xr 
import numpy as np 
import gc 
import tqdm 
from dask.diagnostics import ProgressBar
import pandas as pd
import copy
import sys 

print('grabbing files')
files = glob.glob('/scratch/randychase/testing_2020_cmb_*.nc')
files.sort()

# Arguments passed
input_array_id = int(sys.argv[1])

train_ds = xr.open_dataset(files[input_array_id],chunks={'n_samples':10})

# #calc min max for each one 
# print('Computing Max')
# with ProgressBar():
#     z_max = train_ds.z_patch.max().compute()
# print('Computing Min')
# with ProgressBar():
#     z_min = train_ds.z_patch.min().compute()

# print('Max: {}; Min: {};'.format(z_max,z_min))
    
# train_ds.close()

#Old 2d method fill val 
#fill nans with 0s 
# train_ds = train_ds.fillna(0.0)

# fill nans with minx (its so this is bascially 0 in the scaled space)
# train_ds = train_ds.fillna(-51)

#bring into memory 
train_images = train_ds.z_patch.fillna(-39.7).astype(np.float16).values 

#check max value
print(train_images.max(),train_images.min())

# #check for infs.
print(np.where(np.isinf(train_images)))

# # #scale inputs to 0 - 1, unstable otherwise, loss goes to infinity 
# # # mu = np.mean(train_images[train_images != 0.0])
# # # sigma = np.std(train_images[train_images != 0.0])
# # # print('SCALARS: {} , {}'.format(mu,sigma))
# # # train_images = (train_images - mu)/sigma

# #old factors (for 2d models)
# # maxx = 85.321304
# # minx = -33.83131

# #new factors 
# maxx = 90.338135
# minx = -51.577034

#newnew factors
maxx = 85.8
minx = -39.7

train_images = (train_images - minx) / (maxx - minx)

# #check max value
print(train_images.max(),train_images.min())

#load labels into memory
train_labels = train_ds.w_patch.astype(np.float16).values 

# #check max value
print(train_labels.max(),train_labels.min())

#clear up RAM 
train_ds.close()

#make tensorflow dataset 
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

#dump to disk 
tf.data.experimental.save(train_dataset, files[input_array_id][:-2]+ 'tf')

del train_dataset, train_images, train_labels,train_ds

gc.collect()


#load in the examples from ourdisk
print('grabbing files')
examples_nc = ('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples_full/full_examples.nc')
labels_nc = ('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_full/labels_full.nc')

# Arguments passed
input_array_id = int(sys.argv[1])

train_ex_ds = xr.load_dataset(examples_nc,chunks={'n_samples':2689})
train_lb_ds = xr.load_dataset(labels_nc, chunks = {'n_samples':2689})
#split into train/valid/test (13448,1681,1681)
#split into train into 5 seperate tf ds (2689)
#split val into 1
#split testing into 1

