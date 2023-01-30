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
import argparse
def create_parser():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--run_num', type=int, default=None)

    return parser

#create parser
parser = create_parser()
args = parser.parse_args()

input_array_id = args.run_num

#load in the examples from ourdisk
print('grabbing files')
examples_nc = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples_full/20*')
labels_nc = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_full/labels_20*')

examples_nc.sort()
labels_nc.sort()

year = ['2017-2018', '2019', '2020-2021']

for n in range(np.size(examples_nc)):
    #load in the files
    print('example : %s'%examples_nc[n])
    print('labels : %s'%labels_nc[n])
    cur_examples = xr.load_dataset(examples_nc[n])
    cur_labels = xr.load_dataset(labels_nc[n])

    #make into arrays
    cur_examples=cur_examples.to_array()
    cur_labels=cur_labels.to_array()


    cur_examples = cur_examples.transpose('n_samples',...)
    cur_examples = cur_examples.transpose(...,'variable')

    cur_labels = cur_labels.transpose('n_samples',...)
    cur_labels = cur_labels.transpose(...,'variable')

    training_ds = tf.data.Dataset.from_tensor_slices((cur_examples,cur_labels))

    tf.data.experimental.save(training_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/tf_ds/training_ds_%s.tf'%year[n])
    print('saved %s'%year)


    

# first_indice = input_array_id*2689
# second_indice = first_indice + 2689
# print('first %d'%first_indice)
# print('second %d'%second_indice)

# #load in the chunks
# ex_ds = xr.load_dataset(examples_nc, chunks = {'n_samples' :2689})
# lb_ds = xr.load_dataset(labels_nc, chunks = {'n_samples' :2689})

# print('examples : ', ex_ds)

# print('labels : ', lb_ds)


# #based on the run num it multiplies the run num by 2689 for the first
# #indices and adds 2689 to the first indice to get the second indice. Stop onces it
# #gets to 13448

# #get the slice
# if second_indice <= 13448:
#     #This is the training ds
#     ds_temp_ex = ex_ds.isel(n_samples=slice(first_indice,second_indice))
#     ds_temp_lb = lb_ds.isel(n_samples=slice(first_indice,second_indice))

#     ds_temp_ex = ds_temp_ex.to_array()
#     ds_temp_lb = ds_temp_lb.to_array()

#     #data must be in certain shape (n_samples,lat,lon,time,variable)
#     ds_temp_ex  = ds_temp_ex.transpose('n_samples',...)
#     ds_temp_ex = ds_temp_ex.transpose(...,'variable')

#     ds_temp_lb  = ds_temp_lb.transpose('n_samples',...)
#     ds_temp_lb = ds_temp_lb.transpose(...,'variable')


#     training_ds = tf.data.Dataset.from_tensor_slices((ds_temp_ex,ds_temp_lb))

#     #dump to disk
#     tf.data.experimental.save(training_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/tf_ds/training_ds_%d.tf'%input_array_id)
#     print('saved %d'%input_array_id)
# else:
#     #this is the validation and testing ds
#     ds_temp_ex_val = ex_ds.isel(n_samples=slice(13445,15129))
#     ds_temp_lb_val = lb_ds.isel(n_samples=slice(13445,15129))

#     ds_temp_ex_test = ex_ds.isel(n_samples=slice(15129,16810))
#     ds_temp_lb_test = lb_ds.isel(n_samples=slice(15129,16810))

#     ds_temp_ex_val = ds_temp_ex_val.to_array()
#     ds_temp_lb_val = ds_temp_lb_val.to_array()

#     ds_temp_ex_test = ds_temp_ex_test.to_array()
#     ds_temp_lb_test = ds_temp_lb_test.to_array()

#     #data must be in certain shape (n_samples,lat,lon,time,variable)
#     #validation
#     ds_temp_ex_val  = ds_temp_ex_val.transpose('n_samples',...)
#     ds_temp_ex_val = ds_temp_ex_val.transpose(...,'variable')

#     ds_temp_lb_val  = ds_temp_lb_val.transpose('n_samples',...)
#     ds_temp_lb_val = ds_temp_lb_val.transpose(...,'variable')

#     #testing
#     ds_temp_ex_test  = ds_temp_ex_test.transpose('n_samples',...)
#     ds_temp_ex_test = ds_temp_ex_test.transpose(...,'variable')

#     ds_temp_lb_test  = ds_temp_lb_test.transpose('n_samples',...)
#     ds_temp_lb_test = ds_temp_lb_test.transpose(...,'variable')

#     val_ds = tf.data.Dataset.from_tensor_slices((ds_temp_ex_val,ds_temp_lb_val))
#     test_ds = tf.data.Dataset.from_tensor_slices((ds_temp_ex_test,ds_temp_lb_test))

#     #dump to disk
#     tf.data.experimental.save(val_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/tf_ds/val_ds.tf')
#     tf.data.experimental.save(test_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/tf_ds/val_ds.tf')
#     print('saved val and testing')

