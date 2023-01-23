"""
This takes in a patch and compresses
it down into the maxes and averages of
each ensemble member. It also then changes
each data type from a float64 to a float32.
Finally puts the data into a netCDF4 file.

Written by Chad Wiley

"""

#imports
from importlib.resources import path
import numpy as np
import xarray as xr
import netCDF4 as nc
import os
import tensorflow as tf
import sys
import argparse

def create_parser():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--file_num', type=int, nargs='+', default = None)
    parser.add_argument('--data_path', type=str, default = None)
    parser.add_argument('--save_path',type=str, default = None )
    parser.add_argument('--run_num', type=int, default=None)

    return parser

def load_data(file_num, data_path):
    """
    Load the current dataset.

    Parameters
    --------------
    file_num : int
        This is the file that will be worked on

    data_path : str
        Path of the data

    Returns
    ---------------
    ds : Xarray dataset
        Loaded dataset
    """

    ds = xr.load_dataset(data_path + '/%d.nc'%file_num)

    return ds


def compress_data(ds_examples):
    """
    This compresses the data.
    file1 is the examples,
    it has the comp_dz, w_up,
    w_down, cape_ml, cin_ml,
    cape_sfc, cin_sfc.

    Parameters
    ------------------
    ds_examples : xarray dataset
        This take in the examples and compresses
        down to a single ensemble average/min/max
        and cast the numbers into a float 32

    Returns
    --------------------
    ds_examples : xarray dataset
        This is the new compressed version of example
        data
    """
    #Environmental Variables
    cape_ml = np.float32(np.average(ds_examples['cape_ml'], axis=0))
    cape_sfc = np.float32(np.average(ds_examples['cape_sfc'], axis=0))
    cin_ml = np.float32(np.average(ds_examples['cin_ml'], axis=0))
    cin_sfc = np.float32(np.average(ds_examples['cin_sfc'], axis=0))

    #Storm Variables
    #max
    w_up_max = np.float32(np.max(ds_examples['w_up'][1], axis=0))
    w_down_max = np.float32(np.min(ds_examples['w_down'], axis=0))
    comp_dz_max = np.float32(np.max(ds_examples['comp_dz'], axis=0))

    #90th percentile
    w_up_90 = np.float32(np.percentile((ds_examples['w_up'][0]), 90.))
    w_down_90 = np.float32(np.percentile((ds_examples['w_down'][0]), 90.))
    comp_dz_90 = np.float32(np.percentile(ds_examples['comp_dz'][1],90.))

    #averages
    w_up_avg = np.float32(np.average(ds_examples['w_up'], axis=1))
    w_down_avg = np.float32(np.average(ds_examples['w_down'], axis=1))
    comp_dz_avg = np.float32(np.average(ds_examples['comp_dz'], axis=1))

    print('done converting to float32 data')

    #Normalize the Data
    cape_ml = min_max_norm(cape_ml)
    cape_sfc = min_max_norm(cape_sfc)
    cin_ml = min_max_norm(cin_ml)
    cin_sfc = min_max_norm(cin_sfc)

    w_up_max = min_max_norm(w_up_max)
    w_down_max = min_max_norm(w_down_max)
    comp_dz_max = min_max_norm(comp_dz_max)

    w_up_90 = min_max_norm(w_up_90)
    w_down_90 = min_max_norm(w_down_90)
    comp_dz_90 = min_max_norm(comp_dz_90)

    w_up_avg = min_max_norm(w_up_avg)
    w_down_avg = min_max_norm(w_down_avg)
    comp_dz_avg = min_max_norm(comp_dz_avg)


    print('Done Normalizing')

    vars = [cape_ml, cape_sfc, cin_ml, cin_sfc, comp_dz_max, comp_dz_90, comp_dz_avg,
            w_up_max, w_up_90, w_up_avg, w_down_max, w_down_90, w_down_avg]

    names =['cape_ml', 'cape_sfc', 'cin_ml', 'cin_sfc', 'comp_dz_max' , 'comp_dz_90', 'comp_dz_avg',
            'w_up_max', 'w_up_90', 'w_up_avg', 'w_down_max', 'w_down_90', 'w_down_avg']

    size = ['n_samples', 'NY', 'NX']


    data_1 = {n : (size, v) for n,v in zip(names, vars)}
    out_ds = xr.Dataset(data_1)

    return out_ds

def min_max_norm(variable):
    """
    This does min_max scaling on the
    given variable.
    """
    variable = (variable - np.min(variable)) / (np.max(variable) - np.min(variable))
    return variable


def split_data_examples(ds):
    """
    FIX
    Takes in a dataset and splits it into 5x5
    or 4x4 depending on year (due to different domain sizes)
    """
    ds = ds.to_array()
    ds = ds.to_numpy()

    ds1 = np.split(ds, 2, axis = 2)
    ds2 = np.split(ds1[0], 2, axis = 3)
    ds3 = np.split(ds1[1], 2, axis = 3)

    gh = ds2+ds3
    print(np.shape(gh))

    ds4 = np.concatenate([ds2,ds3])
    ds4t = np.moveaxis(ds4, [1],[2])
    ds5 = np.reshape(ds4t, ((np.size(ds4t, axis = 0)* np.size(ds4t, axis = 1)),8,150,150))

    #ds5 = np.reshape(ds4, ((np.size(ds4, axis = 0)* np.size(ds, axis = 1)),8,150,150))

    ds6 = ds5[:,:,11:139,11:139]
    print(np.shape(ds6))

    return ds6
    
def split_data_labels(ds):
    """
    Takes in a dataset and splits it into
    four equal parts
    """
    ds = ds.to_array()
    ds = ds.to_numpy()

    ds1 = np.split(ds, 2, axis = 2)
    ds2 = np.split(ds1[0], 2, axis = 3)
    ds3 = np.split(ds1[1], 2, axis = 3)

    ds4 = np.concatenate([ds2,ds3])
    ds4t = np.moveaxis(ds4, [1],[2])
    ds5 = np.reshape(ds4t, ((np.size(ds4t, axis = 0) * np.size(ds4t, axis = 1)),1,150,150))

    #ds5 = np.reshape(ds4, ((np.size(ds4, axis = 0) * np.size(ds, axis = 1)),1,150,150))

    ds6 = ds5[:,:,11:139,11:139]
    print(np.shape(ds6))

    return ds6



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print('started')

    #get args
    file_num = args.file_num
    file_num = file_num[0]
    data_path = args.data_path
    save_path =args.save_path

    #load in the dataset
    ds = load_data(file_num, data_path)
    print('Loaded : %d'%file_num)

    #if examples, compress to ensemble averages
    #if labels, drop unneeded key
    if 'examples' in data_path:
        print('Compressing file')
        ds = compress_data(ds)
    else:
        print('Dropping Keys')
        ds = drop_keys(ds, file_num)
    
    print('saving...')
    ds.to_netcdf(save_path + '/%d.nc'%file_num)
    print("Finished")
