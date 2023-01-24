#helper functions for the 3d_data_creator.py

import glob
import numpy as np
import xarray as xr
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

def min_max_norm(variable):
    """
    This does min_max scaling on the
    given variable.
    """
    variable = (variable - np.min(variable)) / (np.max(variable) - np.min(variable))
    return variable

def get_2017_2018_data():
    #get all files for 2017-2018
    example_files_2017_2018 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples/*')

    example_files_2017_2018.sort()


    #load in each of the files and manipulate them
    count = 0
    for i in example_files_2017_2018:
        cur_file = xr.load_dataset(i)
        print('Loaded : %s'%i)

        #if first instance set examples to this dataset, else add the new examples on the n_samples dim
        if count == 0:
            examples = extract_data(cur_file,'2017')
        else:
            examples = xr.concat([examples, extract_data(cur_file,'2017')], dim = 'n_samples')

        print(np.shape(examples['comp_dz_max']))
        
        count = count + 1

    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2017-2018examples.nc')
    print('Saved examples for 2017-2018')

def get_2019_data():
    #get all files for 2017-2018
    example_files_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples_2019/*')

    example_files_2019.sort()

    #load in each of the files and manipulate them
    count = 0
    for i in example_files_2019:
        cur_file = xr.load_dataset(i)
        print('Loaded : %s'%i)

        #if first instance set examples to this dataset, else add the new examples on the n_samples dim
        if count == 0:
            examples = extract_data(cur_file,'2019')
        else:
            examples = xr.concat([examples, extract_data(cur_file,'2019')], dim = 'n_samples')
        print(np.shape(examples['comp_dz_max']))
        
        count = count + 1

    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2019examples.nc')
    print('Saved examples for 2019')

def get_2020_2021_data():
    #get all files for 2017-2018
    example_files_2020_2021 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/examples_2020-2021/*')

    example_files_2020_2021.sort()

    #load in each of the files and manipulate them
    count = 0
    for i in example_files_2020_2021:
        cur_file = xr.load_dataset(i)
        print('Loaded : %s'%i)

        #if first instance set examples to this dataset, else add the new examples on the n_samples dim
        if count == 0:
            examples = extract_data(cur_file,'2021')
        else:
            examples = xr.concat([examples, extract_data(cur_file,'2021')], dim='n_samples')
        print(np.shape(examples['comp_dz_max']))
        
        count = count + 1

    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2020_2021examples.nc')
    print('Saved examples for 2020-2021')


def extract_data(cur_file, year):
    '''
    cur_file : xr file that is already loaded into
               the program

    returns all the data with the wanted ensemble stats performed
    '''

    comp_dz_max = []
    comp_dz_90 = []
    comp_dz_avg = []

    w_up_max = []
    w_up_90 = []
    w_up_avg = []

    #THESE NUMBERS ARE NEGATIVE
    w_down_max = []
    w_down_90 = []
    w_down_avg = []

    cape_sfc_avg = []
    cape_ml_avg = []

    cin_sfc_avg = []
    cin_ml_avg = []

    mrms = []

    #perform model stats on ensemble models
    for i in range(np.size(cur_file['comp_dz'], axis = 0)):
        comp_dz_max.append(np.max(cur_file['w_up'][i], axis = 1))
        comp_dz_90.append(np.percentile(cur_file['w_up'][i], 90., axis = 1))
        comp_dz_avg.append(np.average(cur_file['w_up'][i], axis = 1))

        w_up_max.append(np.max(cur_file['w_up'][i], axis = 1))
        w_up_90.append(np.percentile(cur_file['w_up'][i], 90., axis = 1))
        w_up_avg.append(np.average(cur_file['w_up'][i], axis = 1))

        w_down_max.append(np.min(cur_file['w_down'][i], axis = 1))
        w_down_90.append(np.percentile(cur_file['w_down'][i], 10., axis = 1))
        w_down_avg.append(np.average(cur_file['w_down'][i], axis = 1))

        cape_sfc_avg.append(np.average(cur_file['cape_sfc'][i], axis = 1))
        cape_ml_avg.append(np.average(cur_file['cape_ml'][i], axis = 1))

        cin_sfc_avg.append(np.average(cur_file['cin_sfc'][i], axis = 1))
        cin_ml_avg.append(np.average(cur_file['cin_ml'][i], axis = 1))

        if year == '2017' or year =='2018' or year =='2019':
            mrms.append(cur_file['DZ_CRESSMAN'][i])
        else:
            mrms.append(cur_file['dz_cress'][i])

        #apply normalization
        comp_dz_max_norm = min_max_norm(comp_dz_max)
        comp_dz_90_norm = min_max_norm(comp_dz_90)
        comp_dz_avg_norm = min_max_norm(comp_dz_avg)

        w_up_max_norm = min_max_norm(w_up_max)
        w_up_90_norm = min_max_norm(w_up_90)
        w_up_avg_norm = min_max_norm(w_up_avg)

        w_down_max_norm = min_max_norm(w_down_max)
        w_down_90_norm= min_max_norm(w_down_90)
        w_down_avg_norm = min_max_norm(w_down_avg)

        cape_sfc_avg_norm = min_max_norm(cape_sfc_avg)
        cape_ml_avg_norm = min_max_norm(cape_ml_avg)

        cin_sfc_avg_norm = min_max_norm(cin_sfc_avg)
        cin_ml_avg_norm = min_max_norm(cin_ml_avg)

        mrms_norm = min_max_norm(mrms)

    #make in xr dataset
    vars = [comp_dz_max_norm,comp_dz_90_norm,comp_dz_avg_norm,
            w_up_max_norm, w_up_90_norm, w_up_avg_norm,
            w_down_max_norm, w_down_90_norm, w_down_avg_norm,
            cape_sfc_avg_norm, cape_ml_avg_norm, cin_sfc_avg_norm,
            cin_ml_avg_norm, mrms_norm]
        
    names =['comp_dz_max','comp_dz_90','comp_dz_avg',
            'w_up_max', 'w_up_90', 'w_up_avg',
            'w_down_max', 'w_down_90', 'w_down_avg',
            'cape_sfc_avg', 'cape_ml_avg', 'cin_sfc_avg',
            'cin_ml_avg', 'mrms']

    size = ['n_samples','time','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)
    
    #return each variable as (n_samples, time_dim, lat_dim, lon_dim)
    return out_ds

def merge_examples():
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2017-2018examples.nc')
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2019examples.nc')
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2017-2018examples.nc')
    print('Data is loaded')

    examples = xr.concat([ex_2017,ex_2019,ex_2020],dim='n_samples')

    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/full_examples.nc')




















    

