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
    '''
    Gets all the data from the 2017-2018 years and
    runs ensemble stats on it and returns all the data
    in a xr data set.
    '''
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
    '''
    Gets all the data from the 2019 and
    runs ensemble stats on it and returns all the data
    in a xr data set.
    '''
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
    '''
    Gets all the data from the 2020-2021 years and
    runs ensemble stats on it and returns all the data
    in a xr data set.
    '''
    #get all files for 2020-2021
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
    Takins in the current xr file of data will all ensemble members
    and computes ensemble stats on each of the desired variables.

    PARAMETERS
    --------------------
    cur_file : xr file that is already loaded into
               the program
    year : the year of the file. needed due to different naming convection.

    RETURNS
    --------------------
    All the data with the wanted ensemble stats performed in an xr dataset.
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
    '''
    Takes in all the examples from the years of interest and makes them into
    one dataset for training.
    '''
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2017-2018examples.nc')
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2019examples.nc')
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/2017-2018examples.nc')
    print('Data is loaded')

    examples = xr.concat([ex_2017,ex_2019,ex_2020],dim='n_samples')

    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/full_examples.nc')

def extract_labels(cur_file):
    '''
    Extracts the label data for training. Goes through each sample and gets the
    wanted information. Meant to be used in conjenction with get_labels

    PARAMETER
    -----------------
    cur_file  : xr dataset, will have the dim of (n_samples,time,lat,lon)

    RETURNS
    ------------
    The data in a xr dataset
    '''
    dz_binary = []

    for i in range(np.size(cur_file['dz_cress_binary'], axis = 0)):
        dz_binary.append(cur_file['dz_cress_binary'][i])

    vars = [dz_binary]

    name = ['dz_binary']


    size = ['n_samples','time','lat', 'lon']

    tuples = [(size,var)for var in vars]

    data_vars = {name : data_tup for name, data_tup in zip(name, tuples)}

    
    out_ds = xr.Dataset(data_vars)

    return out_ds

def get_lat_lon(cur_file, year):
    '''
    Extracts the lat and lons for plotting. Meant to be used in
    conjunction with get_labels.

    PARAMETERS
    -----------------

    cur_file : xr dataset
        will have dim of (n_samples, lat,lon)

    RETURNS
    ------------------
    The current file in an xr dataset.
    '''
    lats = []
    lons = []

    for i in range(np.size(cur_file['dz_cress_binary'], axis = 0)):
        lats.append(cur_file['lat'][i])
        lons.append(cur_file['lon'][i])

    vars = [lats,lons]

    name = ['lats', 'lons']

    size= ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]


    data_vars = {name : data_tup for name, data_tup in zip(name, tuples)}

    
    out_ds = xr.Dataset(data_vars)


    return out_ds

def get_labels(glob_files, year):
    '''
    Takes in the glob files from a certain year and 
    uses the helper functions to extract the label and 
    lat-lon data from the year(s) of interest

    PARAMETERS
    ----------------
    glob_files : list of str
        All files that match the wanted files

    year : str
        Year is used for saving path
    '''
    count = 0
    for i in glob_files:
        cur_file = xr.load_dataset(i)
        print('Loaded : %s'%i)

        #if first instance set examples to this dataset, else add the new examples on the n_samples dim
        if count == 0:
            labels = extract_labels(cur_file)
            lat_lon = get_lat_lon(cur_file, year)
        else:
            labels = xr.concat([labels, extract_labels(cur_file)], dim = 'n_samples')
            lat_lon = xr.concat([lat_lon, get_lat_lon(cur_file, year)], dim = 'n_samples')


        print('labels shape: %s'%str(np.shape(labels['dz_binary'])))
        print('labels shape: %s'%str(np.shape(labels['dz_binary'])))
        
        count = count + 1

    #save the data
    labels.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_%s.nc'%year)
    lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/lat_lon_%s.nc'%year)




















    

