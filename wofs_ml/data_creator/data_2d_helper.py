#Written by Chad Wiley

#Helper functions for the data_2d_creator


import numpy as np
import xarray as xr
import glob
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


def extract_ex_data(ds_list,year):
    """
    Takes in a  list of dataset from a glob find.
    Loads in each ds, grabs the needed information, and outputs it.
    Saves the comp_dz still with all ensemble members which is needed for
    getting ensemble probs and doing gaussian filters on. With all wanted data
    for training/validation/testing, the wanted information is extracted,
    and output to a tensorflow dataset
    
    PARAMETERS
    -----------------
    ds_list : list
        A list of str with paths to all netcdf files for a given year
    
    year : str
        Year used for data extraction and saving
        
    """
    comp_dz = []

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

    lats = []
    lons = []

    for cur_ex in ds_list:
        #load in the file
        cur_file = xr.load_dataset(cur_ex)

        #get the comp_dz for probs
        
        comp_dz.append(cur_file['comp_dz'])
        lats.append(cur_file['lat'])
        lons.append(cur_file['lon'])

        #model stats
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

            if year == '2017-2018' or year =='2019':
                mrms.append(cur_file['DZ_CRESSMAN'][i])
            else:
                mrms.append(cur_file['dz_cress'][i])

    #Not normalized until all dataset are together in one place!
    #make in xr dataset
    vars = [comp_dz_max,comp_dz_90,comp_dz_avg,
            w_up_max, w_up_90, w_up_avg,
            w_down_max, w_down_90, w_down_avg,
            cape_sfc_avg, cape_ml_avg, cin_sfc_avg,
            cin_ml_avg, mrms]
        
    names =['comp_dz_max','comp_dz_90','comp_dz_avg',
            'w_up_max', 'w_up_90', 'w_up_avg',
            'w_down_max', 'w_down_90', 'w_down_avg',
            'cape_sfc_avg', 'cape_ml_avg', 'cin_sfc_avg',
            'cin_ml_avg', 'mrms']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print('Examples : ',out_ds)
    print('----------------------')
    print('----------------------')
    print('----------------------')

    #save each variable as (n_samples, lat_dim, lon_dim) not normalized
    if year == '2017-2018' or year == '2019':
        #save to training split
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_%s.nc'%year)
    elif year == '2020':
        #save to validation
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_%s.nc'%year)
    elif year == '2021':
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_%s.nc'%year)
    print('dataset is saved')

    #save comp_dz
    save_comp_prob(comp_dz,year)
    print('saved probs')

    #save lat_lons
    save_lat_lon(lats,lons,year)
    print('saved lat and lons')

def extract_label_data(list_files, year):
    """
    Takes in a glob file of label data for that year. Loads
    in each ds and gets the wanted information from them then save it
    to a netcdf.

    PARAMETERS
    ---------------
    list_files : list
        list of str with all the paths to the files

    year : str
        string of the year for saving purpose
    """
    dz_binary = []
    for cur_label in list_files:
        #load in the file
        cur_file = xr.load_dataset(cur_label)

        for i in range(np.size(cur_file['dz_cress_binary'], axis = 0)):
            dz_binary.append(cur_file['dz_cress_binary'][i])

    vars = [dz_binary]

    name = ['dz_binary']


    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]

    data_vars = {name : data_tup for name, data_tup in zip(name, tuples)}

    
    out_ds = xr.Dataset(data_vars)
    print('label : ',out_ds)
    print('-------------------')
    print('-------------------')
    print('-------------------')
    if year == '2017-2018' or year == '2019':
        #save to training split
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_%s.nc'%year)
    elif year == '2020':
        #save to validation
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/labels/labels_%s.nc'%year)
    elif year == '2021':
        out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/labels/labels_%s.nc'%year)

def save_lat_lon(full_lats, full_lons, year):
    """
    Takes in a list of all the lat and lons and
    saves them to a netcdf file based on year

    PARAMETERS
    --------------------
    full_lats : list
        list of all the lats from that year
    
    full_lons : list
        list of all the lons from that year

    year : str
        year for saving purposes
    """
    #make in xr dataset
    vars = [full_lats,full_lons]
        
    names =['lats','lons']

    size = ['n_samples','lat_dim', 'lon_dim']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print('lat_lons',out_ds)
    print('----------------------')
    print('----------------------')
    print('----------------------')

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_%s.nc'%year)



def save_comp_prob(comp_dz, year):
    """
    Takes in a list of comp_dz with dim of n_samples, ne, lat_dim,lon_dim
    and saves it to netcdf
    """

    #make in xr dataset
    vars = [comp_dz]
        
    names =['comp_dz_max']

    size = ['n_samples','ne','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print('comp_dz',out_ds)
    print('----------------------')
    print('----------------------')
    print('----------------------')

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/probs/comp_dz_probs_%s.nc'%year)


