#Written by Chad Wiley
#This program takes in a variable and normalizes it
#Only done with examples since labels are binary

import numpy as np
import xarray as xr
import glob
import argparse
import tensorflow as tf

def create_parser():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--run_num', type=int, default=None)

    return parser

def get_min_max(variable):
    """
    Takes in a variable and prints out the min and max for that
    variable. Needed to accuratly create the normalization.

    IF variable is negative (i.e. CIN, or downdraft), must change flag.

    PARAMETERS
    ----------------
    variable : xr Dataarray
        The current variable that is getting the max and min
    """

    print('Max : %f'%np.max(variable))
    print('Min : %f'%np.min(variable))

def min_max_norm(variable, min, max):
    """
    Takes in a variable and returns that variable normalized
    between 0-1. This is needed due to the vast differences
    in scales between variables

    PARAMETERS
    -------------------
    variable : xr dataarray
        Variable wanted min max norm on
    
    RETURNS
    ---------------------
        Returns the input variable with 
        normalized numbers.
    """
    norm_variable = (variable - min) / (max - min)
    return norm_variable

def unreasonable_values(ds):
    """
    Takes in a ds and returns all the 
    cases where unreasonable numbers 
    exists.

    PARAMETERS
    ---------------------
    ds : xr dataset
        This is the 2d examples dataset

    RETURNS
    ---------------------
    problem_cases : list
        List of problem cases that need to be removed.
    """

    problem_cases = []

    for n in range(np.size(ds['comp_dz_max'], axis =0)):
        if np.any(ds['comp_dz_max'][n,...]>= 100) or np.any(ds['comp_dz_max'][n,...] < 0):
            problem_cases.append(n)

        elif np.any(ds['comp_dz_90'][n,...] >= 100) or np.any(ds['comp_dz_90'][n,...] < 0):
            problem_cases.append(n)

        elif np.any(ds['comp_dz_avg'][n,...] >= 100) or np.any(ds['comp_dz_avg'][n,...] < 0):
            problem_cases.append(n)

        elif np.any(ds['w_up_max'][n,...] >= 100) or np.any(ds['w_up_max'][n,...] < -10):
            problem_cases.append(n)

        elif np.any(ds['w_up_90'][n,...] >= 100) or np.any(ds['w_up_90'][n,...] < -10):
            problem_cases.append(n)

        elif np.any(ds['w_up_avg'][n,...] >= 100) or np.any(ds['w_up_avg'][n,...] < -10):
            problem_cases.append(n)

        elif np.any(ds['w_down_max'][n,...] > 10) or np.any(ds['w_down_max'][n,...] < -100):
            problem_cases.append(n)

        elif np.any(ds['w_down_90'][n,...] >10) or np.any(ds['w_down_90'][n] < -100):
            problem_cases.append(n)

        elif np.any(ds['w_down_avg'][n,...] > 10) or np.any(ds['w_down_avg'][n,...] < -100):
            problem_cases.append(n)

        elif np.any(ds['cape_ml_avg'][n,...] >= 10000) or np.any(ds['cape_ml_avg'][n,...] < -5):
            problem_cases.append(n)

        elif np.any(ds['cape_sfc_avg'][n,...] >= 10000) or np.any(ds['cape_sfc_avg'][n,...] < -5):
            problem_cases.append(n)

        elif np.any(ds['cin_ml_avg'][n,...] > 1) or np.any(ds['cin_ml_avg'][n,...] < -2000):
            problem_cases.append(n)

        elif np.any(ds['cin_sfc_avg'][n,...] > 1) or np.any(ds['cin_sfc_avg'][n,...] < -2000):
            problem_cases.append(n)

        elif np.any(ds['mrms'][n,...] >= 100) or np.any(ds['mrms'][n,...] < 0):
            problem_cases.append(n)
    # print(problem_cases)
    # print(np.size(problem_cases))
    # print(np.shape(ds['comp_dz_max']))
    return problem_cases



#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num

comp_dz_max_norm = []
comp_dz_90_norm = []
comp_dz_avg_norm = []

w_up_max_norm =[]
w_up_90_norm = []
w_up_avg_norm =[]

w_down_max_norm = []
w_down_90_norm = []
w_down_avg_norm = []

cape_ml_avg_norm =[]
cape_sfc_avg_norm = []

cin_ml_avg_norm = []
cin_sfc_avg_norm = []

mrms_norm = []

if run_num ==0:
    #get all the example files
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2017-2018_fix.nc')
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2019_fix.nc')
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_2020_fix.nc')
    ex_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_2021_fix.nc')

    print('all files loaded')

    full_ex = xr.concat([ex_2017,ex_2019,ex_2020,ex_2021],dim='n_samples')

    print('Comp_dz_max: ')
    get_min_max(full_ex['comp_dz_max'])
    print('-------------------------------')
    print()

    print('Comp_dz_90: ')
    get_min_max(full_ex['comp_dz_90'])
    print('-------------------------------')
    print()

    print('Comp_dz_avg: ')
    get_min_max(full_ex['comp_dz_avg'])
    print('-------------------------------')
    print()

    print('w_up_max: ')
    get_min_max(full_ex['w_up_max'])
    print('-------------------------------')
    print()

    print('w_up_90: ')
    get_min_max(full_ex['w_up_90'])
    print('-------------------------------')
    print()

    print('w_up_avg: ')
    get_min_max(full_ex['w_up_avg'])
    print('-------------------------------')
    print()
    
    print('w_down_max: ')
    get_min_max(full_ex['w_down_max'])
    print('-------------------------------')
    print()

    print('w_down_90: ')
    get_min_max(full_ex['w_down_90'])
    print('-------------------------------')
    print()

    print('w_down_avg: ')
    get_min_max(full_ex['w_down_avg'])
    print('-------------------------------')
    print()

    print('cape_sfc_avg: ')
    get_min_max(full_ex['cape_sfc_avg'])
    print('-------------------------------')
    print()

    print('cape_ml_avg: ')
    get_min_max(full_ex['cape_ml_avg'])
    print('-------------------------------')
    print()

    print('cin_sfc_avg: ')
    get_min_max(full_ex['cin_sfc_avg'])
    print('-------------------------------')
    print()

    print('cin_ml_avg: ')
    get_min_max(full_ex['cin_ml_avg'])
    print('-------------------------------')
    print()

    print('MRMS: ')
    get_min_max(full_ex['mrms'])
    print('-------------------------------')
    print()

def norm_data(variable,min,max):
    """
    Takes in a dataset and applies the min-max
    scaling to the variable and then saves the data
    in a tf dataset. Min and max are needed due to 
    being multiple datasets for this case.

    PARAMETERS
    -------------------
    variable : xr data array
        Variable wanting to apply min-max on
    
    min : float
        Min value of that variable across all ds

    max : float
        Max value of that variable across all ds
    """
    variable = min_max_norm(variablemin=min,max=max)


if run_num == 1:
    #do 2017-2018 examples
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2017-2018_fix.nc')

    for n in range(np.size(ex_2017['comp_dz_max'], axis=0)):
        comp_dz_max_norm.append(min_max_norm(variable=ex_2017['comp_dz_max'][n], min = 0., max= 77.749992))
        comp_dz_90_norm.append(min_max_norm(variable=ex_2017['comp_dz_90'][n], min = 0., max = 75.403117))
        comp_dz_avg_norm.append(min_max_norm(variable=ex_2017['comp_dz_avg'][n], min = 0., max = 72.208328))

        w_up_max_norm.append(min_max_norm(variable=ex_2017['w_up_max'][n], min = 0.,max = 96.656242))
        w_up_90_norm.append(min_max_norm(variable=ex_2017['w_up_90'][n], min = 0., max= 80.421867))
        w_up_avg_norm.append(min_max_norm(variable=ex_2017['w_up_avg'][n], min = 0., max = 66.955719))

        w_down_max_norm.append(min_max_norm(variable=ex_2017['w_down_max'][n], min = -96.734375, max = 0.090332))
        w_down_90_norm.append(min_max_norm(variable= ex_2017['w_down_90'][n], min = -64.167187, max= 0.155081))
        w_down_avg_norm.append(min_max_norm(variable=ex_2017['w_down_avg'][n], min= -45.243816, max = 0.220207))

        cape_ml_avg_norm.append(min_max_norm(variable=ex_2017['cape_ml_avg'][n],  min = 0., max = 5360.163574))
        cape_sfc_avg_norm.append(min_max_norm(variable=ex_2017['cape_sfc_avg'][n], min = 0., max = 7333.146484))

        cin_ml_avg_norm.append(min_max_norm(variable=ex_2017['cin_ml_avg'][n], min = -975.307373, max = 0.))
        cin_sfc_avg_norm.append(min_max_norm(variable=ex_2017['cin_sfc_avg'][n], min= -956.824463, max =0.))

        mrms_norm.append(min_max_norm(variable=ex_2017['mrms'][n], min=0., max=98.694305))


    #save the data to a tf ds
    #make in xr dataset
    vars = [comp_dz_max_norm, comp_dz_90_norm, comp_dz_avg_norm,
            w_up_max_norm, w_up_90_norm, w_up_avg_norm,
            w_down_max_norm, w_down_90_norm, w_down_avg_norm,
            cape_ml_avg_norm, cape_sfc_avg_norm,
            cin_ml_avg_norm, cin_sfc_avg_norm, mrms_norm]
        
    names =['comp_dz_max_norm', 'comp_dz_90_norm', 'comp_dz_avg_norm',
            'w_up_max_norm', 'w_up_90_norm', 'w_up_avg_norm',
            'w_down_max_norm', 'w_down_90_norm', 'w_down_avg_norm',
            'cape_ml_avg_norm', 'cape_sfc_avg_norm',
            'cin_ml_avg_norm', 'cin_sfc_avg_norm', 'mrms_norm']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2017-2018_norm.nc')
    print('saved to netcdf')


if run_num == 2:
    #load in 2019 data
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2019_fix.nc')

    for n in range(np.size(ex_2019['comp_dz_max'], axis=0)):
        comp_dz_max_norm.append(min_max_norm(variable=ex_2019['comp_dz_max'][n], min = 0., max= 77.749992))
        comp_dz_90_norm.append(min_max_norm(variable=ex_2019['comp_dz_90'][n], min = 0., max = 75.403117))
        comp_dz_avg_norm.append(min_max_norm(variable=ex_2019['comp_dz_avg'][n], min = 0., max = 72.208328))

        w_up_max_norm.append(min_max_norm(variable=ex_2019['w_up_max'][n], min = 0.,max = 96.656242))
        w_up_90_norm.append(min_max_norm(variable=ex_2019['w_up_90'][n], min = 0., max= 80.421867))
        w_up_avg_norm.append(min_max_norm(variable=ex_2019['w_up_avg'][n], min = 0., max = 66.955719))

        w_down_max_norm.append(min_max_norm(variable=ex_2019['w_down_max'][n], min = -96.734375, max = 0.090332))
        w_down_90_norm.append(min_max_norm(variable= ex_2019['w_down_90'][n], min = -64.167187, max= 0.155081))
        w_down_avg_norm.append(min_max_norm(variable=ex_2019['w_down_avg'][n], min= -45.243816, max = 0.220207))

        cape_ml_avg_norm.append(min_max_norm(variable=ex_2019['cape_ml_avg'][n],  min = 0., max = 5360.163574))
        cape_sfc_avg_norm.append(min_max_norm(variable=ex_2019['cape_sfc_avg'][n], min = 0., max = 7333.146484))

        cin_ml_avg_norm.append(min_max_norm(variable=ex_2019['cin_ml_avg'][n], min = -975.307373, max = 0.))
        cin_sfc_avg_norm.append(min_max_norm(variable=ex_2019['cin_sfc_avg'][n], min= -956.824463, max =0.))

        mrms_norm.append(min_max_norm(variable=ex_2019['mrms'][n], min=0., max=98.694305))


    #save the data to a tf ds
    #make in xr dataset
    vars = [comp_dz_max_norm, comp_dz_90_norm, comp_dz_avg_norm,
            w_up_max_norm, w_up_90_norm, w_up_avg_norm,
            w_down_max_norm, w_down_90_norm, w_down_avg_norm,
            cape_ml_avg_norm, cape_sfc_avg_norm,
            cin_ml_avg_norm, cin_sfc_avg_norm, mrms_norm]
        
    names =['comp_dz_max_norm', 'comp_dz_90_norm', 'comp_dz_avg_norm',
            'w_up_max_norm', 'w_up_90_norm', 'w_up_avg_norm',
            'w_down_max_norm', 'w_down_90_norm', 'w_down_avg_norm',
            'cape_ml_avg_norm', 'cape_sfc_avg_norm',
            'cin_ml_avg_norm', 'cin_sfc_avg_norm', 'mrms_norm']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2019_norm.nc')
    print('saved to netcdf')

if run_num == 3:
    #load in 2020 data
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_2020_fix.nc')

    for n in range(np.size(ex_2020['comp_dz_max'], axis=0)):
        comp_dz_max_norm.append(min_max_norm(variable=ex_2020['comp_dz_max'][n], min = 0., max= 77.749992))
        comp_dz_90_norm.append(min_max_norm(variable=ex_2020['comp_dz_90'][n], min = 0., max = 75.403117))
        comp_dz_avg_norm.append(min_max_norm(variable=ex_2020['comp_dz_avg'][n], min = 0., max = 72.208328))

        w_up_max_norm.append(min_max_norm(variable=ex_2020['w_up_max'][n], min = 0.,max = 96.656242))
        w_up_90_norm.append(min_max_norm(variable=ex_2020['w_up_90'][n], min = 0., max= 80.421867))
        w_up_avg_norm.append(min_max_norm(variable=ex_2020['w_up_avg'][n], min = 0., max = 66.955719))

        w_down_max_norm.append(min_max_norm(variable=ex_2020['w_down_max'][n], min = -96.734375, max = 0.090332))
        w_down_90_norm.append(min_max_norm(variable= ex_2020['w_down_90'][n], min = -64.167187, max= 0.155081))
        w_down_avg_norm.append(min_max_norm(variable=ex_2020['w_down_avg'][n], min= -45.243816, max = 0.220207))

        cape_ml_avg_norm.append(min_max_norm(variable=ex_2020['cape_ml_avg'][n],  min = 0., max = 5360.163574))
        cape_sfc_avg_norm.append(min_max_norm(variable=ex_2020['cape_sfc_avg'][n], min = 0., max = 7333.146484))

        cin_ml_avg_norm.append(min_max_norm(variable=ex_2020['cin_ml_avg'][n], min = -975.307373, max = 0.))
        cin_sfc_avg_norm.append(min_max_norm(variable=ex_2020['cin_sfc_avg'][n], min= -956.824463, max =0.))

        mrms_norm.append(min_max_norm(variable=ex_2020['mrms'][n], min=0., max=98.694305))


    #save the data to a tf ds
    #make in xr dataset
    vars = [comp_dz_max_norm, comp_dz_90_norm, comp_dz_avg_norm,
            w_up_max_norm, w_up_90_norm, w_up_avg_norm,
            w_down_max_norm, w_down_90_norm, w_down_avg_norm,
            cape_ml_avg_norm, cape_sfc_avg_norm,
            cin_ml_avg_norm, cin_sfc_avg_norm, mrms_norm]
        
    names =['comp_dz_max_norm', 'comp_dz_90_norm', 'comp_dz_avg_norm',
            'w_up_max_norm', 'w_up_90_norm', 'w_up_avg_norm',
            'w_down_max_norm', 'w_down_90_norm', 'w_down_avg_norm',
            'cape_ml_avg_norm', 'cape_sfc_avg_norm',
            'cin_ml_avg_norm', 'cin_sfc_avg_norm', 'mrms_norm']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_2020_norm.nc')
    print('saved to netcdf')

if run_num == 4:
    #load in 2021 data
    ex_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_2021_fix.nc')
    #MIN-MAX Scaling on each variable for 2017-2018
    for n in range(np.size(ex_2021['comp_dz_max'], axis=0)):
        comp_dz_max_norm.append(min_max_norm(variable=ex_2021['comp_dz_max'][n], min = 0., max= 77.749992))
        comp_dz_90_norm.append(min_max_norm(variable=ex_2021['comp_dz_90'][n], min = 0., max = 75.403117))
        comp_dz_avg_norm.append(min_max_norm(variable=ex_2021['comp_dz_avg'][n], min = 0., max = 72.208328))

        w_up_max_norm.append(min_max_norm(variable=ex_2021['w_up_max'][n], min = 0.,max = 96.656242))
        w_up_90_norm.append(min_max_norm(variable=ex_2021['w_up_90'][n], min = 0., max= 80.421867))
        w_up_avg_norm.append(min_max_norm(variable=ex_2021['w_up_avg'][n], min = 0., max = 66.955719))

        w_down_max_norm.append(min_max_norm(variable=ex_2021['w_down_max'][n], min = -96.734375, max = 0.090332))
        w_down_90_norm.append(min_max_norm(variable= ex_2021['w_down_90'][n], min = -64.167187, max= 0.155081))
        w_down_avg_norm.append(min_max_norm(variable=ex_2021['w_down_avg'][n], min= -45.243816, max = 0.220207))

        cape_ml_avg_norm.append(min_max_norm(variable=ex_2021['cape_ml_avg'][n],  min = 0., max = 5360.163574))
        cape_sfc_avg_norm.append(min_max_norm(variable=ex_2021['cape_sfc_avg'][n], min = 0., max = 7333.146484))

        cin_ml_avg_norm.append(min_max_norm(variable=ex_2021['cin_ml_avg'][n], min = -975.307373, max = 0.))
        cin_sfc_avg_norm.append(min_max_norm(variable=ex_2021['cin_sfc_avg'][n], min= -956.824463, max =0.))

        mrms_norm.append(min_max_norm(variable=ex_2021['mrms'][n], min=0., max=98.694305))


    #save the data to a tf ds
    #make in xr dataset
    vars = [comp_dz_max_norm, comp_dz_90_norm, comp_dz_avg_norm,
            w_up_max_norm, w_up_90_norm, w_up_avg_norm,
            w_down_max_norm, w_down_90_norm, w_down_avg_norm,
            cape_ml_avg_norm, cape_sfc_avg_norm,
            cin_ml_avg_norm, cin_sfc_avg_norm, mrms_norm]
        
    names =['comp_dz_max_norm', 'comp_dz_90_norm', 'comp_dz_avg_norm',
            'w_up_max_norm', 'w_up_90_norm', 'w_up_avg_norm',
            'w_down_max_norm', 'w_down_90_norm', 'w_down_avg_norm',
            'cape_ml_avg_norm', 'cape_sfc_avg_norm',
            'cin_ml_avg_norm', 'cin_sfc_avg_norm', 'mrms_norm']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_2021_norm.nc')
    print('saved to netcdf')

if run_num ==5:
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2017-2018.nc')
    lb_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2017-2018.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2017-2018.nc')

    problem_cases = unreasonable_values(ex_2017)

    new_ex_2017  = ex_2017.drop_isel({'n_samples' : problem_cases})
    new_lb_2017 = lb_2017.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})
    print(np.shape(new_ex_2017['comp_dz_max']))

    new_ex_2017.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2017-2018_fix.nc')
    new_lb_2017.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2017-2018_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2017-2018_fix.nc')

    new_ex=new_ex_2017.to_array()
    new_lb=new_lb_2017.to_array()

    new_ex = new_ex.transpose('n_samples',...)
    new_ex = new_ex.transpose(...,'variable')

    new_lb = new_lb.transpose('n_samples',...)
    new_lb = new_lb.transpose(...,'variable')

    tf_ds = tf.data.Dataset.from_tensor_slices((new_ex,new_lb))
    tf.data.experimental.save(tf_ds, '/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/training_2017-2018.tf')

if run_num ==6:
    #NEED TO TRANSPOSE
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2019.nc')
    lb_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2019.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2019.nc')

    problem_cases =unreasonable_values(ex_2019)

    new_ex_2019  = ex_2019.drop_isel({'n_samples' : problem_cases})
    new_lb_2019 = lb_2019.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})
    print(np.shape(new_ex_2019['comp_dz_max']))

    new_ex_2019.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/examples/examples_2019_fix.nc')
    new_lb_2019.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/training/labels/labels_2019_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2019_fix.nc')

    #need to put it into a numpy arr, then transpose before putting into tf ds
    #make into arrays
    new_ex=new_ex_2019.to_array()
    new_lb=new_lb_2019.to_array()

    new_ex = new_ex.transpose('n_samples',...)
    new_ex = new_ex.transpose(...,'variable')

    new_lb = new_lb.transpose('n_samples',...)
    new_lb = new_lb.transpose(...,'variable')

    #put into tf ds and save
    tf_ds = tf.data.Dataset.from_tensor_slices((new_ex,new_lb))
    tf.data.experimental.save(tf_ds, '/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/training_2019.tf')

if run_num ==7:
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_2020.nc')
    lb_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/labels/labels_2020.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2020.nc')

    problem_cases =unreasonable_values(ex_2020)

    new_ex_2020  = ex_2020.drop_isel({'n_samples' : problem_cases})
    new_lb_2020 = lb_2020.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})

    print(np.shape(new_ex_2020['comp_dz_max']))
    new_ex_2020.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/examples/examples_2020_fix.nc')
    new_lb_2020.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/validation/labels/labels_2020_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2020_fix.nc')

    new_ex=new_ex_2020.to_array()
    new_lb=new_lb_2020.to_array()

    new_ex = new_ex.transpose('n_samples',...)
    new_ex = new_ex.transpose(...,'variable')

    new_lb = new_lb.transpose('n_samples',...)
    new_lb = new_lb.transpose(...,'variable')

    tf_ds = tf.data.Dataset.from_tensor_slices((new_ex,new_lb))
    tf.data.experimental.save(tf_ds, '/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/validation.tf')

if run_num == 8:
    ex_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_2021.nc')
    lb_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/labels/labels_2021.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2021.nc')

    problem_cases = unreasonable_values(ex_2021)

    new_ex_2021  = ex_2021.drop_isel({'n_samples' : problem_cases})
    new_lb_2021 = lb_2021.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})

    print(np.shape(new_ex_2021['comp_dz_max']))
    new_ex_2021.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/examples/examples_2021_fix.nc')
    new_lb_2021.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/test/labels/labels_2021_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/lat_lon_2021_fix.nc')

    new_ex=new_ex_2021.to_array()
    new_lb=new_lb_2021.to_array()

    new_ex = new_ex.transpose('n_samples',...)
    new_ex = new_ex.transpose(...,'variable')

    new_lb = new_lb.transpose('n_samples',...)
    new_lb = new_lb.transpose(...,'variable')

    tf_ds = tf.data.Dataset.from_tensor_slices((new_ex,new_lb))
    tf.data.experimental.save(tf_ds, '/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/test.tf')

print('done')
    






    

