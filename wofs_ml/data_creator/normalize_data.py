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

def make_tf_ds(ds_ex, ds_lb):
    """
    Given examples and labels, returns
    a tensorflow dataset. 
    Data should already be normalized.
    
    PARAMETERS
    -----------------------
    ds_ex : xr dataset
        xarray dataset of examples
    
    ds_lb : xr dataset
        dataset of labels

    RETURNS
    -------------------------
    tf_ds : tensorflow dataset
        Returns a tensorflow dataset
        meant for tf models.
    """
    ds_ex=ds_ex.to_array()
    ds_lb=ds_lb.to_array()

    ds_ex = ds_ex.transpose('n_samples',...)
    ds_ex = ds_ex.transpose(...,'variable')

    ds_lb = ds_lb.transpose('n_samples',...)
    ds_lb = ds_lb.transpose(...,'variable')

    tf_ds = tf.data.Dataset.from_tensor_slices((ds_ex,ds_lb))
    return tf_ds

def normalize_data(ds):
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

    for n in range(np.size(ds['comp_dz_max'], axis=0)):
        comp_dz_max_norm.append(min_max_norm(variable=ds['comp_dz_max'][n], min = 0., max= 77.749992))
        comp_dz_90_norm.append(min_max_norm(variable=ds['comp_dz_90'][n], min = 0., max = 75.403117))
        comp_dz_avg_norm.append(min_max_norm(variable=ds['comp_dz_avg'][n], min = 0., max = 72.208328))

        w_up_max_norm.append(min_max_norm(variable=ds['w_up_max'][n], min = 0.,max = 96.656242))
        w_up_90_norm.append(min_max_norm(variable=ds['w_up_90'][n], min = 0., max= 80.421867))
        w_up_avg_norm.append(min_max_norm(variable=ds['w_up_avg'][n], min = 0., max = 66.955719))

        w_down_max_norm.append(min_max_norm(variable=ds['w_down_max'][n], min = -96.734375, max = 0.099314))
        w_down_90_norm.append(min_max_norm(variable= ds['w_down_90'][n], min = -70.164055, max= 0.155081))
        w_down_avg_norm.append(min_max_norm(variable=ds['w_down_avg'][n], min= -45.243816, max = 0.220207))

        cape_ml_avg_norm.append(min_max_norm(variable=ds['cape_ml_avg'][n],  min = 0., max = 5932.417969))
        cape_sfc_avg_norm.append(min_max_norm(variable=ds['cape_sfc_avg'][n], min = 0., max = 7639.454102))

        cin_ml_avg_norm.append(min_max_norm(variable=ds['cin_ml_avg'][n], min = -1029.088013, max = 0.))
        cin_sfc_avg_norm.append(min_max_norm(variable=ds['cin_sfc_avg'][n], min= -1138.226685, max =0.))

        mrms_norm.append(min_max_norm(variable=ds['mrms'][n], min=0., max=99.458038))


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

    return out_ds


#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num


if run_num ==0:
    #get all the example files
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018_fix.nc')
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019_fix.nc')
    ex_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_fix.nc')
    ex_2021 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_fix.nc')

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

#################################################################
#################################################################
#################################################################

if run_num == 1:
    #do 2017-2018 examples
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018_fix.nc')
    out_ds = normalize_data(ds = ex_2017)
    print(out_ds)

    #Save to netcdf
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018_norm.nc')
    print('saved to netcdf')

    #Save to Tensorflow dataset
    #load in 2017 labels
    lb_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2017-2018_fix.nc')

    tf_ds = make_tf_ds(out_ds,lb_2017)

    tf.data.experimental.save(tf_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/training_2017-2018.tf')
    print('saved to tf ds')

#################################################################
#################################################################
#################################################################
if run_num == 2:
    #load in 2019 data
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019_fix.nc')

    out_ds = normalize_data(ds = ex_2019)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019_norm.nc')
    print('saved to netcdf')

    #load in 2019 labels
    lb_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2019_fix.nc')

    tf_ds = make_tf_ds(out_ds,lb_2019)

    tf.data.experimental.save(tf_ds,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/training_2019.tf')
    print('saved to tf ds')

#################################################################
#################################################################
#################################################################

if run_num == 3:
    #load in 2020 data
    #training
    ex_2020_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_train_fix.nc')
    ex_2020_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_test_fix.nc')

    print(ex_2020_train)
    print('-------------------------')
    print(ex_2020_test)
    print('-------------------------')
    print('-------------------------')
    print('-------------------------')


    out_ds_train = normalize_data(ds = ex_2020_train)
    out_ds_test = normalize_data(ds = ex_2020_test)

    print(out_ds_train)
    print('-------------------------')
    print(out_ds_test)
    print('-------------------------')
    print('-------------------------')
    print('-------------------------')

    out_ds_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_train_norm.nc')
    out_ds_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_val_norm.nc')
    print('saved to netcdf')

    #load in 2020 labels
    lb_2020_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_train_fix.nc')
    lb_2020_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_test_fix.nc')

    lb_2020_test = lb_2020_test.drop_vars(['dz'])
    lb_2020_train = lb_2020_train.drop_vars(['dz'])


    tf_ds_train = make_tf_ds(out_ds_train, lb_2020_train)
    tf_ds_test = make_tf_ds(out_ds_test, lb_2020_test)

    tf.data.experimental.save(tf_ds_train,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/validation_2020_train.tf')
    tf.data.experimental.save(tf_ds_test,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/validation_2020_val.tf')
    print('saved to tf ds')

#################################################################
#################################################################
#################################################################

if run_num == 4:
    #load in 2021 data
    ex_2021_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_train_fix.nc')
    ex_2021_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_test_fix.nc')

    #MIN-MAX Scaling on each variable for 2021
    out_ds_train = normalize_data(ds = ex_2021_train)
    out_ds_test = normalize_data(ds = ex_2021_test)
    
    print(out_ds_train)
    print('-------------------------')
    print(out_ds_test)

    out_ds_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_train_norm.nc')
    out_ds_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_test_norm.nc')
    print('saved to netcdf')

    #load in 2021 labels
    lb_2021_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_train_fix.nc')
    lb_2021_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_test_fix.nc')


    tf_ds_train = make_tf_ds(out_ds_train, lb_2021_train)
    tf_ds_test = make_tf_ds(out_ds_test, lb_2021_test)


    tf.data.experimental.save(tf_ds_train,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/test_2021_train.tf')
    tf.data.experimental.save(tf_ds_test,'/ourdisk/hpc/ai2es/chadwiley/patches/data_64/tf_ds/test_2021_test.tf')
    print('saved to tf ds')

#################################################################
#################################################################
#################################################################

if run_num ==5:
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018.nc')
    lb_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2017-2018.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2017-2018.nc')

    problem_cases = unreasonable_values(ex_2017)

    new_ex_2017  = ex_2017.drop_isel({'n_samples' : problem_cases})
    new_lb_2017 = lb_2017.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})

    print(np.shape(new_ex_2017['comp_dz_max']))

    new_ex_2017.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018_fix.nc')
    new_lb_2017.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2017-2018_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2017-2018_fix.nc')

if run_num ==6:
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019.nc')
    lb_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2019.nc')
    lat_lon = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2019.nc')

    problem_cases =unreasonable_values(ex_2019)

    new_ex_2019  = ex_2019.drop_isel({'n_samples' : problem_cases})
    new_lb_2019 = lb_2019.drop_isel({'n_samples' : problem_cases})
    new_lat_lon = lat_lon.drop_isel({'n_samples' : problem_cases})
    print(np.shape(new_ex_2019['comp_dz_max']))

    new_ex_2019.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019_fix.nc')
    new_lb_2019.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/labels/labels_2019_fix.nc')
    new_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2019_fix.nc')


if run_num ==7:
    ex_2020_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_train.nc')
    lb_2020_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_train.nc')
    lat_lon_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2020_train.nc')

    ex_2020_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_test.nc')
    lb_2020_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_test.nc')
    lat_lon_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2020_test.nc')

    #drop cases from the same date
    #in the 2020 case, from the first 80 from the testing
    drop_2020 = np.arange(0,81)
    ex_2020_test = ex_2020_test.drop_isel({'n_samples' : drop_2020})
    lb_2020_test = lb_2020_test.drop_isel({'n_samples' : drop_2020})
    lat_lon_test = lat_lon_test.drop_isel({'n_samples' : drop_2020})


    problem_cases_train =unreasonable_values(ex_2020_train)
    problem_cases_test =unreasonable_values(ex_2020_test)


    new_ex_2020_train  = ex_2020_train.drop_isel({'n_samples' : problem_cases_train})
    new_lb_2020_train = lb_2020_train.drop_isel({'n_samples' : problem_cases_train})
    new_lat_lon_train = lat_lon_train.drop_isel({'n_samples' : problem_cases_train})

    new_ex_2020_test  = ex_2020_test.drop_isel({'n_samples' : problem_cases_test})
    new_lb_2020_test = lb_2020_test.drop_isel({'n_samples' : problem_cases_test})
    new_lat_lon_test = lat_lon_test.drop_isel({'n_samples' : problem_cases_test})

    print(np.shape(new_ex_2020_train['comp_dz_max']))
    new_ex_2020_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_train_fix.nc')
    new_lb_2020_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_train_fix.nc')
    new_lat_lon_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2020_train_fix.nc')

    new_ex_2020_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_test_fix.nc')
    new_lb_2020_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/labels/labels_2020_test_fix.nc')
    new_lat_lon_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2020_test_fix.nc')

if run_num == 8:
    ex_2021_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_train.nc')
    lb_2021_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_train.nc')
    lat_lon_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2021_train.nc')

    ex_2021_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_test.nc')
    lb_2021_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_test.nc')
    lat_lon_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2021_test.nc')

    #drop cases from the same data, in this case the last 49 from training
    drop_2021 = np.arange(-49,-0)
    ex_2021_train = ex_2021_train.drop_isel({'n_samples' : drop_2021})
    lb_2021_train = lb_2021_train.drop_isel({'n_samples' : drop_2021})
    lat_lon_train = lat_lon_train.drop_isel({'n_samples' : drop_2021})

    problem_cases_train = unreasonable_values(ex_2021_train)
    problem_cases_test = unreasonable_values(ex_2021_test)


    print(problem_cases_train)
    print(problem_cases_test)

    new_ex_2021_train  = ex_2021_train.drop_isel({'n_samples' : problem_cases_train})
    new_lb_2021_train = lb_2021_train.drop_isel({'n_samples' : problem_cases_train})
    new_lat_lon_train = lat_lon_train.drop_isel({'n_samples' : problem_cases_train})

    new_ex_2021_test = ex_2021_test.drop_isel({'n_samples' : problem_cases_test})
    new_lb_2021_test = lb_2021_test.drop_isel({'n_samples' : problem_cases_test})
    new_lat_lon_test = lat_lon_test.drop_isel({'n_samples' : problem_cases_test})

    print(np.shape(new_ex_2021_train['comp_dz_max']))
    new_ex_2021_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_train_fix.nc')
    new_lb_2021_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_train_fix.nc')
    new_lat_lon_train.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2021_train_fix.nc')

    new_ex_2021_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_test_fix.nc')
    new_lb_2021_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/labels/labels_2021_test_fix.nc')
    new_lat_lon_test.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/lat_lon_2021_test_fix.nc')
    
if run_num == 9:
    ex_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2017-2018_norm.nc')
    ex_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/training/examples/examples_2019_norm.nc')
    ex_2020_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_train_norm.nc')
    ex_2021_train = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_train_norm.nc')

    ex_2020_val = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/validation/examples/examples_2020_val_norm.nc')

    ex_2021_test = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_64/testing/examples/examples_2021_test_norm.nc')

    train = xr.concat([ex_2017,ex_2019,ex_2020_train,ex_2021_train], dim = 'n_samples')
    

    print(train)
    print('#######################################')
    print('#######################################')
    print('#######################################')
    print(ex_2020_val)

    print('#######################################')
    print('#######################################')
    print('#######################################')
    print(ex_2021_test)

print('done')








    

