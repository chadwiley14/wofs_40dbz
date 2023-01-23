
import numpy as np
import xarray as xr
from data_manipulation import *
from patcher_helper import *


def load_file(path):
    ds = xr.load_dataset(path)
    return ds

parser = create_parser1()
args = parser.parse_args()

run_num = args.run_num
run_num = run_num[0]


if run_num == 8: #make example dataset
    ens_2020 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/ENS_data_2020_06.nc')
    ens_2019 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/ENS_data_2019_06.nc')
    svr_2020 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/SVR_data_2020_06.nc')
    svr_2019 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/SVR_data_2019_06.nc')
    init_2020 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/init_mrms_data_2020_06.nc')
    init_2019 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/init_mrms_data_2019_06.nc')
    print('data loaded')

    examples_2020 = xr.merge([ens_2020,svr_2020,init_2020])
    examples_2019 = xr.merge([ens_2019, svr_2019,init_2019])

    examples = xr.concat([examples_2020,examples_2019], dim = 'n_samples')

    ae = split_data_examples(examples)
    print(np.shape(ae))

    comp_dz = ae[:,0,:,:]
    w_up =  ae[:,1,:,:]
    w_down =  ae[:,2,:,:]
    cape_ml = ae[:,3,:,:]
    cape_sfc = ae[:,4,:,:]
    cin_ml = ae[:,5,:,:]
    cin_sfc=  ae[:,6,:,:]
    dz_cress  =ae[:,7,:,:]

  
    vars = [comp_dz, w_up, w_down, cape_ml,cape_sfc, cin_ml, cin_sfc, dz_cress]
        
    names =['comp_dz', 'w_up', 'w_down', 'cape_ml', 'cape_sfc', 'cin_ml', 'cin_sfc', 'dz_cress']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/examples_30.nc')

    print('data saved')

if run_num == 9:
    val_2020 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/val_mrms_data_2020_06nonorm.nc')
    val_2019 = load_file('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/val_mrms_data_2019_06nonorm.nc')
    print('data loaded')

    labels = xr.concat([val_2020,val_2019], dim = 'n_samples')

    arr_labels = split_data_labels(labels)

    print(np.shape(arr_labels))
    dz = arr_labels[:,0,:,:]
    #dz_cress = scipy.ndimage.gaussian_filter(dz, sigma = 3)
    #dz = scipy.ndimage.maximum_filter(input = dz, size = 3)


    vars = [dz]
    names =['dz_cress']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/labels_30_nonorm.nc')
    print('saved')

if run_num == 10:
    ens_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/ENS_data_2020_06nonorm.nc')
    ens_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/ENS_data_2019_06nonorm.nc')
    print('data is loaded')

    ds = xr.concat([ens_2020,ens_2019], dim = 'n_samples')

    ds = ds.to_array()
    ds = ds.to_numpy()

    ds1 = np.split(ds, 2, axis = 2)
    ds2 = np.split(ds1[0], 2, axis = 3)
    ds3 = np.split(ds1[1], 2, axis = 3)

    gh = ds2+ds3
    print(np.shape(gh))

    ds4 = np.concatenate([ds2,ds3])
    ds4t = np.moveaxis(ds4, [1],[2])
    ds5 = np.reshape(ds4t, ((np.size(ds4t, axis = 0)* np.size(ds4t, axis = 1)),3,150,150))

    #ds5 = np.reshape(ds4, ((np.size(ds4, axis = 0)* np.size(ds, axis = 1)),8,150,150))

    ae = ds5[:,:,11:139,11:139]
    print(np.shape(ae))

    comp_dz = ae[:,0,:,:]
    w_up =  ae[:,1,:,:]
    w_down =  ae[:,2,:,:]

    vars = [comp_dz, w_up, w_down]
        
    names =['comp_dz', 'w_up', 'w_down']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30/examples_30_nonorm.nc')

    print('data saved')



    


