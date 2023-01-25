import argparse
import numpy as np
import xarray as xr
import netCDF4
import scipy

def create_parser():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--run_num', type=int, default = None)
    return parser

def break_into_patches(dataset):
    ds = []
    for i in dataset:
        print(i)
        cur_data = xr.load_dataset(i)
        # print(cur_data)
        cur_data = cur_data.to_array()
        cur_data = cur_data.to_numpy()

        #break into 5x5, 64x64
        for n in range(np.size(cur_data,axis=1)):
            patch_1 = cur_data[:,n,0:64,0:64]
            patch_2 = cur_data[:,n,64:128,0:64]
            patch_3 = cur_data[:,n,128:192,0:64]
            patch_4 = cur_data[:,n,192:256,0:64]
            # print(patch_1.shape)
            # print(patch_2.shape)
            # print(patch_3.shape)
            # print(patch_4.shape)

            patch_6 = cur_data[:,n,0:64,64:128]
            patch_7 = cur_data[:,n,64:128,64:128]
            patch_8 = cur_data[:,n,128:192,64:128]
            patch_9 = cur_data[:,n,192:256,64:128]
            # print(patch_6.shape)
            # print(patch_7.shape)
            # print(patch_8.shape)
            # print(patch_9.shape)

            patch_11 = cur_data[:,n,0:64,128:192]
            patch_12 = cur_data[:,n,64:128,128:192]
            patch_13 = cur_data[:,n,128:192,128:192]
            patch_14 = cur_data[:,n,192:256,128:192]
            # print(patch_11.shape)
            # print(patch_12.shape)
            # print(patch_13.shape)
            # print(patch_14.shape)

            patch_16 = cur_data[:,n,0:64,192:256]
            patch_17 = cur_data[:,n,64:128,192:256]
            patch_18 = cur_data[:,n,128:192,192:256]
            patch_19 = cur_data[:,n,192:256,192:256]
            # print(patch_16.shape)
            # print(patch_17.shape)
            # print(patch_18.shape)
            # print(patch_19.shape)

            if np.shape(cur_data[0,0,:,:]) == (320,320):
                #for 2021-2019 cases. Domains are 300x300 with padding of 10.
                patch_5 = cur_data[:,n,256:320,0:64]
                patch_10 = cur_data[:,n,256:320,64:128]
                patch_15 = cur_data[:,n,256:320,128:192]
                patch_20 = cur_data[:,n,256:320,192:256]
                # print(patch_5.shape)
                # print(patch_10.shape)
                # print(patch_15.shape)
                # print(patch_20.shape)

                patch_21 = cur_data[:,n,0:64,256:320]
                patch_22 = cur_data[:,n,64:128,256:320]
                patch_23 = cur_data[:,n,128:192,256:320]
                patch_24 = cur_data[:,n,192:256,256:320]
                patch_25 = cur_data[:,n,256:320,256:320]
                # print(patch_21.shape)
                # print(patch_22.shape)
                # print(patch_23.shape)
                # print(patch_24.shape)
                # print(patch_25.shape)

                ds.append(patch_1)
                ds.append(patch_2)
                ds.append(patch_3)
                ds.append(patch_4)
                ds.append(patch_5)
                ds.append(patch_6)
                ds.append(patch_7)
                ds.append(patch_8)
                ds.append(patch_9)
                ds.append(patch_10)
                ds.append(patch_11)
                ds.append(patch_12)
                ds.append(patch_13)
                ds.append(patch_14)
                ds.append(patch_15)
                ds.append(patch_16)
                ds.append(patch_17)
                ds.append(patch_18)
                ds.append(patch_19)
                ds.append(patch_20)
                ds.append(patch_21)
                ds.append(patch_22)
                ds.append(patch_23)
                ds.append(patch_24)
                ds.append(patch_25)     
            else:
                ds.append(patch_1)
                ds.append(patch_2)
                ds.append(patch_3)
                ds.append(patch_4)
                ds.append(patch_6)
                ds.append(patch_7)
                ds.append(patch_8)
                ds.append(patch_9)
                ds.append(patch_11)
                ds.append(patch_12)
                ds.append(patch_13)
                ds.append(patch_14)
                ds.append(patch_16)
                ds.append(patch_17)
                ds.append(patch_18)
                ds.append(patch_19)
    ds = np.moveaxis(ds,[0],[1])
    print(np.shape(ds))       
    return ds


def save_ens(ens_patches):
    comp_dz_max = ens_patches[0,:,:,:]
    comp_dz_90 = ens_patches[1,:,:,:]
    comp_dz_avg = ens_patches[2,:,:,:]

    w_up_max = ens_patches[3,:,:,:]
    w_up_90 = ens_patches[4,:,:,:]
    w_up_avg = ens_patches[5,:,:,:]

    w_down_max = ens_patches[6,:,:,:]
    w_down_90 = ens_patches[7,:,:,:]
    w_down_avg = ens_patches[8,:,:,:]

    vars = [comp_dz_max, comp_dz_90, comp_dz_avg,
        w_up_max, w_up_90, w_up_avg,
        w_down_max, w_down_90, w_down_avg]
        
    names =['comp_dz_max' , 'comp_dz_90', 'comp_dz_avg',
        'w_up_max', 'w_up_90', 'w_up_avg', 
        'w_down_max', 'w_down_90', 'w_down_avg']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}


    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/ens_patches.nc')
    print('saved ENS patches')

def save_svr(svr_patches):
    cape_ml = svr_patches[0,:,:,:]
    cape_sfc = svr_patches[1,:,:,:]
    cin_ml = svr_patches[2,:,:,:]
    cin_sfc = svr_patches[3,:,:,:]

    vars = [cape_ml,cape_sfc,cin_ml,cin_sfc]
    names =['cape_ml', 'cape_sfc', 'cin_ml', 'cin_sfc']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/svr_patches.nc')
    print('saved SVR patches')

def save_init(init_patches):
    dz_cress = init_patches[0,:,:,:]
    vars = [dz_cress]
    names =['dz_cress']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/init_patches.nc')

def save_val(val_patches):
    dz_cress = val_patches[0,:,:,:]
    dz_cress_class_np = val_patches[1,:,:,:]
    vars = [dz_cress, dz_cress_class_np]
    names =['dz_cress', 'dz_cress_binary']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/val_patches.nc')


def save_lat_lon(lat_patches):
    #FIX
    lat = lat_patches[0,:,:]
    lon = lat_patches[1,:,:]
    
    vars2 =[lat,lon]
    names2 = ['lat', 'lon']
    size2 = ['n_samples','NY', 'NX']
    tup = [(size2,var)for var in vars2]
    data_vars2  = {names2 : data_tups for names2, data_tups in zip(names2, tup)}
    out_ds = xr.Dataset(data_vars2)
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/patches/lat_lon_patches.nc')
    print('saved init patches')

def make_wofs_patches(ds, year):
    wofs_patches = break_into_patches(ds)
    print('wofs patches shape : %s'%str(np.shape(wofs_patches)))
    
    #save data to an xr file
    vars = [wofs_patches[0]]
    names =['wofs_probs']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/wofs_probs/wofs_probs_patches_%s.nc'%year)
    print(out_ds)
    print('saved wofs patches %s'%year)
    

    
