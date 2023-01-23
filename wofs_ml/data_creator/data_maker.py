from patcher_helper import *
from data_manipulation import *


parser = create_parser1()
args = parser.parse_args()

year = args.year
lead_time = args.lead_time
run_num = args.run_num2
run_num = run_num[0]

if run_num == 0:
    ens = glob.glob('/scratch/chadwiley/inter_data/ENS_*')
    svr = glob.glob('/scratch/chadwiley/inter_data/SVR_*')
    init = glob.glob('/scratch/chadwiley/inter_data/init_mrms_*')
    val = glob.glob('/scratch/chadwiley/inter_data/val_mrms_*')

    ens.sort()
    svr.sort()
    init.sort()
    val.sort()

    print(ens[9])
    print(svr[9])
    print(init[9])
    print(val[9])


    ens_data = xr.open_mfdataset(ens, concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    svr_data = xr.open_mfdataset(svr,concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    init_data = xr.open_mfdataset(init,concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    val_data = xr.open_mfdataset(val,concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')

    val_data['val_dz'] = val_data['dz_cress']
    val_data = val_data.drop(['dz_cress'])

    examples = xr.merge([ens_data, svr_data, init_data,val_data], compat="identical")
    examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/1hour_lead/nonsplit_examples_labels.nc')
    print('done')
    # print(examples)

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
    mrms  =ae[:,8,:,:]

  
    vars = [comp_dz, w_up, w_down, cape_ml,cape_sfc, cin_ml, cin_sfc, dz_cress, mrms]
        
    names =['comp_dz', 'w_up', 'w_down', 'cape_ml', 'cape_sfc', 'cin_ml', 'cin_sfc', 'dz_cress', 'mrms']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/1hour_lead/examples/2019_2020_examples_labels.nc')
    print('saved')

if run_num ==1:
    #creates labels dataset
    val = glob.glob('/scratch/chadwiley/inter_data/val_mrms_*')
    ens_data = xr.open_mfdataset(ens, concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    val.sort()
    ens_data.sort()
    print(val[9])
    print(ens[9])

    labels = xr.open_mfdataset(val,concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    ens_data = xr.open_mfdataset(ens_data['comp_dz'], concat_dim= 'n_samples',  combine='nested', engine= 'netcdf4')
    labels = xr.merge([labels,ens_data])

    labels.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/1hour_lead/nonsplit_labels.nc')
    print('done')

    arr_labels = split_data_labels(labels)

    print(np.shape(arr_labels))


    vars = [arr_labels[:,0,:,:]]
    names =['dz_cress']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/1hour_lead/labels/2019_2020_labels.nc')
    print('saved')

