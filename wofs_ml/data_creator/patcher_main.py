from pkg_resources import ensure_directory
from data_manipulation import *
from patcher_helper import *

#create parser
parser = create_parser1()
args = parser.parse_args()

year = args.year
lead_time = args.lead_time
run_num = args.run_num
run_num = run_num[0]

#get wofs for 2019 and 2020
ENS_files, SVR_files = get_all_wofs_files(year, lead_time)
print('Got ENS and SVR files from %s'%year)
print('There are %d ENS files and %d SVR files for %s with a lead time of %s\n'%(len(ENS_files), len(SVR_files), year, lead_time))

#get MRMS files based on dates and times from ENS data
print(ENS_files[0])
init_files, val_files = mrms_files(ENS_files, lead_time, year)

print('\nGot MRMS files from %s'%year)
print('There are %d init files and %d val files for %s'%(len(init_files), len(val_files), year))

#from MRMS files, extract the ENS and SVR files
complete_ens, complete_svr = match_wofs_mrms(init_files, val_files, lead_time, year)
print('There are %d ens matched and %d svr matched from %s\n'%(len(complete_ens), len(complete_svr), year))
print('examples:')
print('ENS : %s and size: %s'%(complete_ens[60], np.size(complete_ens)))
print('SVR : %s and size : %s'%(complete_svr[60], np.size(complete_svr)))
print('init mrms : %s and size: %s'%(init_files[60],np.size(init_files)))
print('val mrms : %s and size: %s'%(val_files[60], np.size(val_files)))

#load in the data


if year == "2020":
    
    if run_num == 1:
        get_ens_data(complete_ens, lead_time, year)
    elif run_num == 2:
        get_svr_data(complete_svr, lead_time, year)
    elif run_num == 3:
        init_data = xr.open_mfdataset(init_files, decode_times = False, concat_dim = 'ne', combine='nested', engine= 'netcdf4', drop_variables=['lat', 'lon'])
        get_init_data(init_data, lead_time, year)
    elif run_num == 4:
        val_data = xr.open_mfdataset(val_files, decode_times = False,concat_dim = 'ne', combine='nested', engine= 'netcdf4', drop_variables=['lat', 'lon'])
        get_val_data(val_data, lead_time, year)
 
else:
    if run_num == 1:
        get_ens_data(complete_ens, lead_time, year)
    elif run_num == 2:
        get_svr_data(complete_svr, lead_time, year)
    elif run_num == 3:
        init_data = xr.open_mfdataset(init_files, concat_dim = 'lat',combine='nested', engine= 'netcdf4', decode_times = False, drop_variables = ['XLAT', 'XLON'])
        get_init_data(init_data, lead_time, year)
    elif run_num == 4:
        val_data = xr.open_mfdataset(val_files, concat_dim = 'lat', combine='nested', engine= 'netcdf4', decode_times = False, drop_variables = ['XLAT', 'XLON'])
        get_val_data(val_data, lead_time, year)

print("Data is loaded and extracted")