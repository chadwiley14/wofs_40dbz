import re
import xarray as xr
import numpy as np
import glob
import os
import scipy 
from datetime import datetime, timedelta
from data_manipulation import *

#all these methods are used to find matching files with a given lead time and match them
def add_padding(cur_var, year):
    """
    adds in padding based on the year
    Parameters
    ------------
    cur_var : array
        This is the current vaule that is adding padding
    year : int
        Year of the data.
    Returns
    ------------
    based on the year either returns the variable with
    a padding of 10 or 3 on all edges. 
    """
    np.shape(cur_var)
    if year == '2017' or year =='2018':
        return np.pad(cur_var, pad_width=3,mode='constant')
    else:
        return np.pad(cur_var, pad_width=10,mode='constant')

def create_parser1():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--lead_time', type=str, default = '00', help= 'Amount of time between init and valid')
    parser.add_argument('--year', type=str, default = '2020', help = "Year of interest. As of now only 2019 and 2020")
    parser.add_argument('--save_path',type=str, default = None )
    parser.add_argument('--run_num', type=int, nargs='+', default = None)
    parser.add_argument('--run_num2', type=int, nargs='+', default = None)
    return parser


#load example data
def get_all_wofs_files(year, lead_time):
    """
    This gets all the paths of ENS and SVR files

    Parameters
    -----------------
    year : str
        The year but in a str format. Currenly only 2019/2020.
    lead_time : str
        How long of a lead time between init and val.
        Must give with a leading 0 if less than 10.
        ex 09 is 45 minute forecast.

    Returns
    ---------------------
    ENS_files : list
        This is a list of all ENS files with that
        lead time from that year
    
    SVR_files : list
        List of all SVR files with the given lead
        time from given year
    """
    SVR_files = []
    ENS_files = []
    temp_ENS = []

    for i in range(np.size(lead_time)):
        for j in range(np.size(year)):
            temp_ENS += glob.glob("/ourdisk/hpc/ai2es/wofs/{year}_summary/*/*/wofs_ENS_{lead_time}_*".format(year=year[j], lead_time = lead_time[i]))


    for i in temp_ENS:   
        temp_SVR = i.replace("ENS", "SVR")
        if os.path.exists(temp_SVR) and os.path.getsize(i) > 0 and os.path.getsize(temp_SVR) > 0:
            ENS_files.append(i)
            SVR_files.append(temp_SVR)

    ENS_files.sort()
    SVR_files.sort()

    return ENS_files, SVR_files

 #/ourdisk/hpc/ai2es/wofs/MRMS/2020/2020/20200515/wofs_MRMS_RAD_20200515_2220.nc
 #/ourdisk/hpc/ai2es/wofs/MRMS/2019/2019/20190510/20190510_192000.nc

def match_datetime(val, lead_time, year):
    """
    Takes in a MRMS file that is the vaild time
    and returns the initalization time in a str
    format.

    Parameters
    ---------------
    val : str
        The MRMS file that is the label of the dataset
    lead_time : str
        How long of a lead time between init and val.
        Must give with a leading 0 if less than 10.
        ex 09 is 45 minute forecast.
    year : str
        The year but in a str format. Currenly only 2019/2020.
    Returns
    -------------------
    Returns the init time and date matching the convections
    of 2019 and 2020.
    """
    if year == "2020":
        cur_val_time = re.findall("_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].nc",val)
        cur_val_time = datetime.strptime(cur_val_time[0], "_%Y%m%d_%H%M.nc")
    else:
        cur_val_time = re.findall("/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].nc", val)
        cur_val_time = datetime.strptime(cur_val_time[0], '/%Y%m%d_%H%M%S.nc')
    
    init_datetime = cur_val_time - timedelta(minutes=(int(lead_time)*5))
    to_str_init = datetime.strptime(str(init_datetime), "%Y-%m-%d %H:%M:%S")
    
    if year == "2020":
        return to_str_init.strftime('_%Y%m%d_%H%M.nc')
    else:
        return to_str_init.strftime('%Y%m%d_%H%M%S.nc')

def mrms_files(ens, lead_time, year):
    """
    Based on the ENS files, finds all MRMS files that present.

    Parameters
    --------------
    ens : list
        List of file paths for ENS
    lead_time : str
        How long of a lead time between init and val.
        Must give with a leading 0 if less than 10.
        ex 09 is 45 minute forecast.
    year : str
        The year but in a str format. Currenly only 2019/2020.
    Returns
    -----------------
    init_mrms : list
        List of str containing file paths of init_mrms files
    val_mrms : list
        List of str containing file paths of val_mrms files
    """
    val_mrms =[]
    init_mrms = []
    temp_val = []
    temp_init = []
    for cur_ens in ens:
        cur_date = re.findall("20[1-2][0-9][0-1][0-9][0-3][0-9]", cur_ens)
        cur_val_time = re.findall('_[0-9][0-9][0-9][0-9].nc', cur_ens)
        if year == '2020':
            #grabs the val file path
            if cur_date[0] != '20200522' and cur_val_time[0] != '_1730.nc':
                temp_val = glob.glob('/ourdisk/hpc/ai2es/wofs/MRMS/{year1}/{year2}/{cur_date}/*_{val_time}.nc'\
                    .format(year1 = cur_date[0][:4], year2 = cur_date[0][:4], cur_date = cur_date[0], val_time = cur_val_time[0][1:-3]))

            #if val file exist, grab init file
            if len(temp_val) > 0 :
                init_datetime = match_datetime(temp_val[0], lead_time, year)

                temp_init = glob.glob('/ourdisk/hpc/ai2es/wofs/MRMS/{year1}/{year2}/{cur_date}/*{init_datetime}'\
                    .format(year1 = cur_date[0][:4], year2 = cur_date[0][:4], cur_date = cur_date[0], init_datetime = init_datetime)) 

        #for 2019
        else:
            if cur_date[0] != '20190530' and cur_val_time[0] != '_0030.nc' and cur_date[0] != '20190522' and cur_val_time[0] != '_0000.nc':
                temp_val = glob.glob("/ourdisk/hpc/ai2es/wofs/MRMS/{year1}_NEW/{year2}/{cur_date}/*{val_time}00.nc"\
                    .format(year1 = cur_date[0][:4], year2 = cur_date[0][:4], cur_date = cur_date[0], val_time = cur_val_time[0][:-3]))
                    #/ourdisk/hpc/ai2es/wofs/MRMS/2019/2019/20190510/20190510_192000.nc

                if len(temp_val) > 0:
                    init_datetime = match_datetime(temp_val[0], lead_time, year)

                    temp_init = glob.glob('/ourdisk/hpc/ai2es/wofs/MRMS/{year1}_NEW/{year2}/{cur_date}/{init_datetime}'\
                        .format(year1 = cur_date[0][:4], year2 = cur_date[0][:4], cur_date = cur_date[0], init_datetime = init_datetime)) 

        if len(temp_val) > 0 and os.path.getsize(temp_val[0]) > 0 and len(temp_init) > 0 and os.path.getsize(temp_init[0]) > 0:
            val_mrms.append(temp_val[0])
            init_mrms.append(temp_init[0])
        
    return init_mrms,val_mrms

def _dates_times(init_mrms, val_mrms, year):
    """
    Helper function for match_wofs_mrms.
    Gets the importatn date and time elements from
    val_mrms and init_mrms so that ENS and SVR files
    can be build off of MRMS files that exist.
    
    Parameters
    -------------
    init_mrms : List of str
        All init MRMS files that exist
        based off of the ENS files
    val_mrms : List of str
        All the valid MRMS files that
        exist based off of the ENS files.
    
    Returns
    -------------
    cur_date : str
        The dates of the init file.
    cur_init_time : str
        The init times in 24 hr utc.
    cur_val_time
        The valid time in 24 hr utc.
    """
    files_dates = []
    cur_date = []
    cur_init_time = []
    cur_val_time = []
    for n,cur_init in enumerate(init_mrms):
        #wofs_MRMS_RAD_20200428_1920.nc
        #20190502_203500.nc
        if year == '2020':
            files_dates.append(re.findall("/20[1-2][0-9][0-1][0-9][0-3][0-9]/", cur_init))
            cur_date.append(re.findall("_20[1-2][0-9][0-1][0-9][0-3][0-9]_", cur_init))
            cur_init_time.append(re.findall('[0-9][0-9][0-9][0-9].nc', cur_init))
            cur_val_time.append(re.findall("[0-9][0-9][0-9][0-9].nc",val_mrms[n]))
        elif year == "2019":
            files_dates.append(re.findall("/20[1-2][0-9][0-1][0-9][0-3][0-9]/", cur_init))
            cur_date.append(re.findall("/20[1-2][0-9][0-1][0-9][0-3][0-9]_", cur_init))
            cur_init_time.append(re.findall('[0-9][0-9][0-9][0-9]00.nc', cur_init))
            cur_val_time.append(re.findall("[0-9][0-9][0-9][0-9]00.nc",val_mrms[n]))


    return files_dates, cur_date, cur_init_time, cur_val_time

def match_wofs_mrms(init_mrms, val_mrms, lead_time, year):
    """
    Based off of the dates from the init_mrms,
    builds the ENS and SVR files for a lead_time
    
    Parameter
    --------------
    init_mrms : List of str
        All init MRMS files that exist
        based off of the ENS files
    val_mrms : List of str
        All the valid MRMS files that
        exist based off of the ENS files.
    lead_time : str
        How long of a lead time between init and val.
        Must give with a leading 0 if less than 10.
        ex 09 is 45 minute forecast.
    
    Returns
    --------------
    complete_ens : list of str
        File paths of ENS files that match up with 
        init mrms and val mrms.

    complete_svr : list of str
        File paths of SVR files that match up with 
        init mrms and val mrms
    """
    complete_ens = []
    files_dates, dates, init_times, cur_val_time = _dates_times(init_mrms, val_mrms, year)

    for n,i in enumerate(dates):
        if year == '2020':
            complete_ens.append('/ourdisk/hpc/ai2es/wofs/%s_summary/%s/%s/wofs_ENS_%s_%s_%s_%s'\
                %(year, files_dates[n][0][1:-1], init_times[n][0][:-3], lead_time, i[0][1:-1], init_times[n][0][:-3], cur_val_time[n][0]))
        else:
            complete_ens.append('/ourdisk/hpc/ai2es/wofs/%s_summary/%s/%s/wofs_ENS_%s_%s_%s_%s.nc'\
                %(year, files_dates[n][0][1:-1], init_times[n][0][:-5], lead_time, i[0][1:-1], init_times[n][0][:-5], cur_val_time[n][0][:-5]))
            #/ourdisk/hpc/ai2es/wofs/2019_summary/20190503/0000/wofs_ENS_24_20190504_0000_0200.nc 
                

        complete_svr = [f.replace("ENS", "SVR") for f in complete_ens]

    return complete_ens, complete_svr


def get_ens_data(ens_data, lead_time, year):
    """
    Takes in the ENS data from a time step,
    extracts
    """
    comp_dz_max = []
    comp_dz_90 = []
    comp_dz_avg = []

    w_up_max = []
    w_up_90 = []
    w_up_avg = []

    w_down_max = []
    w_down_90 =[]
    w_down_avg =[]

    lat = []
    lon = []
    for i in ens_data:
        ens = xr.load_dataset(i, decode_cf = False, drop_variables = ['wz_0to2', 'uh_0to2'])
        if np.shape(ens['comp_dz'][0,:,:]) == (300,300) or np.shape(ens['comp_dz'][0,:,:]) == (250,250):
            if year == '2018' and np.shape(ens['comp_dz'][0,:,:]) == (300,300):
                pass
            else:
                #Storm Variables
                #max/min
                cur_w_up_max = np.float32(np.max(ens['w_up'], axis=0))
                cur_w_down_max = np.float32(np.min(ens['w_down'], axis=0))
                cur_comp_dz_max = np.float32(np.max(ens['comp_dz'], axis=0))
                print(np.shape(cur_w_down_max))

                #90th percentile
                cur_w_up_90 = np.float32(np.percentile(ens['w_up'], 90., axis=0))
                cur_w_down_90 = np.float32(np.percentile(ens['w_down'], 90., axis=0))
                cur_comp_dz_90 = np.float32(np.percentile(ens['comp_dz'],90., axis = 0))

                #averages
                cur_w_up_avg = np.float32(np.average(ens['w_up'], axis=0))
                cur_w_down_avg = np.float32(np.average(ens['w_down'], axis=0))
                cur_comp_dz_avg = np.float32(np.average(ens['comp_dz'], axis=0))

                lat.append(ens['xlat'])
                lon.append(ens['xlon'])

                #min-max normalize
                norm_w_up_max = min_max_norm(cur_w_up_max)
                norm_w_down_max = min_max_norm(cur_w_down_max)
                norm_comp_dz_max = min_max_norm(cur_comp_dz_max)

                norm_w_up_90 = min_max_norm(cur_w_up_90)
                norm_w_down_90 = min_max_norm(cur_w_down_90)
                norm_comp_dz_90 = min_max_norm(cur_comp_dz_90)

                norm_w_up_avg = min_max_norm(cur_w_up_avg)
                norm_w_down_avg = min_max_norm(cur_w_down_avg)
                norm_comp_dz_avg = min_max_norm(cur_comp_dz_avg)
                print('before padding: %s'%str(np.shape(norm_comp_dz_max)))

                #add in padding
                norm_comp_dz_max = add_padding(norm_comp_dz_max, year)
                norm_comp_dz_90 = add_padding(norm_comp_dz_90, year)
                norm_comp_dz_avg = add_padding(norm_comp_dz_avg, year)
                print('after padding: %s'%str(np.shape(norm_comp_dz_max)))

                norm_w_up_max = add_padding(norm_w_up_max, year)
                norm_w_up_90 = add_padding(norm_w_up_90, year)
                norm_w_up_avg = add_padding(norm_w_up_avg, year)

                norm_w_down_max = add_padding(norm_w_down_max,year)
                norm_w_down_90 = add_padding(norm_w_down_90, year)
                norm_w_down_avg = add_padding(norm_w_down_avg, year)


                print('Done Normalizing')
                comp_dz_max.append(norm_comp_dz_max)
                comp_dz_90.append(norm_comp_dz_90)
                comp_dz_avg.append(norm_comp_dz_avg)

                w_up_max.append(norm_w_up_max)
                w_up_90.append(norm_w_up_90)
                w_up_avg.append(norm_w_up_avg)

                w_down_max.append(norm_w_down_max)
                w_down_90.append(norm_w_down_90)
                w_down_avg.append(norm_w_down_avg)
        
                print('Done : %s'%i)
            
    print(np.shape(comp_dz_max))

    vars = [comp_dz_max, comp_dz_90, comp_dz_avg,
            w_up_max, w_up_90, w_up_avg,
            w_down_max, w_down_90, w_down_avg]
            
    names =['comp_dz_max' , 'comp_dz_90', 'comp_dz_avg',
            'w_up_max', 'w_up_90', 'w_up_avg', 
            'w_down_max', 'w_down_90', 'w_down_avg']

    size = ['n_samples','lat', 'lon']

    vars2 =[lat,lon]
    names2 = ['lat', 'lon']
    size2 = ['n_samples','NY', 'NX']


    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    tup = [(size2,var)for var in vars2]
    data_vars2  = {names2 : data_tups for names2, data_tups in zip(names2, tup)}



    out_ds = xr.Dataset(data_vars)
    out_ds2 = xr.Dataset(data_vars2)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/ENS_data_%s_%s.nc'%(year, lead_time))
    out_ds2.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/lat_lon_%s.nc'%year)

def get_svr_data(svr_data, lead_time, year):
    cape_ml = []
    cape_sfc = []
    cin_sfc = []
    cin_ml = []
    for cur_svr in svr_data:
        svr = xr.load_dataset(cur_svr, decode_cf = False)
        if np.shape(svr['cape_ml'][0,:,:]) == (300,300) or np.shape(svr['cape_ml'][0,:,:]) == (250,250):
            if year == '2018' and np.shape(svr['cape_ml'][0,:,:]) == (300,300):
                pass
            else:
                cur_cape_ml = np.float32(np.average(svr['cape_ml'], axis=0))
                cur_cape_sfc = np.float32(np.average(svr['cape_sfc'], axis=0))
                cur_cin_ml = np.float32(np.average(svr['cin_ml'], axis=0))
                cur_cin_sfc = np.float32(np.average(svr['cin_sfc'], axis=0))

                cur_cape_ml = min_max_norm(cur_cape_ml)
                cur_cape_sfc = min_max_norm(cur_cape_sfc)
                cur_cin_ml = min_max_norm(cur_cin_ml)
                cur_cin_sfc = min_max_norm(cur_cin_sfc)
                print('before padding: %s'%str(np.shape(cur_cape_ml)))

                #add in padding
                cur_cape_ml = add_padding(cur_cape_ml,year)
                cur_cape_sfc = add_padding(cur_cape_sfc,year)
                cur_cin_ml = add_padding(cur_cin_ml, year)
                cur_cin_sfc = add_padding(cur_cin_sfc, year)
                print('after padding: %s'%str(np.shape(cur_cape_ml)))

                cape_ml.append(cur_cape_ml)
                cape_sfc.append(cur_cape_sfc)
                cin_ml.append(cur_cin_ml)
                cin_sfc.append(cur_cin_sfc)
                print('Done : %s'%cur_svr)

    print(np.shape(cin_sfc))
    
    vars = [cape_ml,cape_sfc,cin_ml,cin_sfc]
    names =['cape_ml', 'cape_sfc', 'cin_ml', 'cin_sfc']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/SVR_data_%s_%s.nc'%(year, lead_time))

def get_init_data(init_data, lead_time, year):
    dz_cress = []
    for cur_init in init_data:
        #load in the data
        cur_init_data = xr.load_dataset(cur_init, decode_cf = False,drop_variables = ['XLAT', 'XLON', 'lon', 'lat'])

        if year == '2020' or year =='2021':
            cur_dz_cress = cur_init_data['dz_cress']
        else:
           cur_dz_cress = cur_init_data['DZ_CRESSMAN']
                        
        if np.shape(cur_dz_cress) == (300,300) or np.shape(cur_dz_cress) == (250,250):
            if year == '2018' and np.shape(cur_dz_cress) == (300,300):
                pass
            else:
                cur_dz_cress = min_max_norm(cur_dz_cress)
                print('before padding: %s'%str(np.shape(cur_dz_cress)))

                cur_dz_cress = add_padding(cur_dz_cress, year)
                dz_cress.append(cur_dz_cress)
                print('after padding: %s'%str(np.shape(cur_dz_cress)))
                print('done: %s'%cur_init)

    vars = [dz_cress]
    names =['dz_cress']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/init_mrms_data_%s_%s.nc'%(year, lead_time))


def get_val_data(val_data, lead_time, year):
    dz_cress = []
    dz_cress_class_np = []
    for cur_val in val_data:
        cur_val_data = xr.load_dataset(cur_val,decode_cf = False,drop_variables = ['XLAT', 'XLON', 'lon', 'lat'])
        
        if year == '2020' or year =='2021':
            cur_dz_cress = cur_val_data['dz_cress'].to_numpy()
        else:
            cur_dz_cress = cur_val_data['DZ_CRESSMAN'].to_numpy()
        
        if np.shape(cur_dz_cress) == (300,300) or np.shape(cur_dz_cress) == (250,250):
            if year == '2018' and np.shape(cur_dz_cress) == (300,300):
                pass
            else:
                #make into binary
                cur_dz_cress_class_np  = np.zeros(cur_dz_cress.shape)
                cur_dz_cress_class_np[cur_dz_cress >= 40] = 1
                print('before padding: %s'%str(np.shape(cur_dz_cress)))

                #add padding
                cur_dz_cress = add_padding(cur_dz_cress, year)
                cur_dz_cress_class_np = add_padding(cur_dz_cress_class_np, year)
                print('after padding: %s'%str(np.shape(cur_dz_cress)))

                dz_cress.append(cur_dz_cress)
                dz_cress_class_np.append(cur_dz_cress_class_np)
                print('done: %s'%cur_val)



    vars = [dz_cress, dz_cress_class_np]
    names =['dz_cress', 'dz_cress_binary']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/val_mrms_data_%s_%s.nc'%(year, lead_time))

def make_wofs_probs(ens_data,year,lead_time):
    ens_probs = []
    wofs_probs_padding = []
    for i in ens_data:
        ens = xr.load_dataset(i, decode_cf = False, drop_variables = ['wz_0to2', 'uh_0to2'])
        if np.shape(ens['comp_dz'][0,:,:]) == (300,300) or np.shape(ens['comp_dz'][0,:,:]) == (250,250):
            if year == '2018' and np.shape(ens['comp_dz'][0,:,:]) == (300,300):
                pass
            else:
                ens_binary = np.where(ens['comp_dz'] >= 40,1,0)
                print('ens binary: %s'%str(np.shape(ens_binary)))

                ens_probs.append(np.average(ens_binary,axis=0))

    print('ENS Probs shape : %s'%str(np.shape(ens_probs)))

    for n in range(np.size(ens_probs,axis=0)):
        wofs_probs_padding.append(add_padding(ens_probs[n], year))
    print('WoFS with padding: %s'%str(np.shape(wofs_probs_padding)))

    #save in a xr dataset
    vars = [wofs_probs_padding]
    names =['wofs_probs']
    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    print(out_ds)

    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/norm/wofs_probs_data_%s_%s.nc'%(year, lead_time))

 #/ourdisk/hpc/ai2es/wofs/MRMS/2020/2020/20200515/wofs_MRMS_RAD_20200515_2220.nc
 #/ourdisk/hpc/ai2es/wofs/MRMS/2019/2019/20190510/20190510_192000.nc
 
 #/ourdisk/hpc/ai2es/wofs/2020_summary/20200518/0000/wofs_ENS_48_20200519_0000_0400.nc 
 #/ourdisk/hpc/ai2es/wofs/2020_summary/20200518/0000/wofs_SVR_18_20200519_0000_0130.nc

 #/ourdisk/hpc/ai2es/wofs/2019_summary/20190503/0000/wofs_ENS_24_20190504_0000_0200.nc 
 #/ourdisk/hpc/ai2es/wofs/2019_summary/20190503/0000/wofs_SVR_24_20190504_0000_0200.nc