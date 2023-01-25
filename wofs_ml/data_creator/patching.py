import numpy as np
import xarray as xr
import os
import glob
import re
from datetime import datetime, timedelta
from data_manipulation import *
from patcher_helper import *


year = ['2017','2018','2019', '2020', '2021']
parser = create_parser() #from data_manipulation.py
args = parser.parse_args()

run_num = args.run_num

#Get the file paths of ENS and SVR
ENS_files = []
SVR_files = []
temp_ens = []

for i in year:
    temp_ens += glob.glob("/ourdisk/hpc/ai2es/wofs/{year}_summary/*/*/wofs_ENS_06_*".format(year = i))


for i in temp_ens:
    temp_SVR = i.replace("ENS", "SVR")
    if os.path.exists(temp_SVR) and os.path.getsize(i) > 0 and os.path.getsize(temp_SVR) > 0:
            ENS_files.append(i)
            SVR_files.append(temp_SVR)

ENS_files.sort()
SVR_files.sort()
print(np.size(ENS_files))
print(np.size(SVR_files))

def _get_datetime(cur_file):
    file_date = re.findall('/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/',cur_file)
    init_date = re.findall('_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_',cur_file)
    lead_time = re.findall('_[0-9][0-9]_',cur_file)
    init_time = re.findall('/[0-9][0-9][0-9][0-9]/',cur_file)
    val_time = re.findall('_[0-9][0-9][0-9][0-9].nc',cur_file)

    return file_date, init_date, lead_time, init_time, val_time
    #/ourdisk/hpc/ai2es/wofs/2021_summary/20210503/0000/wofs_ENS_58_20210504_0000_0450.nc 
    #/ourdisk/hpc/ai2es/wofs/2020_summary/20200429/0000/wofs_ENS_58_20200430_0000_0450.nc
    #/ourdisk/hpc/ai2es/wofs/2019_summary/20190430/0000/wofs_ENS_58_20190501_0000_0450.nc 
    #/ourdisk/hpc/ai2es/wofs/2018_summary/20180528/0030/wofs_ENS_06_20180529_0030_0030.nc
    #/ourdisk/hpc/ai2es/wofs/2017_summary/20170502/0000/wofs_ENS_12_20170503_0000_0100.nc

#init mrms
init_mrms_2021 = []
init_mrms_2020 = []
init_mrms_2019 = []
init_mrms_2018 = []
init_mrms_2017 = []

#Val mrms
val_mrms_2021 = []
val_mrms_2020 = []
val_mrms_2019 = []
val_mrms_2018 = []
val_mrms_2017 = []

init_mrms=[]
val_mrms=[]

for i in ENS_files:
    file_date, init_date, lead_time, init_time, val_time = _get_datetime(i)

    #/ourdisk/hpc/ai2es/wofs/MRMS/2021/RAD_AZS_MSH/20210406/wofs_MRMS_RAD_20210406_2225.nc 
    #/ourdisk/hpc/ai2es/wofs/MRMS/2020/2020/20200515/wofs_MRMS_RAD_20200515_2220.nc
    #/ourdisk/hpc/ai2es/wofs/MRMS/2019/2019/20190530/20190530_220000.nc 
    #/ourdisk/hpc/ai2es/wofs/MRMS/2018/2018/20180501/20180501-233000.nc
    #/ourdisk/hpc/ai2es/wofs/MRMS/2017/2017/20170502/20170502-204500.nc 

    if file_date[0][1:5] == '2021':
        temp_init = '/ourdisk/hpc/ai2es/wofs/MRMS/2021/RAD_AZS_MSH/{file_date}wofs_MRMS_RAD_{init_date}_{init_time}.nc'\
            .format(file_date = file_date[0], init_date = init_date[0][1:-1], init_time = init_time[0][1:-1])

    elif file_date[0][1:5] == '2020':
        temp_init = '/ourdisk/hpc/ai2es/wofs/MRMS/2020/2020{file_date}wofs_MRMS_RAD_{init_date}_{init_time}.nc'\
            .format(file_date = file_date[0], init_date = init_date[0][1:-1], init_time = init_time[0][1:-1])

    elif file_date[0][1:5] == '2019':
        temp_init = '/ourdisk/hpc/ai2es/wofs/MRMS/2019/2019{file_date}{init_date}_{init_time}00.nc'\
            .format(file_date=file_date[0], init_date = init_date[0][1:-1], init_time = init_time[0][1:-1])

    elif file_date[0][1:5] == '2018':
        temp_init = '/ourdisk/hpc/ai2es/wofs/MRMS/2018/2018{file_date}{init_date}-{init_time}00.nc'\
            .format(file_date=file_date[0], init_date = init_date[0][1:-1], init_time = init_time[0][1:-1])

    elif file_date[0][1:5] == '2017':
        temp_init = '/ourdisk/hpc/ai2es/wofs/MRMS/2017/2017{file_date}{init_date}-{init_time}00.nc'\
            .format(file_date=file_date[0], init_date = init_date[0][1:-1], init_time = init_time[0][1:-1])
  

    temp_init_time = init_date[0][1:-1]+"_"+init_time[0][1:-1]
    temp_init_time_1 = datetime.strptime(temp_init_time, '%Y%m%d_%H%M')
    temp_val_time = temp_init_time_1 + timedelta(minutes=(30))

    #for 2021 and 2020 MRMS
    if file_date[0][1:5] == '2020' or file_date[0][1:5] == '2021':
        #puts the time into the correct format
        temp_val = temp_val_time.strftime('%Y%m%d_%H%M')
        if os.path.exists(temp_init) and os.path.getsize(temp_init) > 0:
            temp_val_path = temp_init.replace(temp_init_time, temp_val)
            if os.path.exists(temp_val_path) and os.path.getsize(temp_val_path) > 0:
                if file_date[0][1:5] == '2020':
                    init_mrms_2020.append(temp_init)
                    val_mrms_2020.append(temp_val_path)
                    init_mrms.append(temp_init)
                    val_mrms.append(temp_val_path)
                if file_date[0][1:5] == '2021':
                    init_mrms_2021.append(temp_init)
                    val_mrms_2021.append(temp_val_path)
                    init_mrms.append(temp_init)
                    val_mrms.append(temp_val_path)

    #for 2019 MRMS
    elif file_date[0][1:5] == '2019':
        temp_val = temp_val_time.strftime('%Y%m%d_%H%M')
        if os.path.exists(temp_init) and os.path.getsize(temp_init) > 0 and init_date[0] != '_20190531_' and init_date[0] != '_20190530_':
            temp_val_path = temp_init.replace(temp_init_time, temp_val)
            if os.path.exists(temp_val_path) and os.path.getsize(temp_val_path) > 0:
                init_mrms_2019.append(temp_init)
                val_mrms_2019.append(temp_val_path)
                init_mrms.append(temp_init)
                val_mrms.append(temp_val_path)


    #for 2018 and 2017 MRMS
    else:
        temp_val = temp_val_time.strftime('%Y%m%d-%H%M')
        temp_init_time = temp_init_time.replace('_', '-')
        if os.path.exists(temp_init) and os.path.getsize(temp_init) > 0:
            temp_val_path = temp_init.replace(temp_init_time, temp_val)
            if os.path.exists(temp_val_path) and os.path.getsize(temp_val_path) > 0:
                if file_date[0][1:5] == '2018':
                    init_mrms_2018.append(temp_init)
                    val_mrms_2018.append(temp_val_path)
                    init_mrms.append(temp_init)
                    val_mrms.append(temp_val_path)
                if file_date[0][1:5] == '2017':
                    init_mrms_2017.append(temp_init)
                    val_mrms_2017.append(temp_val_path)
                    init_mrms.append(temp_init)
                    val_mrms.append(temp_val_path)



full_ens = []
full_svr = []

full_ens_2021 = []
full_ens_2020 = []
full_ens_2019 = []
full_ens_2018 = []
full_ens_2017 = []

full_svr_2021 = []
full_svr_2020 = []
full_svr_2019 = []
full_svr_2018 = []
full_svr_2017 = []


#get ENS path from the inital time
for n,i in enumerate(init_mrms):
    file_date = re.findall('/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/',i)
    init_date = re.findall('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_',i)

    if file_date[0][1:5] =='2021':
        #/ourdisk/hpc/ai2es/wofs/2021_summary/20210503/0000/wofs_ENS_58_20210504_0000_0450.nc 
        init_time = re.findall('_[0-9][0-9][0-9][0-9].nc', i)
        val_time = re.findall('_[0-9][0-9][0-9][0-9].nc', val_mrms[n])

        ens_path = '/ourdisk/hpc/ai2es/wofs/2021_summary{file_date}{init_time}/wofs_ENS_06_{init_date}{init_time}{val_time}'\
            .format(file_date = file_date[0], init_time = init_time[0][1:-3],init_date = init_date[0] ,val_time = val_time[0])
        
    elif file_date[0][1:5] =='2020':
        #/ourdisk/hpc/ai2es/wofs/2020_summary/20200429/0000/wofs_ENS_58_20200430_0000_0450.nc
        init_time = re.findall('_[0-9][0-9][0-9][0-9].nc', i)
        val_time = re.findall('_[0-9][0-9][0-9][0-9].nc', val_mrms[n])

        ens_path = '/ourdisk/hpc/ai2es/wofs/2020_summary{file_date}{init_time}/wofs_ENS_06_{init_date}{init_time}{val_time}'\
            .format(file_date = file_date[0], init_time = init_time[0][1:-3],init_date = init_date[0] ,val_time = val_time[0])

    elif file_date[0][1:5] == '2019':
        #/ourdisk/hpc/ai2es/wofs/2019_summary/20190430/0000/wofs_ENS_58_20190501_0000_0450.nc 
        init_time = re.findall('_[0-9][0-9][0-9][0-9]00.nc', i)
        val_time = re.findall('_[0-9][0-9][0-9][0-9]00.nc', val_mrms[n])
        ens_path = '/ourdisk/hpc/ai2es/wofs/2019_summary{file_date}{init_time1}/wofs_ENS_06_{init_date}{init_time2}_{val_time}.nc'\
            .format(file_date = file_date[0], init_time1 = init_time[0][1:-5],init_date = init_date[0], init_time2= init_time[0][1:-5], val_time = val_time[0][1:-5])
    
    elif file_date[0][1:5] == '2018':
        #/ourdisk/hpc/ai2es/wofs/2018_summary/20180528/0030/wofs_ENS_06_20180529_0030_0030.nc

        init_date = re.findall('/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-', i)
        init_time = re.findall('-[0-9][0-9][0-9][0-9]00.nc', i)
        val_time = re.findall('-[0-9][0-9][0-9][0-9]00.nc', val_mrms[n])
        ens_path = '/ourdisk/hpc/ai2es/wofs/2018_summary{file_date}{init_time1}/wofs_ENS_06_{init_date}_{init_time2}_{val_time}.nc'\
            .format(file_date = file_date[0], init_time1 = init_time[0][1:-5],init_date = init_date[0][1:-1], init_time2= init_time[0][1:-5], val_time = val_time[0][1:-5])

    elif file_date[0][1:5] == '2017':
        #/ourdisk/hpc/ai2es/wofs/2017_summary/20170502/0000/wofs_ENS_12_20170503_0000_0100.nc
        init_date = re.findall('/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-', i)
        init_time = re.findall('-[0-9][0-9][0-9][0-9]00.nc', i)
        val_time = re.findall('-[0-9][0-9][0-9][0-9]00.nc', val_mrms[n])
        ens_path = '/ourdisk/hpc/ai2es/wofs/2017_summary{file_date}{init_time1}/wofs_ENS_06_{init_date}_{init_time2}_{val_time}.nc'\
            .format(file_date = file_date[0], init_time1 = init_time[0][1:-5],init_date = init_date[0][1:-1], init_time2= init_time[0][1:-5], val_time = val_time[0][1:-5])
    

    #checks to make sure all files exist and are not empty
    if os.path.exists(ens_path) and os.path.getsize(ens_path) > 0:
        svr_path = ens_path.replace("ENS", "SVR")
        if os.path.exists(svr_path) and os.path.getsize(svr_path) > 0:

            full_ens.append(ens_path)
            full_svr.append(svr_path)

            if file_date[0][1:5] == '2021':
                full_ens_2021.append(ens_path)
                full_svr_2021.append(svr_path)
            elif file_date[0][1:5] =='2020':
                full_ens_2020.append(ens_path)
                full_svr_2020.append(svr_path)
            elif file_date[0][1:5] =='2019':
                full_ens_2019.append(ens_path)
                full_svr_2019.append(svr_path)
            elif file_date[0][1:5] =='2018':
                full_ens_2018.append(ens_path)
                full_svr_2018.append(svr_path)
            elif file_date[0][1:5] =='2017':
                full_ens_2017.append(ens_path)
                full_svr_2017.append(svr_path)

#checking to make sure everything is lined up
print('Number of Cases')
print('2021')
print('ENS : %d'%np.size(full_ens_2021))
print('SVR : %d'%np.size(full_svr_2021))
print('Init : %d'%np.size(init_mrms_2021))
print('Val : %d'%np.size(val_mrms_2021))
print('2020')
print('ENS : %d'%np.size(full_ens_2020))
print('SVR : %d'%np.size(full_svr_2020))
print('Init : %d'%np.size(init_mrms_2020))
print('Val : %d'%np.size(val_mrms_2020))
print('2019')
print('ENS : %d'%np.size(full_ens_2019))
print('SVR : %d'%np.size(full_svr_2019))
print('Init : %d'%np.size(init_mrms_2019))
print('Val : %d'%np.size(val_mrms_2019))
print('2018')
print('ENS : %d'%np.size(full_ens_2018))
print('SVR : %d'%np.size(full_svr_2018))
print('Init : %d'%np.size(init_mrms_2018))
print('Val : %d'%np.size(val_mrms_2018))
print('2017')
print('ENS : %d'%np.size(full_ens_2017))
print('SVR : %d'%np.size(full_svr_2017))
print('Init : %d'%np.size(init_mrms_2017))
print('Val : %d'%np.size(val_mrms_2017))
print('total cases')
print('ENS : %d'%np.size(full_ens))
print('SVR : %d'%np.size(full_svr))
print('Init : %d'%np.size(init_mrms))
print('Val : %d'%np.size(val_mrms))
print()

#file paths
print('File Paths:')
print('2021')
print(full_ens_2021[100])
print(full_svr_2021[100])
print(init_mrms_2021[100])
print(val_mrms_2021[100])
print('2020')
print(full_ens_2020[100])
print(full_svr_2020[100])
print(init_mrms_2020[100])
print(val_mrms_2020[100])
print('2019')
print(full_ens_2019[100])
print(full_svr_2019[100])
print(init_mrms_2019[100])
print(val_mrms_2019[100])
print('2018')
print(full_ens_2018[100])
print(full_svr_2018[100])
print(init_mrms_2018[100])
print(val_mrms_2018[100])
print('2017')
print(full_ens_2017[100])
print(full_svr_2017[100])
print(init_mrms_2017[100])
print(val_mrms_2017[100])

#load and save data in parallel
if run_num == 0:
    get_ens_data(full_ens_2021, '06', '2021')

elif run_num == 1:
    get_svr_data(full_svr_2021, '06', '2021')

elif run_num == 2:
    get_ens_data(full_ens_2020, '06', '2020')

elif run_num == 3:
    get_svr_data(full_svr_2020, '06', '2020')

elif run_num == 4:
    get_ens_data(full_ens_2019,'06', '2019')

elif run_num == 5:
    get_svr_data(full_svr_2019, '06', '2019')

elif run_num == 6:
    get_ens_data(full_ens_2018,'06', '2018')

elif run_num == 7:
    get_svr_data(full_svr_2018, '06', '2018')

elif run_num == 8:
    get_ens_data(full_ens_2017,'06', '2017')

elif run_num == 9:
    get_svr_data(full_svr_2017, '06', '2017')

elif run_num == 10:
    get_init_data(init_mrms_2021, '06', '2021')

elif run_num == 11:
    get_val_data(val_mrms_2021, '06', '2021')

elif run_num == 12:
    get_init_data(init_mrms_2020, '06', '2020')

elif run_num == 13:
    get_val_data(val_mrms_2020, '06', '2020')

elif run_num == 14:
    get_init_data(init_mrms_2019, '06', '2019')

elif run_num == 15:
    get_val_data(val_mrms_2019, '06', '2019')

elif run_num == 16:
    get_init_data(init_mrms_2018, '06', '2018')

elif run_num == 17:
    get_val_data(val_mrms_2018, '06', '2018')

elif run_num == 18:
    get_init_data(init_mrms_2017, '06', '2017')

elif run_num == 19:
    get_val_data(val_mrms_2017, '06', '2017')   
elif run_num == 20:
    make_wofs_probs(full_ens_2017, '2017', '06')
    make_wofs_probs(full_ens_2018, '2018', '06')
elif run_num == 21:
    make_wofs_probs(full_ens_2019, '2019', '06')
    make_wofs_probs(full_ens_2020, '2020', '06')
elif run_num == 22:
    make_wofs_probs(full_ens_2021, '2021', '06')

print('data is loaded')
print('saved')