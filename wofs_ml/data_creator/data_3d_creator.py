from data_3d_helper import *

#This data is used to grab all the example and label data
#from the 3d unet and create data the is usable for the 3d unet

#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num


if run_num == 0:
    get_2017_2018_data()
elif run_num == 1:
    get_2019_data()
elif run_num == 2:
    get_2020_2021_data()
elif run_num == 3:
    merge_examples()
elif run_num == 4:
    labels_2017 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels/*')
    labels_2017.sort()
    get_labels(labels_2017,'2017-2018')
elif run_num == 5:
    labels_2019 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_2019/*')
    labels_2019.sort()
    get_labels(labels_2019,'2019')
elif run_num == 6:
    labels_2020 = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_2020-2021/*')
    labels_2020.sort()
    get_labels(labels_2020,'2020-2021')

elif run_num==7:
    labels_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_2017-2018.nc')
    lat_lon_2017 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/lat_lon_2017-2018.nc')

    labels_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_2019.nc')
    lat_lon_2019 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/lat_lon_2019.nc')

    labels_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/labels_2020-2021.nc')
    lat_lon_2020 = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/lat_lon_2020-2021.nc')


    full_labels = xr.concat([labels_2017,labels_2019,labels_2020], dim ='n_samples')
    full_lat_lon = xr.concat([lat_lon_2017,lat_lon_2019,lat_lon_2020], dim='n_samples')

    full_labels.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/full_labels.nc')
    full_lat_lon.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/3d_patches/full_lat_lon.nc')




