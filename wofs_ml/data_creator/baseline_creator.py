import numpy as np
import glob
import xarray as xr
import argparse
import scipy

def create_parser():
    """
    This gets the informaiton from the command line
    that is needed to run the program
    """

    parser = argparse.ArgumentParser(description="files given",
                                    fromfile_prefix_chars ='@')

    parser.add_argument('--run_num', type=int, default=None)
    parser.add_argument('--neighborhood_size', type = int, default = 15)

    return parser

def apply_neighborhood(ds, size):
    """
    Takes in a ds with the dim (n_samples, ne, lat, lon)
    loops through each ensemble members, makes it binary,
    and applies a max filter on te numbers. From there makes it into
    a probablity.

    PARAMETERS
    -------------------
    ds : xr dataset
        dataset with comp_dz_max as the only variable
        and dims of (n_samples, ne, lat, lon)

    size : int
        size of the neighborhood

    RETURNS
    --------------------
    comp_prob : list
        A list with all the probs for each n_samples
        over all the ens members

    """
    comp_prob = []
    #load in the ds
    ds = xr.load_dataset(ds)

    dz_binary = np.where(ds['comp_dz_max'] >=40, 1,0)

    dz_neigh = scipy.ndimage.maximum_filter(dz_binary, size)

    for n in range(np.size(ds['comp_dz_max'], axis =0)):
        comp_prob.append(np.average(dz_neigh[n],axis =0))

    return comp_prob

def save_probs(ds, year):
    #make in xr dataset
    vars = [ds]
        
    names =['com_prob']

    size = ['n_samples','lat', 'lon']

    tuples = [(size,var)for var in vars]
    data_vars = {name : data_tup for name, data_tup in zip(names, tuples)}

    out_ds = xr.Dataset(data_vars)
    out_ds.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/probs/probs_neigh_%s.nc'%year) 
    print('saved to disk')


#This program take in the data from ensemble composite dBZ
#and makes it into a prob

#create parser
parser = create_parser()
args = parser.parse_args()

run_num = args.run_num
neighborhood_size = args.neighborhood_size


#find all the files
files = glob.glob('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/probs/*')
files.sort()



#run in parallel
ds = apply_neighborhood(files[run_num], neighborhood_size)

    #save file

