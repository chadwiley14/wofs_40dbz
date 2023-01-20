#Helper function to run the storm tracking notebook
#Works with the storm_motion.ipynb 
#Needs MontePython python package
#Written by Chad Wiley
from curses.ascii import NL
import sys, os
from time import time 
current_dir = os.getcwd()
path = os.path.dirname(current_dir)
sys.path.append(path)

#imports all py files within monte_python
import MontePython_master.monte_python as mp
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib

#cartopy imports
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature

import re
import xarray as xr
import imageio

from glob import glob
from pathlib import Path
from scipy import ndimage, misc
from IPython.core.debugger import set_trace
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
plt.rcParams["animation.html"] = "jshtml"

def make_dir(full_date, fig_dir):
    """
    This creates all the directories based on the date,
    then creates folders for the storm object, gifs, and
    total ci.

    Parameters
    --------------
    full_date : string
        This is the date of the event. Used to create dir.

    Returns
    ---------------
    storm_obj_path : string
        This is the path of the storm_obj png will go
    
    storm_evo_path : string
        This is the path of the storm_evo gif will go
    
    tot_ci_path : string
        This is the path of the tot_ci png will go
    """
    if os.path.exists(fig_dir) == False:
        os.mkdir(fig_dir)
        os.mkdir(fig_dir+ '/storm_obj')
        os.mkdir(fig_dir+ '/storm_obj/sbs')
        os.mkdir(fig_dir+ '/storm_obj/overlap')
        os.mkdir(fig_dir+'/storm_evo_gifs')
        os.mkdir(fig_dir+'/tot_ci')
    
    storm_obj_path = './figs/figs_%8s/storm_obj/'%full_date
    storm_evo_path = './figs/figs_%8s/storm_evo_gifs/'%full_date
    tot_ci_path = './figs/figs_%8s/tot_ci/'%full_date

    return storm_obj_path, storm_evo_path, tot_ci_path


def load_projection(full_date):
    """
    Loads in the lat and lon from summary files
    so that the proper projection can be created

    Parameters
    ----------------------
    full_date : string
        The full date in YYYYMMDD format.
    
    Returns
    -----------------------
    projection : object
        This is the projection of the domain.
    
    lats : list
        This is a list of all the lats within the domain.

    lons : list
        This is a list of all the lons within the domain.
    """

    init_time = [p for p in os.listdir(f'/work/mflora/SummaryFiles/{full_date}') if len(p)==4][0]
    indir = glob(f'/work/mflora/SummaryFiles/{full_date}/{init_time}/wofs_ENS_*')[0]

    ds = xr.load_dataset(indir, decode_times=False)
    lats = ds['xlat']
    lons = ds['xlon']

    central_longitude = ds.attrs['STAND_LON']
    central_latitude = ds.attrs['CEN_LAT']

    standard_parallels = (ds.attrs['TRUELAT1'], ds.attrs['TRUELAT2'])
    projection=ccrs.LambertConformal(central_longitude=central_longitude,
                                     central_latitude=central_latitude,
                                     standard_parallels=standard_parallels)
    
    return projection, lats, lons


    
def data_loader(path, base, max_timestep):
    """
    Loads the data from a date and
    returns the composite reflectivity
    at each time step

    Parameter
    -------------------
    path : string
        The path of the data wanting to exstract, ex: path = '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2020/20200619/

    base : string
        The base of the file of interest, ex: base = 'wofs_MRMS_' Can add more to get a specific time period.

    max_timesteps : integer, default: None
        The maximum number of timesteps wanted. Can be used to limit size of files

    Returns
    ---------------------
    dataset : list
        A 2D list of dBZ values for each point and each time step
    
    time : list
        A list of each time, used to add time into the plotting.
    
    projection : object
        Returns the projection from the domain for plotting.
    """

    #loads all files matching within the path that match the base
    files = [f for f in os.listdir(path) if re.match(r'%s.+.nc'%(base), f)]

    files.sort()
    dataset = []
    time=[]

    for n,i in enumerate(files):
        if n < max_timestep:
            #loads each timestep the dBz and the time into an array
            cur_file = xr.load_dataset(path+i, drop_variables = ['lat', 'lon'])   
            dataset.append(cur_file['dz_cress'])
            time.append(i[-7:-3])

    return dataset, time


def find_new_CI(matched_new, storm_labels_new):
    """
    This takes the new time step and finds if there
    is any new convection that was not present at 
    previous time step. Used within track_storm_evolution.
    If new, it is marked with 1. If existing, marked as -1.

    Parameters
    ---------------------
    matched_new : list
        The current time step labels that are matched to the privous time step.
    
    strom_labels_new: list
        The current time step storm labels, from MontePython.labels.

    Return
    -----------------------
    unmatched_arr : list
        A 2D array with 1, 0, -1 denoting the status of each point.

    """
    unmatched = []
    #gets the unique labels that are not in the previous step
    new_objects = np.unique(storm_labels_new)[1:]
    for label in new_objects:
        if label not in matched_new:
            unmatched.append(label)

    unmatched_arr = np.zeros(storm_labels_new.shape)

    for label in new_objects:
        #This is new CI
        if label in unmatched:
            unmatched_arr[storm_labels_new == label] = 1
        else:
            #This is NOT new CI
            unmatched_arr[storm_labels_new == label] = -1
    return unmatched_arr
    

def track_storm_evolution(data, output_path, time, projection, lons, lats, plot_side = True,
                          plot_ol = True, max_size = 5, cent_dist = 30, min_dist = 15,
                          time_max = 0, score_thresh = 0.2, method = 'single_threshold',
                          bdry = 35, qc_params = [('min_area', 12)]):
    
    """
    This takes in the data from a file and create storm objects,
    identifies new convection at each time step, and plots the objects.
    Red is new convection, blue is existing. This is the main function.

    Parameters
    ----------------
    data : list
        This is the input data, given in dBz. Use the load_data function.
        
    output_path : string
        This is where each time step plot will be saved
    
    time: list of stings
        This is the time data that is given back by the data loader, in UTC.
    
    projection : object
        This is the projection for the plotting.
    
    lon : list
        This is the lonitudes within the domain.
    
    lat : list
        List of all the latitudes within the domain.
    
    plot_side : boolean, default = True
        Asking whether to plot the side by side plots.

    plot_ol : boolean, default = True,
        Asking whether to plot the overlaping plots.

    max_size : interger
        This is the size of the max filter size for the finding new CI polygon.

    cent_dist : integer, default: 30
        From the object tracking portion of the MontePython package,
        it is the max distance from the center of the centroid.
    
    min_dist : integer, default: 15
        From the object tracking portion of the MontePython package,
        it is the minimum distance a centroid must be from each other
        to be considered seperate.

    time_max : integer, default: 0
        From the object tracking portion of the MontePython package,
        the minimum time seperation needed in order to be considered
        a new object

    score_thresh : float, default: 0.2
        From the object tracking portion of the MontePython package,
        the score needed to match objects together.
    
    method : string, default: single_threshold
        The method needed to find centroids, in most cases use single_theshold.

    bdry : integer, default: 35
        The dBz needed to be considered an object
    
    qc_param : list, default: [('min_area',12)]
        Quality Control parameter, from the MontePython package.

    Return
    --------------------
    tot_new_CI: 2d list of integers
        This is a list of location of where all new CI takes place over the course of the event.
    """

    qcer = mp.QualityControler()
    obj_match = mp.ObjectMatcher(cent_dist_max = cent_dist, min_dist_max = min_dist, time_max= time_max, score_thresh= score_thresh, 
                           one_to_one = False)

    for n, input_data in enumerate(data):
        #gets the first timestep since there is nothing to compare it to
        if n == 0:
            #gets the storm labels and object labels from that timestep
            storm_labels0, object_props0 = mp.label(input_data = input_data,
                                              method = method,
                                              return_object_properties = True,
                                              params = {'bdry_thresh':bdry})

            #Ensure the correct objects  are being identified
            storm_labels0, object_props0 = qcer.quality_control(input_data, storm_labels0, object_props0, qc_params)
        
        
            matched_0, matched_1, _= obj_match.match_objects(storm_labels0, storm_labels0)
        
            storm_matches = find_new_CI(matched_new = matched_0, storm_labels_new = storm_labels0)
            tot_new_CI = np.zeros(np.shape(storm_labels0))

            #plots the side by side plots
            if plot_side == True:
                plot_sbs(storm_matches = storm_matches, input_data = input_data,
                        output_path = output_path, cur_time = time[n],
                        projection = projection, lons = lons, lats = lats)

            #plots the overlaping plots
            if plot_ol == True:
                plot_overlap(storm_matches = storm_matches, max_size = max_size,
                             input_data = input_data, output_path = output_path, cur_time = time[n],
                             projection = projection, lons = lons, lats = lats)
                

        
        #Goes from the 2nd timestep until the end/max_timesteps
        elif n >=1:
            #gets old data, in future should make so it doesnt have to keep loading data already loaded.
            storm_labels0, object_props0 = mp.label(input_data = data[n-1],
                                              method = method,
                                              return_object_properties = True,
                                              params = {'bdry_thresh':bdry})

        
            storm_labels0, object_props0 = qcer.quality_control(data[n-1], storm_labels0, object_props0, qc_params)

            #gets new data
            storm_labels1, object_props1 = mp.label(input_data = input_data,
                                              method = method,
                                              return_object_properties = True,
                                              params = {'bdry_thresh':bdry})

            storm_labels1, object_props1 = qcer.quality_control(input_data, storm_labels1, object_props1, qc_params)
        
            #matches old time step to new time step
            matched_0, matched_1, _ = obj_match.match_objects(storm_labels0, storm_labels1)
            #finds where new convection is and makes it equal to 1
            storm_matches = find_new_CI(matched_new = matched_1, storm_labels_new = storm_labels1)

            
            #gets all the new convection and puts it into one array
            row = 0
            col = 0
            for i in storm_matches:
                for j in i:
                    if j == 1:
                        tot_new_CI[row][col] = 1
                    col = col + 1
                row = row + 1
                col = 0
        


            #plots the side by side plots
            if plot_side == True:
                plot_sbs(storm_matches = storm_matches, input_data = input_data,
                        output_path = output_path, cur_time = time[n],
                        projection = projection, lons = lons, lats = lats)

            #plots the overlaping plots
            if plot_ol == True:
                plot_overlap(storm_matches = storm_matches, max_size = max_size,
                             input_data = input_data, output_path = output_path, cur_time = time[n],
                             projection = projection, lons = lons, lats = lats)

    return tot_new_CI             


def animate(storm_obj_path, storm_evo_path, full_date, duration = 0.5):
    """
    This takes in a folder of images and returns a gif of those images.

    Parametes
    ---------------
    storm_obj_path : string
        This path that the input images are located. ex: './figs'
    
    storm_evo_path : string
        The name of the gif that is saved, will add .gif to the end of the string.

    full_date : string
        This is the full date of the event

    duration : float, default: 0.5
        The duration of each image will be shown, given in seconds
    """
    images_ol = list()
    images_sbs = list()

    for file in Path(storm_obj_path+'overlap').iterdir():
        if not file.is_file():
            continue
        images_ol.append(imageio.imread(file))

    for file in Path(storm_obj_path+'sbs').iterdir():
        if not file.is_file():
            continue
        images_sbs.append(imageio.imread(file))
    #add in regular expression to match sbs and overlapping images.
    imageio.mimwrite(storm_evo_path + 'OL_storm_evo_%8s.gif'%full_date, images_ol, duration = duration)
    imageio.mimwrite(storm_evo_path + 'sbs_storm_evo_%8s.gif'%full_date, images_sbs, duration = duration)


def new_CI_plot(new_CI, full_date, tot_ci_path, max_size):
    """
    This takes in a list of of 0 and 1
     and plots the numbers. The 1's represent
     locations of all new convection from that event.
     Saves the plot in the current dir.

     Parameters
     -----------------
     new_CI : list of ints
        This is the list with the locations of all the new convections from the event
        
     full_date : string
        This is the date of the event
     
     tot_ci_path : string
        This is where the png will be saved.
     
     max_size : integer
        The size of the max filter.
    """
    plt.figure()
    plt.pcolormesh(new_CI, cmap = 'bwr', vmax=1, vmin=-1)
    plt.colorbar()
    plt.title("Locations of New Convection for %8s"%full_date)
    plt.savefig(tot_ci_path + 'new_CI_%8s_maxfilter_%d.png'%(full_date,max_size))
    plt.close()


def create_metadata(full_date, max_timestep, cent_dist, min_dist,
                    time_max, score_thresh, max_size):
    """
    This creates a metadata text file which allows the
    reader be able to tell what each parameter was for each run.
    
    Parmeters:
    --------------
    full_date : String
        This is the date of the event.

    max_timestep : integer
        This is number of timestep wanted.
    
    cent_dist : integer
        Max distance from the center of one centroid to the next.
    
    min_dist : interger
        Min distance from one centroid to next.
    
    time_max : integer
        This is the time between steps allowed in order to count as differnt cells.

    score_thresh : float
        The score needed to match storms together.
    
    max_size : integer
        This is the max filter size.
    """
    with open('./figs/figs_%8s/README.txt'%full_date, 'w') as f:
        f.write('For the event on %8s'%full_date)
        f.write('\nNumber of Timesteps: %2d'%max_timestep)
        f.write('\nCenter distance max: %2d'%cent_dist)
        f.write('\nMin Dist: %2d'%min_dist)
        f.write('\nTime Max: %1d'%time_max)
        f.write('\nScore Threshold : %.1f'%score_thresh)
        f.write('\nMax Filter Size: %d'%max_size)


def plot_sbs(storm_matches, input_data, output_path, cur_time, projection, lons, lats):
    """
    This plots the data side-by-side with storm object the left,
    and dBz on the right.

    Parameters
    ------------
    storm_matches : list
        This is the locations of new convection for the timestep

    input_data : list
        This is the dBz.
    
    output_path : string
        This is where the png's will be saved
    
    cur_time : string
        The current time, within the event in UTC.

    projection : object
        This is the projection for the plotting
    
    lon : list
        This is the lonitudes within the domain.
    
    lat : list
        List of all the latitudes within the domain.
    """
    #Plots the new ci and current reflectivity next to each other
    crs = ccrs.PlateCarree()
    nws_levels = np.arange(20.,80.,5.)
    nws_cmap = get_nws_cmap()


    #create the figure
    fig, ax = plt.subplots(figsize = (10,8), facecolor = 'w',
                          dpi = 170, nrows=1, ncols=2, subplot_kw = {'projection': projection},
                          constrained_layout=True)
    
    #plot the storms in each box
    input_data = np.ma.masked_where(input_data==0,input_data)
    storm_matches = np.ma.masked_where(storm_matches==0,storm_matches)
    ax[0].pcolormesh(lons, lats, storm_matches, cmap='bwr', vmin = -1, vmax = 1, transform=crs)
    cf = ax[1].contourf(lons,lats,input_data, cmap = nws_cmap, transform = crs, levels = nws_levels)

    #add in states, lakes, and coastlines
    add_map_stuff(ax[0])
    add_map_stuff(ax[1])

    ax[0] = set_extent(ax[0], projection, crs, lats, lons)
    ax[1] = set_extent(ax[1], projection, crs, lats, lons)

    #add a colorbar
    fig.colorbar(cf, label = 'dBz', drawedges = False, spacing = 'uniform', location = 'bottom', shrink = 0.6)

    #set titles
    fig.suptitle('Storm Evolution at Time : %4s UTC'%cur_time, fontsize = 20)
    ax[0].title.set_text('Storm Objects')
    ax[1].title.set_text('Reflectivity in dBz')

    #Save figures
    fig.savefig(output_path+'sbs/storm_obj_%4s_sbs'%cur_time)
    plt.close()



def plot_overlap(storm_matches, max_size, input_data, output_path,
                cur_time, projection, lons, lats):
    """
    This plots the identified new convection on top of the dBz.

    Parameters
    ------------
    storm_matches : list
        This is the locations of new convection for the timestep

    max_size : integer
        This is the size of the max_filter.

    input_data : list
        This is the dBz.
    
    output_path : string
        This is where the png's will be saved.
    
    cur_time : string
        The current time, within the event in UTC.
    
    projection : object
        This is the projection for the plotting.
    
    lon : list
        This is the lonitudes within the domain.
    
    lat : list
        List of all the latitudes within the domain.
    """
    #Finds all places where storm label is -1 and changes to 0
    row = 0
    col = 0
    for i in storm_matches:
        for j in i:
            if j == -1:
                storm_matches[row][col] = 0
            col = col + 1
        row = row + 1
        col = 0

    crs = ccrs.PlateCarree()
    nws_levels = np.arange(20.,80.,5.)
    nws_cmap = get_nws_cmap()

    #create the figure
    fig, ax = plt.subplots(figsize = (10,8), facecolor = 'w',
                          dpi = 170, nrows=1, ncols=1, subplot_kw = {'projection': projection},
                          constrained_layout=True)
    
    #plot the storm
    input_data = np.ma.masked_where(input_data==0,input_data)
    storm_matches = np.ma.masked_where(storm_matches==0,storm_matches)

    cf = ax.contourf(lons,lats,input_data, cmap = nws_cmap, transform = crs, levels = nws_levels)

    #add in the max filter box
    ax.pcolormesh(lons, lats, ndimage.maximum_filter(input = storm_matches, size = max_size),
                  cmap = 'bwr', vmax = 1, vmin = -1, alpha = 0.3, transform = crs)
    
    #add in states, lakes, and coastlines
    add_map_stuff(ax)

    ax = set_extent(ax, projection , crs, lats, lons)

    fig.suptitle('Storm Evolution at Time: %4s UTC'%cur_time, fontsize = 20)

    #add a colorbar
    fig.colorbar(cf, label = 'dBz', drawedges = False, spacing = 'uniform', location = 'bottom', shrink = 0.6)

    #save the figure
    fig.savefig(output_path+'overlap/storm_obj_%4s_ol'%cur_time)
    plt.close()

def add_map_stuff(ax):
    """
    This is a helper function to add in states
    lakes, coastlines, ect.
    
    Parameters
    ----------------
    ax : matplotlib plot
        This is the plot of interest
    """
    #download the states
    states = NaturalEarthFeature(category="cultural", scale="10m",
                             facecolor="none",
                             name="admin_1_states_provinces")

    ax.add_feature(states, linewidth=.1, facecolor='none', edgecolor="black")
    ax.add_feature(cfeature.LAKES, linewidth=.1, facecolor='none', edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=.1, facecolor='none', edgecolor="black")


def set_extent(ax, projection , crs, lats, lons,):
    """ Set the Map extent based the WoFS domain """
    # Set the extent. 
    xs, ys, _ = projection.transform_points(
            crs,
            np.array([lons.min(), lons.max()]),
            np.array([lats.min(), lats.max()])).T
    _xlimits = xs.tolist()
    _ylimits = ys.tolist()

    # The limit is max(lower bound), min(upper bound). This will create 
    # a square plot and make sure there is no white spaces between the map
    # the bounding box created by matplotlib. This also allows us to set the
    # WoFS domain boundaries in cases where we aren't plotting WoFS data 
    # (e.g., storm reports, warning polygons, etc.) 
    lims = (max([_xlimits[0]]+[_ylimits[0]]),min([_xlimits[-1]]+[_ylimits[-1]]))
        
    ax.set_xlim(lims)
    ax.set_ylim(lims) 
    
    return ax

def get_nws_cmap():
    """
    This is the color map for reflectivity for the 
    National Weather Service"""
    c5 =  (0.0,                 0.9254901960784314, 0.9254901960784314)
    c10 = (0.00392156862745098, 0.6274509803921569, 0.9647058823529412)
    c15 = (0.0,                 0.0,                0.9647058823529412)
    c20 = (0.0,                 1.0,                0.0)
    c25 = (0.0,                 0.7843137254901961, 0.0)
    c30 = (0.0,                 0.5647058823529412, 0.0)
    c35 = (1.0,                 1.0,                0.0)
    c40 = (0.9058823529411765,  0.7529411764705882, 0.0)
    c45 = (1.0,                 0.5647058823529412, 0.0)
    c50 = (1.0,                 0.0,                0.0)
    c55 = (0.8392156862745098,  0.0,                0.0)
    c60 = (0.7529411764705882,  0.0,                0.0)
    c65 = (1.0,                 0.0,                1.0)
    c70 = (0.6,                 0.3333333333333333, 0.788235294117647)
    c75 = (0.0,                 0.0,                0.0) 

    nws_cmap = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, 
                 c50, c55, c60, c65, c70])

    return nws_cmap
