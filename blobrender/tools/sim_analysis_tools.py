import numpy as np
import scipy
from scipy.interpolate import griddata
import os 
import pyPLUTO as pypl
import pyPLUTO.pload as pp
from . import basics as b

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_data_obj(ddir, num=-1, data_type='dbl'):
    """
    Function to load a .dbl file into an object ad a corresponding image

    Parameters
    ----------
    wdir_in_PLUTO : string
        name of local directory in $PLUTO_DIRs
    num : int
        timestep to load, default is the last timestep, or else specify between zero and last step

    Returns
    -------
    D : Data object
        pyPLUTO data object with all attributes from .dbl file
    I : Image object
        pyPLUTO image object made from Data object 

    """
    data_dir = ddir+'/'
    if num==-1:
        nlinf = pypl.nlast_info(w_dir=data_dir)
        print("data type: "+data_type)
        D = pp.pload(nlinf['nlast'],w_dir=data_dir,datatype=data_type) # Loading the data into a pload object D.
    else: 
        print("data type: "+data_type)
        D = pp.pload(num,w_dir=data_dir,datatype=data_type) # Loading the data into a pload object D.
    return D

def get_max_step(local_wdir):
    """_summary_

    Args:
        local_wdir (_type_): _description_

    Returns:
        _type_: _description_
    """
    plutodir = os.environ['PLUTO_DIR']
    full_wdir = plutodir+local_wdir
    try:
        nlinf = pypl.nlast_info(w_dir=full_wdir,datatype='dbl')
    except:
        nlinf = pypl.nlast_info(w_dir=full_wdir,datatype='flt')
    max_step = nlinf['nlast']
    return max_step

def gamma_to_beta(gamma):
    beta = (1.-(gamma**-2))**0.5
    return beta

def beta_to_gamma(beta):
    gamma = 1./((1.-beta**2)**0.5)
    return gamma


def cyl_to_cart(r,z,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z

def angle_to_boost(viewing_angle,beta):
    gamma = (1-(beta**2))**(-0.5)
    #change minus sign for approaching (-) or receding (+) jet
    delta = (gamma**(-1))*(1.0-beta*np.cos(viewing_angle))**(-1)
    return delta

def doppler_boost_lum(beta,viewing_angle,alpha,lum):
    delta = angle_to_boost(viewing_angle,beta)
    boosted_lum = lum*(delta**(2-alpha))
    return boosted_lum

def interpolate_cyl_to_cart(x_cart,y_cart,z_ordered,ll,grid_x,grid_y,grid_size,system_name,results_folder,image_timestep,load_interp=False):
    if load_interp:
        print("loading")
        interp_clean = b.load_list(results_folder,'interpolated_frame_'+system_name+'_'+str(image_timestep))
        integrated_frames = b.load_list(results_folder,'integrated_frame_'+system_name+'_'+str(image_timestep))
    else:
        interps = []
        iter_num = len(z_ordered)
        for i in range(iter_num):
            b.loader_bar(i,iter_num,5)
            x_arr = x_cart[i].flatten() #160x130 points -> 20800
            y_arr = y_cart[i].flatten()
            points = (x_arr,y_arr)
            values = ll[i].flatten()
            interp_grid = (grid_x,grid_y)
            interp = griddata(points,values,interp_grid,method='linear')
            interps.append(interp)
        interp_clean = np.nan_to_num(interps) #remove all nans and turn them into zeros for the sake of integration 
        b.save_list(np.asarray(interp_clean),results_folder,'interpolated_frame_'+system_name+'_'+str(image_timestep))
        len(interp_clean)
        integrated_frames = []
        pixel_vol = (grid_size)**3 #multiply by the volume of each pixel when integrating
        for frame in interp_clean:
            integrated_frame = np.sum(frame,axis=0)*pixel_vol*2
            integrated_frames.append(integrated_frame)
        b.save_list(np.asarray(integrated_frames),results_folder,'integrated_frame_'+system_name+'_'+str(image_timestep))
    return interp_clean, integrated_frames

def m_to_arcseconds(m,distance_in_pc):
    acs = (m*100)/(1.496*10**(13))*(1/distance_in_pc)
    return acs

####fix magic numbers below

def theta_from_beta(beta):
    theta = np.arccos(0.4/beta)*180/np.pi
    return theta

