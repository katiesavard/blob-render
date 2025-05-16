import sys
sys.path.append("/Users/savard/PLUTO/pluto_playtime/plotting_analysis/")  
sys.path.append("/Users/savard/PLUTO/")
import pyPLUTO as pypl
import pyPLUTO.pload as pp
from pyplutplot import *
wdir = '/pluto_playtime/data_storage/' #set up working directory where data is stored
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata
from pluto_luminosity_conversion import *
import multiprocessing as mp
from numba import jit
import numba as nb

def tstep_to_days(t,dbl_step):
    """converts simulation steps into days.
    assumes a fixed simulation time unit writing a step every multiple of dbl_step
    """
    T_SIM_YEARS = 3.173e-02
    d = t*T_SIM_YEARS*365*dbl_step
    return d

def days_to_tstep(days, dbl_step):
    T_SIM_YEARS = 3.173e-02
    t = days/(T_SIM_YEARS*365*dbl_step)
    return t

def gamma_to_beta(gamma):
    beta = (1.-(gamma**-2))**0.5
    return beta

def beta_to_gamma(beta):
    gamma = 1./((1.-beta**2)**0.5)
    return gamma

def theta_from_beta(beta):
    theta = np.arccos(0.4/beta)*180/np.pi
    return theta

@jit
def cyl_to_cart(r,z,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z

@jit
def angle_to_boost(viewing_angle,beta):
    gamma = (1-(beta**2))**(-0.5)
    #change minus sign for approaching (-) or receding (+) jet
    delta = (gamma**(-1))*(1.0-beta*np.cos(viewing_angle))**(-1)
    return delta

@jit
def doppler_boost_lum(beta,viewing_angle,alpha,lum):
    delta = angle_to_boost(viewing_angle,beta)
    boosted_lum = lum*(delta**(2-alpha))
    return boosted_lum

@jit
def los_angle(theta,viewing_angle,vx1,vx2,beta):
    denom = (vx1*np.sin(theta)*np.cos(-viewing_angle))-(vx2*np.sin(-viewing_angle))
    phi = np.arccos(denom/beta)
    return phi

@jit
def ordered_cylindrical_grid(Dvx1,Dvx2,z,r,values,thetas,viewing_angle,alpha):
    # creating long 2D arrays which correspond to each point in the frame -> ordered by constant z 
    theta_len = thetas.size
    height = z.size
    r_len = r.size
    width = r_len*theta_len
    

    rr = np.zeros((height,width))
    zz = np.zeros((height,width))
    tt = np.zeros((height,width))
    ll = np.zeros((height,width))
    beta_abs = np.sqrt(Dvx2**2 + Dvx1**2)
    for i in range(height):
        for j in range(r_len):
            #constant r iteration with all theta
            arr_z = np.repeat(z[i],theta_len)
            arr_r = np.repeat(r[j],theta_len)
            #doppler boost the emissivity with rotation of 30 deg and theta dependence
            phi_dopps = los_angle(thetas,viewing_angle,Dvx1[j][i],Dvx2[j][i],beta_abs[j][i])
            boosted_ems = doppler_boost_lum(beta_abs[j][i],phi_dopps,alpha,values[j][i])
            start_index = j*theta_len
            ll[i][start_index:start_index+theta_len] = boosted_ems
            rr[i][start_index:start_index+theta_len] = arr_r
            zz[i][start_index:start_index+theta_len] = arr_z
            tt[i][start_index:start_index+theta_len] = thetas
    
    return rr,zz,tt,ll

@jit
def rotate_cartesian(x,y,z,theta):
    yp = (y*np.cos(theta))-(np.array(z)*np.sin(theta))
    zp = (y*np.sin(theta))+(np.array(z)*np.cos(theta))
    xp = x
    return xp, yp, zp

def dontrotate(x,y,z):
    return x,y,z

@jit
def new_cartesian_grid(x_new,y_new, z_new,r):
    #### make sure to pass numpy arrays, not lists!

    #create new grid 2D to interpolate on
    cell_size = r[1]-r[0] #used to be grid size
    xlen = int((np.max(x_new)-np.min(x_new))/cell_size)
    ylen = int((np.max(y_new)-np.min(y_new))/cell_size)
    zlen = int((np.max(z_new)-np.min(z_new))/cell_size)
    #the spacing of the goal grid is the same as the original 2D cylindrical grid
    x_ar = np.linspace(np.min(x_new),np.max(x_new),xlen)
    y_ar = np.linspace(np.min(y_new),np.max(y_new),ylen)
    z_ar = np.linspace(np.min(z_new),np.max(z_new),zlen)
    return x_ar, y_ar, z_ar

def m_to_arcseconds(m,distance_in_pc):
    acs = (m*100)/(1.496*10**(13))*(1/distance_in_pc)
    return acs

def loader_bar(i,range,modulo): #just for output purposes
    perc = int((i/range)*100)
    prev_perc = int(((i-1)/range)*100)
    if i==0:
        s = str(0)+"%"
        print(s,end="...",flush=True)
    if perc%modulo==0 and perc!=prev_perc:
        s = str(perc)+"%"
        print(s,end="...",flush=True)
        prev=perc+modulo

def interpolate_onerow_flat(args):
    x_flat,y_flat,ll_flat,grid_x,grid_y = args
    points = (x_flat,y_flat)
    interp_grid = (grid_x,grid_y)
    interp = griddata(points,ll_flat,interp_grid,method='linear')
    return interp

def interpolate_flat_MPI(x_irr,y_irr,z_irr,ll,grid_x,grid_y,num_pools,fig_dir,sim_name,image_timestep):
    items = range(len(z_irr))
    array = z_irr
    #need to make grid_x and grid_y accessible everywhere somehow 
    with mp.Pool(num_pools) as pool:
        iter_num = range(len(z_irr))
        args = [(x_irr[i].flatten(),y_irr[i].flatten(),ll[i].flatten(),grid_x,grid_y) for i in iter_num]
        result = pool.map(interpolate_onerow_flat,args)
    interp_clean = np.nan_to_num(result)
    save_list(np.asarray(interp_clean),fig_dir,'interpolated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')
    return interp_clean

def integrate_flat(cell_size,interp_clean,fig_dir,sim_name,image_timestep):
    integrated_frames = []
    pixel_vol = (cell_size)**3 #multiply by the volume of each pixel when integrating
    for frame in interp_clean:
        integrated_frame = np.sum(frame,axis=0)*pixel_vol*2
        integrated_frames.append(integrated_frame)
    save_list(np.asarray(integrated_frames),fig_dir,'integrated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')


def interpolate_total(x_ar,y_ar,z_ar,ll,grid_x,grid_y,grid_size,system_name,results_folder,image_timestep,save='save'):
    if save=='save':
        grid_x, grid_y = np.meshgrid(x_ar,y_ar,z_ar,indexing='ij')
        interps = []
        iter_num = len(z_ar)
        for i in range(iter_num):
            loader_bar(i,iter_num,5)
            x_arr = x_ar[i].flatten() #160x130 points -> 20800
            y_arr = y_ar[i].flatten()
            points = (x_arr,y_arr)
            values = ll[i].flatten()
            interp_grid = (grid_x,grid_y)
            interp = griddata(points,values,interp_grid,method='linear')
            interps.append(interp)
        interp_clean = np.nan_to_num(interps) #remove all nans and turn them into zeros for the sake of integration 
        save_list(np.asarray(interp_clean),results_folder,'interpolated_frame_'+system_name+'_'+str(image_timestep))
        len(interp_clean)
        integrated_frames = []
        pixel_vol = (grid_size)**3 #multiply by the volume of each pixel when integrating
        for frame in interp_clean:
            integrated_frame = np.sum(frame,axis=0)*pixel_vol*2
            integrated_frames.append(integrated_frame)
        save_list(np.asarray(integrated_frames),results_folder,'integrated_frame_'+system_name+'_'+str(image_timestep))
    elif save=='load':
        print("loading")
        interp_clean = load_list(results_folder,'interpolated_frame_'+system_name+'_'+str(image_timestep))
        integrated_frames = load_list(results_folder,'integrated_frame_'+system_name+'_'+str(image_timestep))
    return interp_clean, integrated_frames

def interpolate_flat(x_ar,y_ar,x_new,y_new,z_new,ll,cell_size,fig_dir,sim_name,image_timestep,save):
    if save=='save':
        grid_x, grid_y = np.meshgrid(x_ar,y_ar)
        interps = []
        iter_num = len(z_new)
        for i in range(iter_num):
            loader_bar(i,iter_num,5)
            x_arr = x_new[i].flatten() #160x130 points -> 20800
            y_arr = y_new[i].flatten()
            points = (x_arr,y_arr)
            values = np.array(ll)[i].flatten()
            interp_grid = (grid_x,grid_y)
            interp = griddata(points,values,interp_grid,method='linear')
            interps.append(interp)
        
        interp_clean = np.nan_to_num(interps) #remove all nans and turn them into zeros for the sake of integration 
        save_list(np.asarray(interp_clean),fig_dir,'interpolated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')
        
        integrated_frames = []
        pixel_vol = (cell_size)**3 #multiply by the volume of each pixel when integrating
        for frame in interp_clean:
            integrated_frame = np.sum(frame,axis=0)*pixel_vol*2
            integrated_frames.append(integrated_frame)
        save_list(np.asarray(integrated_frames),fig_dir,'integrated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')
    elif save=='load':
        print("loading")
        interp_clean = load_list(fig_dir,'interpolated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')
        integrated_frames = load_list(fig_dir,'integrated_frame_'+sim_name+'_'+str(image_timestep)+'_flat')
    return interp_clean, integrated_frame

def plot_basic(a1,a2,data,figname,cbartitle,title,results_folder):
    fig = plt.figure(figsize=[8,8])
    plt.pcolormesh(a1,a2,data.T,shading='auto')
    plt.pcolormesh(-a1,a2,data.T,shading='auto')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar()
    plt.title(title)
    cbar.set_label(cbartitle)
    save_fig(results_folder,figname,overwrite=True)
    plt.close()
    del fig, ax, cbar

def main():

    ###########   input arguments    ###############

    """
    You can input arguments on the command line OR just default to the ones listed here
    """

    args = sys.argv[1:]


    default_sim_name = 'C5' #where the PLUTO output data is stored (name of the folder)
    default_image_timestep = 100 #which timestep
    sampling = 4

    if len(args)==2:
        sim_name = args[0] #'ri0.001_rb0.02_k0.1_lz25_6hhres_newprs2'
        image_timestep = args[1]
    else:
        default = input('need 2 args: use default arguments? '+'\n'+'default arguments are: '+'\n'+default_sim_name+'\n'+'timestep '+str(default_image_timestep)+'\n'+'(y/n)')
        if default=='y':
            sim_name = default_sim_name
            image_timestep = default_image_timestep
        else:
            return

    save_or_load = input('save or load interpolation?')
    if save_or_load!= 'save' and save_or_load!='load':
        print('bad input: save or load!')
        return
    
    ###########   definitions    ###############

    kappa = 0.1 #fraction of pressure contribution to luminosity
    distance_in_pc = 9400
    alpha= -0.507 #spectral index
    P_sim = 1.503*10**(-4) #kg m^-1 s^2  (PLUTO units)
    L_sim = 1*10**13 #m (PLUTO units)
    nu_observe = (1.28* 1e9) #1.28 GHz (Bright2020) observing frequency
    eta=0.75 #equipartition
    exponent = (3-alpha)/2
    p = (4*exponent) - 5 #2.01
    dtype = 'flt' #what data type PLUTO outputs

    ###########    setups    ###############
    data_dir = '/pluto_playtime/data_storage/'+sim_name+'/' #set up working directory where data is stored
    fig_dir = '/Users/Shared/Data/savard/PLUTO/pluto_playtime/analysis_cirX1/'+sim_name+'/'
    angle_degrees = 30.0
    viewing_angle = (2*np.pi)/360 *angle_degrees

    ###################  plot original 2D  #######################################


    D = load_data_obj(data_dir,image_timestep,data_type=dtype)
    
    rotate = False

    em = ((kappa*D.prs)**exponent).T[::sampling,::sampling].T
    ax1 = D.x1[::sampling]
    ax2 = D.x2[::sampling]

    crop_name = 'simimage_original_'+sim_name+'_'+str(image_timestep)
    crop_cbartitle='pseudo-emissivity sim units'
    plot_basic(ax1,ax2,em,crop_name,crop_cbartitle,sim_name,fig_dir) #save the cropped image original



    ############      make a cylindrical grid and boost        #########################

    r = ax1 #r, 160 vals
    z = ax2 #z, 1600 vals
    values = em

    NTHETA = 50

    thetas = np.linspace(0,np.pi/2,NTHETA) #create some theta to wrap it around pi/2, arbitrary number of steps for now
    theta_len = len(thetas)
    z_len = len(z)
    r_len = len(r)
    cell_size = r[1]-r[0] #used to be grid size
    rr, zz, tt, ll = ordered_cylindrical_grid(D,z,r,values,thetas,viewing_angle,alpha)

    #convert to cartesian:
    x_cart, y_cart, z_cart = cyl_to_cart(rr, zz, tt)

    # rotate the cartsian grid: 
    if rotate:
        x_new, y_new , z_new = rotate_cartesian(x_cart,y_cart,z_cart,-viewing_angle)
    else:
        x_new, y_new , z_new = dontrotate(x_cart , y_cart, z_cart)

    #create new grid to interpolate on
    #  (the spacing of the goal grid is the same as the original 2D cylindrical grid)
    x_ar, y_ar, z_ar = new_cartesian_grid(x_new,y_new,z_new,r)
    
    #turn this into meshgrid:
    print('making meshgrid')
    if rotate:
        grid_x, grid_y, grid_z = np.meshgrid(x_ar,y_ar,z_ar)
    

    ############      interpolate        ###########################################

    print('beginning interpolation')
    interp_clean, integrated_frames = interpolate_flat(x_ar,y_ar,x_new,y_new,z_new,ll,cell_size,fig_dir,sim_name,image_timestep,save=save_or_load)

main()