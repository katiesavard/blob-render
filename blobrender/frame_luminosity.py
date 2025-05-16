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

def tstep_to_days(t,dbl_step):
    """converts simulation steps into days.
    assumes a fixed simulation time unit writing a step every multiple of dbl_step
    """
    T_SIM_YEARS = 1.05*10**(-3)
    d = t*T_SIM_YEARS*365*dbl_step
    return d

def days_to_tstep(days, dbl_step):
    T_SIM_YEARS = 1.05*10**(-3)
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

def cyl_to_cart(r,z,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z

def find_limits(v,x1,x2,buffer_size,minimum_size):
    ##find top limit

    is_zero = True #want this to turn false before true again 
    stop_index_top = 0
    for index, xrow in enumerate(v.T):
        if np.any(xrow):
            is_zero = False
        if not np.any(xrow) and not is_zero:
            stop_index_top = index
            break


    ### find bottom limit

    detection_limit = 1e-10

    is_zero = True #want this to turn false before true again 
    stop_index_bottom = 0
    for index, xrow in enumerate(v.T):
        if np.any(xrow):
            if np.max(xrow)>detection_limit:
                is_zero = False
                stop_index_bottom = index
                break

    #### find side limit

    detection_limit = 1e-10

    is_zero = True #want this to turn false before true again 
    stop_index_side = 0
    for index, yrow in reversed(list(enumerate(v))):
        if np.any(yrow):
            if np.max(yrow)>detection_limit:
                is_zero = False
                stop_index_side = index
                break

    buffer_size = 100
    minimum_size = 1300

    stop_index_top = stop_index_top + buffer_size
    stop_index_bottom = stop_index_bottom - buffer_size
    stop_index_side = stop_index_side + buffer_size


    vdiff = stop_index_top-stop_index_bottom
    if vdiff<minimum_size:
        print('hit minimum vertical size: resorting to default size')
        stop_index_bottom = stop_index_bottom+(int(vdiff/2))-(int(minimum_size/2))
        stop_index_top = stop_index_top-(int(vdiff/2))+(int(minimum_size/2))

    if stop_index_side<int(minimum_size/2):
        print("hit minimum lateral size: resorting to default size")
        stop_index_side = int(minimum_size/2)


    return stop_index_top, stop_index_bottom, stop_index_side
    
def find_limits_com(image_timestep,d_vals):
    stop_index_side = 650
    minimum_size = 1300
    stop_index_bottom = (int(d_vals[image_timestep])*6-int(minimum_size/2))
    stop_index_top = int(d_vals[image_timestep])*6+int(minimum_size/2)
    stop_index_bottom
    return stop_index_top, stop_index_bottom, stop_index_side


def angle_to_boost(viewing_angle,beta):
    gamma = (1-(beta**2))**(-0.5)
    #change minus sign for approaching (-) or receding (+) jet
    delta = (gamma**(-1))*(1.0-beta*np.cos(viewing_angle))**(-1)
    return delta

def doppler_boost_lum(beta,viewing_angle,alpha,lum):
    delta = angle_to_boost(viewing_angle,beta)
    boosted_lum = lum*(delta**(2-alpha))
    return boosted_lum

def cyl_to_cart(r,z,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z

def ordered_cylindrical_grid(theta_len,z_len,r_len,theta,z,r,values):
    # creating long 2D arrays which correspond to each point in the frame -> ordered by constant z 
    rr = []
    zz = []
    tt = []
    ll = []

    for i in range(z_len):
        #constant z iteration
        z_ordered = []
        r_ordered = []
        l_ordered = []
        theta_ordered = []
        for j in range(r_len):
            #constant r iteration with all theta
            arr_z = np.repeat(z[i],theta_len).tolist()
            arr_r = np.repeat(r[j],theta_len).tolist()
            arr_l = np.repeat(values[j][i],theta_len).tolist()
            r_ordered.append(arr_r) # r at i (const in j loop)
            z_ordered.append(arr_z) # z at j (iterating through const r)
            theta_ordered.append(theta) # for all theta
            l_ordered.append(arr_l)
        rr.append(np.asarray(r_ordered).flatten())
        zz.append(np.asarray(z_ordered).flatten())
        ll.append(np.asarray(l_ordered).flatten())
        tt.append(np.asarray(theta_ordered).flatten())
    
    return rr,zz,tt,ll

def ordered_cartesian_grid(ax1,ax2,theta,z_cart):
    z_ordered = []
    num_z = len(ax2)
    num_xtheta = len(ax1)*len(theta)

    for i in range(num_z):
        z_ = z_cart[(0+(num_xtheta*i)):(0+(num_xtheta*(i+1)))]
        z_ordered.append(z_)
        
    return z_ordered

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
        prev_perc=perc+modulo

def interpolate_cyl_to_cart(x_cart,y_cart,z_ordered,ll,grid_x,grid_y,grid_size,system_name,results_folder,image_timestep,save='save'):
    if save=='save':
        interps = []
        iter_num = len(z_ordered)
        for i in range(iter_num):
            loader_bar(i,iter_num,5)
            x_arr = x_cart[i].flatten() #160x130 points -> 20800
            y_arr = y_cart[i].flatten()
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

def plot_radio(a1,a2,data,figname,cbartitle,title,results_folder):
    fig = plt.figure(figsize=[8,8])
    plt.pcolormesh(a1,a2,data.T,cmap='afmhot',shading='auto')
    plt.pcolormesh(-a1,a2,data.T,cmap='afmhot',shading='auto')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar()
    plt.title(title)
    plt.xlabel('arcseconds')
    plt.ylabel('arcseconds')
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

    default_system_name = 'ri0.001_rb0.02_lz2' #where the PLUTO output data is stored (name of the folder)
    default_image_timestep = 150 #which timestep


    if len(args)==2:
        system_name = args[0] #'ri0.001_rb0.02_k0.1_lz25_6hhres_newprs2'
        image_timestep = args[1]
    else:
        default = input('need 2 args: use default arguments? '+'\n'+'default arguments are: '+'\n'+default_system_name+'\n'+'timestep '+str(default_image_timestep)+'\n'+'(y/n)')
        if default=='y':
            system_name = default_system_name
            image_timestep = default_image_timestep
        else:
            return

    save_or_load = input('save or load interpolation?')
    if save_or_load!= 'save' and save_or_load!='load':
        print('bad input: save or load!')
        return
    
    ###########   definitions    ###############

    kappa = 0.1 #fraction of pressure contribution to luminosity
    gamma = 2.0 #lorentz factor
    distance_in_pc = 2960
    alpha= -0.507 #spectral index
    P_sim = 1.503*10**(-4) #kg m^-1 s^2  (PLUTO units)
    L_sim = 1*10**13 #m (PLUTO units)
    nu_observe = (1.28* 1e9) #1.28 GHz (Bright2020) observing frequency
    eta=0.75 #equipartition
    exponent = (3-alpha)/2
    p = (4*exponent) - 5 #2.01
    dtype = 'flt' #what data type PLUTO outputs

    ###########    setups    ###############
    data_dir = '/pluto_playtime/data_storage/'+system_name+'/' #directory where data is stored
    results_folder = '/Users/savard/PLUTO/pluto_playtime/plotting_analysis/sim_results/{}'.format(system_name) #where you want the results to go
    angle_degrees = theta_from_beta(gamma_to_beta(gamma))
    viewing_angle = (2*np.pi)/360 *angle_degrees
    file_disp = 'disp_array'+str(system_name)
    init_offset = 200
        
    ###########################################################################

    d_vals = load_list(results_folder,file_disp)+init_offset

    D = load_data_obj(data_dir,image_timestep,data_type=dtype)
    

    v = D.tr1*((kappa*D.prs)**exponent)
    x1 = D.x1
    x2 = D.x2


    ############      crop image        #####################################

    """if emission isnt in the whole domain then crop before you do the analysis (to save computational time)
    """

    #buffer_size = 100
    #minimum_size = 1300
    #stop_index_top, stop_index_bottom, stop_index_side = find_limits(v,x1,x2,buffer_size,minimum_size)
    stop_index_top, stop_index_bottom, stop_index_side = find_limits_com(image_timestep,d_vals)

    em = (v[:stop_index_side].T[stop_index_bottom:stop_index_top]).T #psuedo emissivity, 2D array 
    ax1 = x1[:stop_index_side]
    ax2 = x2[stop_index_bottom:stop_index_top]

    crop_name = 'simimage_originalcrop_'+system_name+'_'+str(image_timestep)
    crop_cbartitle='pseudo-emissivity sim units'
    plot_basic(ax1,ax2,em,crop_name,crop_cbartitle,system_name,results_folder) #save the cropped image original

    
   ###### ######      boost emissivity        ###############################

    beta = (D.vx2[:stop_index_side].T[stop_index_bottom:stop_index_top]).T
    values = doppler_boost_lum(beta,viewing_angle,alpha,em)


    ############      make a cylindrical grid        #########################

    r = ax1 #r, 160 vals
    z = ax2 #z, 1600 vals
    theta = np.linspace(0,np.pi/2,130) #create some theta to wrap it around pi/2, arbitrary number of steps for now
    theta_len = len(theta)
    z_len = len(z)
    r_len = len(r)

    rr, zz, tt, ll = ordered_cylindrical_grid(theta_len,z_len,r_len,theta,z,r,values)

    #convert to cartesian:
    x_cart, y_cart, z_cart = cyl_to_cart(rr, zz, tt)

    #order cartesian grid:
    z_ordered = ordered_cartesian_grid(ax1,ax2,theta,z_cart)

    #create new grid 2D to interpolate on
    x_max = y_max = np.max(r)
    grid_size = r[1]-r[0]
    r_len = len(r)
    #the spacing of the goal grid is the same as the original 2D cylindrical grid
    x_ar = np.linspace(-(grid_size/2),x_max,r_len+1)[1:]
    y_ar = np.linspace(-(grid_size/2),y_max,r_len+1)[1:]
    grid_x, grid_y = np.meshgrid(x_ar,y_ar)

    ############      interpolate        ###########################################

    print('beginning interpolation')
    interp_clean, integrated_frames = interpolate_cyl_to_cart(x_cart,y_cart,z_ordered,ll,grid_x,grid_y,grid_size,system_name,results_folder,image_timestep,save=save_or_load)

    #plot a horizontal slice to check interpolation interpolation#
    populated_slices = []
    for index, arr in enumerate(interp_clean):
        # get array of indices with detection in it 
        if np.any(arr):
            populated_slices.append(index)

    test_slice_index = populated_slices[int(len(populated_slices)/2)] #get a test slice about halfway through detection
    slice_name = 'simimage_interpslice_'+str(test_slice_index)+'_'+system_name+'_'+str(image_timestep)
    slice_cbartitle='pseudo emissivity sim units'
    plot_basic(grid_x,grid_y,interp_clean[test_slice_index],slice_name,slice_cbartitle,system_name,results_folder)

    #plot a rotated slice 
    rotate_name = 'simimage_interpintegrated_'+system_name+'_'+str(image_timestep)
    rotate_cbartitle = 'pseudo emissivity sim units'
    plot_basic(ax1,ax2,np.asarray(integrated_frames).T,rotate_name,rotate_cbartitle,system_name,results_folder)

    ############      convert to janskys        #######################################
    lum_real_unit = lum_unit_si(eta,P=P_sim,L=L_sim,nu=nu_observe, q=p) #W Hz^-1 sr^-1
    lum_sim = np.asarray(integrated_frames)
    lum_true = lum_sim*lum_real_unit # W Hz^-1 sr^-1
    lum_jansky = lum_true*(10**26)*np.pi*4 # 10^26 W Hz^-1  -> 10^-26 W m^-2 Hz^-1 = 1 jansky 
    distance = distance_in_pc*3.086*(10**16) # convert parsec to m 

    boosted_lum = np.asarray(lum_jansky / (4*np.pi*(distance**2)),dtype='float64')
    save_list(boosted_lum,results_folder,'pixel_lum_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/(1e9))+str('GHz'))

    frame_flux = np.sum((boosted_lum*10**3))*2
    print('total luminosity in frame in mJy: '+str(frame_flux))

    # plot image in jaskys and arcseconds 

    x = m_to_arcseconds(ax1*L_sim,distance_in_pc)
    y = m_to_arcseconds(ax2*L_sim,distance_in_pc)
    radio_name = 'radio_frame_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/(1e9))+str('GHz')
    plot_radio(x,y,(boosted_lum*10**3).T,radio_name,'mJy per pixel',system_name,results_folder)


main()