from . import tools
import numpy as np 
import argparse
import yaml
import os
from .paths import SIM_DAT, PLOTS, CONFIGS
from blobrender.help_strings import HELP_DICT

# default_simulation.yaml is shared with fits_conversion.py; only these keys
# are actually read by this script.
RELEVANT_KEYS = [
    'system_name', 'image_timestep', 'kappa', 'theta', 'distance_in_pc',
    'alpha', 'P_sim', 'L_sim', 'yresolution', 'nu_observe', 'eta', 'dtype',
    'load_interp', 'crop_mode', 'crop_size_r', 'crop_size_z',
    'crop_buffer_size', 'crop_min_size',
]

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
    
def crop_frame(crop_mode, v, x1, x2, yres, data_dir, system_name, image_timestep,
                crop_size_r, crop_size_z, crop_buffer_size, crop_min_size):
    """
    Decide which portion of the simulation grid to process, before the
    (expensive) interpolation step below. Returns index bounds
    (stop_index_top, stop_index_bottom, stop_index_side) into x2 (z-axis,
    vertical) and x1 (r-axis, radial).

    crop_mode:
        'none'     - process the whole domain. No assumptions, but slower and
                     more memory-hungry, and the larger frame carries through
                     fits-conversion/pad-fits/predict too.
        'position' - a crop_size_z-tall window centred on a precomputed blob
                     position (disp_array_<system_name>.npy), truncated to
                     crop_size_r along the radial axis. Specific to analysis
                     pipelines that maintain such a position file; this isn't
                     a standard PLUTO output.
        'detect'   - crop_buffer_size of padding around emission detected
                     directly from the data (no position file needed),
                     falling back to crop_min_size if the detected region is
                     smaller than that. Its detection threshold is a fixed
                     assumption too, so it may not generalise to every
                     simulation's flux scale.
    """
    print(f"crop_mode: '{crop_mode}'")

    if crop_mode == 'none':
        print("Processing the full simulation domain (no cropping).")
        return len(x2), 0, len(x1)

    if crop_mode == 'position':
        print(f"Using crop_size_r={crop_size_r}, crop_size_z={crop_size_z}")
        file_disp = 'disp_array_'+str(system_name)
        init_offset = 200
        try:
            d_vals = tools.load_list(data_dir,file_disp)+init_offset
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not find {file_disp}.npy in {data_dir}. crop_mode='position' "
                "crops the frame around the blob's position at each timestep, using a "
                "precomputed per-timestep position file that is specific to this "
                f"analysis pipeline (not a standard PLUTO output). Provide {file_disp}.npy "
                "in that directory, set crop_mode: 'none' to process the full domain, or "
                "crop_mode: 'detect' to find the blob position from the data directly."
            ) from exc
        centre = d_vals[image_timestep]
        print(f"Cropping frame around position {centre} (timestep {image_timestep}, from {file_disp}.npy)")
        stop_index_bottom = int(centre)/yres - int(crop_size_z/2)
        stop_index_top = int(centre)/yres + int(crop_size_z/2)
        return int(stop_index_top), int(stop_index_bottom), int(crop_size_r)

    if crop_mode == 'detect':
        print(f"Detecting emission directly from the data, using crop_buffer_size={crop_buffer_size}, crop_min_size={crop_min_size}")
        return find_limits(v, x1, x2, crop_buffer_size, crop_min_size)

    raise ValueError(f"Unknown crop_mode {crop_mode!r}. Use 'none', 'position', or 'detect'.")

def ordered_cylindrical_grid(theta_len,z_len,r_len,thetas,z,r,values):
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
            theta_ordered.append(thetas) # for all theta
            l_ordered.append(arr_l)
        rr.append(np.asarray(r_ordered).flatten())
        zz.append(np.asarray(z_ordered).flatten())
        ll.append(np.asarray(l_ordered).flatten())
        tt.append(np.asarray(theta_ordered).flatten())
    
    return rr,zz,tt,ll

def ordered_cartesian_grid(ax1,ax2,thetas,z_cart):
    z_ordered = []
    num_z = len(ax2)
    num_xtheta = len(ax1)*len(thetas)

    for i in range(num_z):
        z_ = z_cart[(0+(num_xtheta*i)):(0+(num_xtheta*(i+1)))]
        z_ordered.append(z_)
        
    return z_ordered


def main():

    ###########   input arguments    ###############

    """
    You can input arguments on the command line OR just use default to the ones in default.framelum.yaml
    The default values are set in the YAML file, which is loaded at the beginning of the script.
    """
    # Load defaults from YAML
    yaml_file = os.path.join(CONFIGS,'default_simulation.yaml')
    args = tools.get_arguments(yaml_file,HELP_DICT,allowed_keys=RELEVANT_KEYS)

    # Unpack arguments

    system_name = args.system_name
    image_timestep = args.image_timestep
    kappa = args.kappa
    theta = args.theta
    distance_in_pc = args.distance_in_pc
    alpha = args.alpha
    P_sim = args.P_sim
    L_sim = args.L_sim
    yres = args.yresolution
    nu_observe = args.nu_observe
    eta = args.eta
    dtype = args.dtype
    load_interp = args.load_interp
    crop_mode = args.crop_mode
    crop_size_r = args.crop_size_r
    crop_size_z = args.crop_size_z
    crop_buffer_size = args.crop_buffer_size
    crop_min_size = args.crop_min_size

    #some calculated and related values
    exponent = (3-alpha)/2
    p = (4*exponent) - 5 #2.01
    viewing_angle = (2*np.pi)/360 *theta
    

    ###########    setups    ###############
    data_dir = os.path.join(SIM_DAT, system_name)
    plots_folder = os.path.join(PLOTS, system_name)
    results_folder = SIM_DAT

    ###########   load in simulation data ##################
     
   

    # this section is specific to PLUTO simulations, where the data is stored in a specific format, and is cylindrical axisymmetric
    D = tools.load_data_obj(data_dir,image_timestep,data_type=dtype)
    v = D.tr1*((kappa*D.prs)**exponent) #pseudo emissivity, 2D array
    x1 = D.x1 # x1 is the radial axis
    x2 = D.x2 # x2 is the vertical axis


    ############      crop image        #################

    """if emission isnt in the whole domain then crop before you do the analysis (to save computational time)
    """

    stop_index_top, stop_index_bottom, stop_index_side = crop_frame(
        crop_mode, v, x1, x2, yres, data_dir, system_name, image_timestep,
        crop_size_r, crop_size_z, crop_buffer_size, crop_min_size,
    )

    print('top: '+str(stop_index_top))
    print('bottom: '+str(stop_index_bottom))
    print('side: '+str(stop_index_side))
    em = (v[:stop_index_side].T[stop_index_bottom:stop_index_top]).T #psuedo emissivity, 2D array 
    ax1 = x1[:stop_index_side]
    ax2 = x2[stop_index_bottom:stop_index_top]

    crop_name = 'simimage_originalcrop_'+system_name+'_'+str(image_timestep)
    crop_cbartitle='pseudo-emissivity sim units'
    tools.plot_basic(ax1,ax2,em,crop_name,crop_cbartitle,system_name,plots_folder) #save the cropped image original

    
   ###### ######      boost emissivity (special relativity)        ###############################

    beta = (D.vx2[:stop_index_side].T[stop_index_bottom:stop_index_top]).T
    values = tools.doppler_boost_lum(beta,viewing_angle,alpha,em)


    # ---------- all below may be replaced by DART at some point --------------- #

    ############      make a cylindrical grid        #########################

    r = ax1 #r, 160 vals
    z = ax2 #z, 1600 vals
    thetas = np.linspace(0,np.pi/2,130) #create some theta to wrap it around pi/2, arbitrary number of steps for now
    theta_len = len(thetas)
    z_len = len(z)
    r_len = len(r)

    rr, zz, tt, ll = ordered_cylindrical_grid(theta_len,z_len,r_len,thetas,z,r,values)

    #convert to cartesian:
    x_cart, y_cart, z_cart = tools.cyl_to_cart(rr, zz, tt)

    #order cartesian grid:
    z_ordered = ordered_cartesian_grid(ax1,ax2,thetas,z_cart)

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
    interp_clean, integrated_frames = tools.interpolate_cyl_to_cart(x_cart,y_cart,z_ordered,ll,grid_x,grid_y,grid_size,system_name,results_folder,image_timestep,load_interp)

    #plot a horizontal slice to check interpolation interpolation#
    populated_slices = []
    for index, arr in enumerate(interp_clean):
        # get array of indices with detection in it 
        if np.any(arr):
            populated_slices.append(index)

    test_slice_index = populated_slices[int(len(populated_slices)/2)] #get a test slice about halfway through detection
    slice_name = 'simimage_interpslice_'+str(test_slice_index)+'_'+system_name+'_'+str(image_timestep)
    slice_cbartitle='pseudo emissivity sim units'
    tools.plot_basic(grid_x,grid_y,interp_clean[test_slice_index],slice_name,slice_cbartitle,system_name,plots_folder)

    #plot a rotated slice 
    rotate_name = 'simimage_interpintegrated_'+system_name+'_'+str(image_timestep)
    rotate_cbartitle = 'pseudo emissivity sim units'
    tools.plot_basic(ax1,ax2,np.asarray(integrated_frames).T,rotate_name,rotate_cbartitle,system_name,plots_folder)

    ############      convert to janskys        #######################################

    lum_real_unit = tools.lum_unit_si(eta,P=P_sim,L=L_sim,nu=nu_observe, q=p) #W Hz^-1 sr^-1
    lum_sim = np.asarray(integrated_frames)
    lum_true = lum_sim*lum_real_unit # W Hz^-1 sr^-1
    lum_jansky = lum_true*(10**26)*np.pi*4 # 10^26 W Hz^-1  -> 10^-26 W m^-2 Hz^-1 = 1 jansky 
    distance = distance_in_pc*3.086*(10**16) # convert parsec to m 

    boosted_lum = np.asarray(lum_jansky / (4*np.pi*(distance**2)),dtype='float64')
    tools.save_list(boosted_lum,results_folder,'pixel_lum_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/(1e9))+str('GHz'))

    frame_flux = np.sum((boosted_lum*10**3))*2 
    print('total luminosity in frame in mJy: '+str(frame_flux))

    # plot image in jaskys and arcseconds 

    x = tools.m_to_arcseconds(ax1*L_sim,distance_in_pc)
    y = tools.m_to_arcseconds(ax2*L_sim,distance_in_pc)
    radio_name = 'radio_frame_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/(1e9))+str('GHz')
    tools.plot_radio(x,y,(boosted_lum*10**3).T,radio_name,'mJy per pixel',system_name,plots_folder)


if __name__ == "__main__":
    main()