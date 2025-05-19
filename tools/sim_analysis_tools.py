import numpy as np


def load_data_obj(wdir_in_PLUTO, num=-1, data_type='dbl'):
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
    plutodir = os.environ['PLUTO_DIR']
    wdir = plutodir+wdir_in_PLUTO
    if num==-1:
        nlinf = pypl.nlast_info(w_dir=wdir)
        print("data type: "+data_type)
        D = pp.pload(nlinf['nlast'],w_dir=wdir,datatype=data_type) # Loading the data into a pload object D.
    else: 
        print("data type: "+data_type)
        D = pp.pload(num,w_dir=wdir,datatype=data_type) # Loading the data into a pload object D.
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


####fix magic numbers below

def theta_from_beta(beta):
    theta = np.arccos(0.4/beta)*180/np.pi
    return theta

def m_to_arcseconds(m,distance_in_pc):
    acs = (m*100)/(1.496*10**(13))*(1/distance_in_pc)
    return acs