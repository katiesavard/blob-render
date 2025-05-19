import os
import numpy as np



def save_list(d,folder,name):
    """ saves a np array into a given location
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file = folder+'/'+name+'.npy'
    np.save(file,d)

def load_list(folder, name):
    """loads a numpy array from a given location
    """
    file = folder+'/'+name+'.npy'
    arr = np.load(file,allow_pickle=True)
    return arr

