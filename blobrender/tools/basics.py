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

        

