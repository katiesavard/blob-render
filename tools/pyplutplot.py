#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:25:57 2022

@author: savard
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyPLUTO as pypl
import pyPLUTO.pload as pp
#import pyPLUTO as pypl
from matplotlib import animation
import matplotlib
import gc
import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import matplotlib.colors as colors
#matplotlib.use('Agg') # for avoiding memory leak




def set_plot_defaults():
    """
    assign default plot settings
    """
    ## FIGURE
    #plt.rcParams["text.usetex"] = "True"

    ## FONTplut 
    #plt.rcParams['font.serif']=['cm']
    #plt.rcParams['font.family']='serif'
    # plt.rcParams['font.serif']=['cm']
    #plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

    plt.rcParams['font.size']=18
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams['ytick.labelsize']=15
    plt.rcParams['legend.fontsize']=14
    plt.rcParams['axes.titlesize']=16
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['axes.linewidth']=2
    plt.rcParams["lines.linewidth"] = 2.2
    ## TICKS
    plt.rcParams['xtick.top']='True'
    plt.rcParams['xtick.bottom']='True'
    plt.rcParams['xtick.minor.visible']='True'
    plt.rcParams['xtick.direction']='out'
    plt.rcParams['ytick.left']='True'
    plt.rcParams['ytick.right']='True'
    plt.rcParams['ytick.minor.visible']='True'
    plt.rcParams['ytick.direction']='out'
    plt.rcParams['xtick.major.width']=1.5
    plt.rcParams['xtick.minor.width']=1
    plt.rcParams['xtick.major.size']=4
    plt.rcParams['xtick.minor.size']=3
    plt.rcParams['ytick.major.width']=1.5
    plt.rcParams['ytick.minor.width']=1
    plt.rcParams['ytick.major.size']=4
    plt.rcParams['ytick.minor.size']=3


def loader_bar(i,range,modulo):
    if i%modulo==0:
            perc = int(i/range)
            s = str(perc)+"%"
            print(s, end="...")

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

def save_fig(dir_title,fig_title,overwrite=False):
    """saves a figure to a given location. includes a switch to 
    check if you want to overwrite the figure if it already exists. 
    """
    if not os.path.isdir(dir_title):
        os.mkdir(dir_title)
    if os.path.exists(dir_title+"/"+fig_title+".jpg"):
        if overwrite==True:
            print("Overwriting existing figure!!!")
            plt.savefig(dir_title+"/"+fig_title+'.jpg',bbox_inches='tight',dpi=400)
        else:
            print("Not overwriting existing figure!!!!")
    else:
        print("saving new figure!!")
        plt.savefig(dir_title+"/"+fig_title+'.jpg',bbox_inches='tight',dpi=400)

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

def plot_cmesh(ax1,ax2,var,title,figsize,vmin=0,vmax=0):
    """
    Plotting function for 2D colorplot

    Parameters
    ----------
    ax1 : 1D array
        x-axis
    ax2 : 1D array
        y-axis
    var : 2D array
        variable to be plotted
    title : string
        plot title
    figsize : [int,int]
        matplotlib figure dimensions

    Returns
    -------
    Quadmesh object (matplotlib object)

    """
    plt.figure(figsize=figsize)
    plt.title(title)
    if vmin==0 and vmax==0:
        plt.pcolormesh(ax1,ax2,var.T,shading='auto')
        plt.pcolormesh(-ax1,ax2,var.T,shading='auto')
        plt.tight_layout()
    else:
        plt.pcolormesh(ax1,ax2,var.T,vmin=vmin,vmax=vmax,shading='auto')
        plt.pcolormesh(-ax1,ax2,var.T,vmin=vmin,vmax=vmax,shading='auto')
        plt.tight_layout()
    plt.colorbar()
 
def png_to_gif(png_path_and_title,max_step,interval=300,modulo=1):

    """Turns a series of pngs in some location into one ordered gif
    pngs must be labelled as title_01.png etc 

    Args:
        png_path_and_title (string): path/title of pngs without extension or final number
        gif_title (string): title of gif you want to create
        max_step (int): last timestep integer
        interval (int): delay between frames in milliseconds, default 300
    """


    
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')

    ims = []
    if modulo==1:
        for i in range(max_step+1):
            im = ax.imshow(plt.imread(png_path_and_title+"_{}.png".format(i)), animated = True)
            ims.append([im])
    else:
        for i in range(0,max_step+1,modulo):
            im = ax.imshow(plt.imread(png_path_and_title+"_{}.png".format(i)), animated = True)
            ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval)
    ani.save(png_path_and_title+'.gif',dpi=400)

def plot_all_timesteps(var,exponent,new_folder_title,figsize,data_folder_name,show_plots=True,modulo=1,vmin=0,vmax=0):
    """Plots colourmaps of given variable at all time steps, and saves them in a new or 
    existing directory 

    Args:
        var (string): variable to access from pyPLUTO data object
        new_folder_title (string): name of folder to store figures in, also plot title and figure title
        figsize ([int,int]): figure size of plot
        data_folder_name (string): directory name within data_storage to find data
        show_plots (bool, optional): Decides weather to output plots to screen whiel running. Defaults to True.
    """
    pluto_wdir = '/pluto_playtime/data_storage/'+data_folder_name+'/'
    storage_dir = '../data_storage/'+data_folder_name+'/'+new_folder_title
    max_step = get_max_step(pluto_wdir)
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)
    if modulo>1:
        for i in range(0,max_step+1,modulo):
            D = load_data_obj(pluto_wdir,i)
            if var=='synch':
                new_var = D.tr1*(D.prs**exponent)
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            elif var =='lorentz':
                total_vel_sq = (D.vx1**2)+(D.vx2**2)
                new_var = 1.0/np.sqrt((1.0-total_vel_sq))
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            elif var == '4plot':
                plot_4plot(D,i,exponent)
            else:
                new_var = getattr(D,var)
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            title = storage_dir+"/"+new_folder_title+"_{}.png".format(i)
            plt.savefig(title)
            plt.close()
            #del D
            #gc.collect()
    else:
        for i in range(max_step+1):
            D = load_data_obj(pluto_wdir,i)
            if var=='synch':
                new_var = D.tr1*(D.prs**exponent)
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            elif var =='lorentz':
                total_vel_sq = (D.vx1**2)+(D.vx2**2)
                new_var = 1.0/np.sqrt((1.0-total_vel_sq))
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            elif var == '4plot':
                plot_4plot(D,i)
            else:
                new_var = getattr(D,var)
                plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)

            title = storage_dir+"/"+new_folder_title+"_{}.png".format(i)
            plt.savefig(title)
            plt.close()
            #del D 
            #gc.collect()


""" def plot_4plot(D,i,exponent):
    plt.rcParams['font.size'] = 22
    total_vel_sq = (D.vx1**2)+(D.vx2**2)
    lorentz = 1.0/np.sqrt((1.0-total_vel_sq))
    synch = D.tr1*(D.prs**exponent)
    plt.figure(figsize=(37,15))
    f, c = single_plot(D.x1,D.x2,D.rho,"Density",151,0,0,c=plt.get_cmap('afmhot'))
    c.ax.set_title(r"$cm^{-3}$")
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    f, c = single_plot(D.x1,D.x2,D.tr1,"Tracer",152,0,0,c=plt.get_cmap('plasma'))
    c.ax.set_title(r'$n$')
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    f, c = single_plot(D.x1,D.x2,synch,"Pseudo Emissivity",153,0,0,c=plt.get_cmap('viridis'))
    #c.ax.set_title(r'$cm^{-3}$')
    f, c = single_plot(D.x1,D.x2,lorentz,"Lorentz Factor",155,0,0,c=plt.get_cmap('inferno'))
    c.ax.set_title(r'$\Gamma$')
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    f, c = single_plot(D.x1,D.x2,D.prs*(1.503),"Pressure",154,0,0,c=plt.get_cmap('cividis'))
    c.ax.set_title(r"dyne ""\n"" $cm^{-2} e-03$")
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    plt.tight_layout(pad=3)
    dbl_frequency = 4
    unit_time_years = 1.05*10**(-3)
    day = i*unit_time_years*365*dbl_frequency
    plt.suptitle("{:.0f} days".format(day))

def plot_4plot_3D(D,i,exponent):
    plt.rcParams['font.size'] = 22
    total_vel_sq = (D.vx1**2)+(D.vx2**2)
    lorentz = 1.0/np.sqrt((1.0-total_vel_sq))
    synch = D.tr1*(D.prs**exponent)
    plt.figure(figsize=(36,25))
    f, c = single_plot(D.x1,D.x2,D.rho,"Density",151,0,0,c=plt.get_cmap('afmhot'))
    c.ax.set_title(r"$atom$""\n""$cm^{-3}$")
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    f, c = single_plot(D.x1,D.x2,D.tr1,"Tracer",152,0,0,c=plt.get_cmap('plasma'))
    c.ax.set_title(r'$n$')
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    f, c = single_plot(D.x1,D.x2,synch,"Pseudo Emissivity",153,0,0,c=plt.get_cmap('viridis'))
    #c.ax.set_title(r'$cm^{-3}$')
    f, c = single_plot(D.x1,D.x2,lorentz,"Lorentz Factor",155,0,0,c=plt.get_cmap('inferno'))
    c.ax.set_title(r'$\Gamma$')
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    f, c = single_plot(D.x1,D.x2,D.prs/(1.503e-03),"Pressure",154,0,0,c=plt.get_cmap('cividis'))
    c.ax.set_title(r"dyne ""\n"" $cm^{-2}$")
    c.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    plt.tight_layout(pad=3)
    dbl_frequency = 10
    unit_time_years = 1.05*10**(-3)
    day = i*unit_time_years*365*dbl_frequency
    plt.suptitle("{:.0f} days".format(day))
    
def single_plot(ax1,ax2,var,title,id,vmin=0,vmax=0,c=plt.get_cmap('viridis')):
    f = plt.subplot(id)
    plt.pcolormesh(ax1,ax2,var.T,shading='auto',cmap=c)
    plt.pcolormesh(-ax1,ax2,var.T,shading='auto',cmap=c)
    plt.title(title,fontsize=30)
    f.get_yaxis().set_visible(False)
    f.get_xaxis().set_visible(False)
    c = plt.colorbar()
    return f, c"""

"""def plot_some_timesteps(var,exponent,new_folder_title,figsize,data_folder_name,show_plots=False,vmin=0,vmax=0,num_min=0,num_max=0):
    """"""Plots colourmaps of given variable at all time steps, and saves them in a new or 
    existing directory 
    Args:
        var (string): variable to access from pyPLUTO data object
        new_folder_title (string): name of folder to store figures in, also plot title and figure title
        figsize ([int,int]): figure size of plot
        data_folder_name (string): directory name within data_storage to find data
        show_plots (bool, optional): Decides weather to output plots to screen whiel running. Defaults to True.
    """"""
    pluto_wdir = '/pluto_playtime/data_storage/'+data_folder_name+'/'
    storage_dir = '../data_storage/'+data_folder_name+'/'+new_folder_title
    max_step = get_max_step(pluto_wdir)
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    if num_max==0:
        num_max = max_step
    elif num_max>max_step:
        num_max = max_step

    for i in range(num_min,num_max+1):
        D = load_data_obj(pluto_wdir,i)
        if var == 'synch':
            new_var = D.tr1*(D.prs**exponent)
            plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)
        elif var =='lorentz':
            total_vel_sq = (D.vx1**2)+(D.vx2**2)
            new_var = 1.0/np.sqrt((1.0-total_vel_sq))
            plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)
        elif var == '4plot':
                plot_4plot(D,i,exponent)
        else:
            new_var = getattr(D,var)
            plot_cmesh(D.x1,D.x2,new_var,new_folder_title+": timestep {}".format(i),figsize,vmin,vmax)
        title = storage_dir+"/"+new_folder_title+"_{}.png".format(i)
        plt.savefig(title)
        plt.close()"""

def double_plot(a,ax1,ax2,var1,var2,title,xzoom=1,yzoom=1,m1=plt.get_cmap('viridis'),m2=plt.get_cmap('viridis'),showtitle=False,min=False,gamma=False):
    #f = plt.subplot(id)
    offset=100
    if xzoom!=1:
        xlim = int(np.max(ax1)*xzoom)
        a.set_xlim([-xlim,xlim])
    if yzoom!=1:
        ylim = int(np.max(ax2)*yzoom)
        a.set_ylim([0+offset,ylim+offset])
    if min:
        p1 = a.pcolormesh(-ax1,ax2,var1.T,shading='auto',cmap=m1)
        p2 = a.pcolormesh(ax1,ax2,var2.T,shading='auto',vmin=-13,cmap=m2)
    elif gamma:
        p1 = a.pcolormesh(-ax1,ax2,var1.T,shading='auto',cmap=m1)
        p2 = a.pcolormesh(ax1,ax2,var2.T,shading='auto',vmin=1.0,vmax=2.1,cmap=m2)
    else:
        p1 = a.pcolormesh(-ax1,ax2,var1.T,shading='auto',cmap=m1)
        p2 = a.pcolormesh(ax1,ax2,var2.T,shading='auto',cmap=m2)
    if showtitle:
        a.set_title(title,fontsize=35)
    a.get_yaxis().set_visible(False)
    a.get_xaxis().set_visible(False)
    

    divider = make_axes_locatable(a)
    cax2 = divider.append_axes("right", size="10%", pad=0.3)
    cax1 = divider.append_axes("left", size="10%", pad=0.3)
    c2 = plt.colorbar(p2,cax=cax2)
    c1 = plt.colorbar(p1,cax=cax1,ticklocation='left')
    
    return a, c1, c2


def single_plot(a, ax1,ax2,var1,title,xzoom=1,yzoom=1,c=plt.get_cmap('viridis'),radio=False,showtitle=False):
    #f = plt.subplot(id)
    offset=100
    if xzoom!=1:
        xlim = int(np.max(ax1)*xzoom)
        a.set_xlim([-xlim,xlim])
    if yzoom!=1:
        ylim = int(np.max(ax2)*yzoom)
        a.set_ylim([0+offset,ylim+offset])
    if radio:
        p1 = a.pcolormesh(ax1,ax2,var1.T,norm=colors.PowerNorm(gamma=0.3),shading='auto',cmap=c)
        p2 = a.pcolormesh(-ax1,ax2,var1.T,norm=colors.PowerNorm(gamma=0.3),shading='auto',cmap=c)
    else:
        p1 = a.pcolormesh(ax1,ax2,var1.T,shading='auto',cmap=c)
        p2 = a.pcolormesh(-ax1,ax2,var1.T,shading='auto',cmap=c)
    if showtitle:
        a.set_title(title,fontsize=35)
    a.get_yaxis().set_visible(False)
    a.get_xaxis().set_visible(False)
    divider = make_axes_locatable(a)
    cax1 = divider.append_axes("right", size="10%", pad=0.3)
    c = plt.colorbar(p1,cax=cax1)

    
    return a, c



def plot_4plot(D,i,exponent,fsize,showt,xzoom=1,yzoom=1):

    plt.rcParams["font.family"] = "Times"
    plt.rcParams['font.size']=20
    plt.rcParams["lines.linewidth"] = 5
    plt.rcParams['axes.labelsize']=10
    plt.rcParams['legend.fontsize']=20

    ## CALCULATE 1 ARCSECOND IN SIM UNITS FOR SCALEBAR 
    offset=100

    distance_in_pc = 2960
    asec = 1
    scalebar_unit = (distance_in_pc*asec*1.496*10**(13))/10**15   #/(86400*2.998*10**10)
    

    total_vel_sq = (D.vx1**2)+(D.vx2**2)
    lorentz = 1.0/np.sqrt((1.0-total_vel_sq))
    synch = D.tr1*(D.prs**exponent)
    fig, (a1, a2, a3) = plt.subplots(1,3,figsize=fsize,gridspec_kw={'width_ratios': [1, 0.85, 1]})

    densitycmap=plt.get_cmap('Purples')
    pressurecmap=LinearSegmentedColormap.from_list('',["aliceblue","lightblue","cadetblue","cadetblue","teal","teal",'darkslategrey','black'])
    tracercmap=plt.get_cmap('RdPu')
    gammacmap=LinearSegmentedColormap.from_list('',["midnightblue",'steelblue','lightblue','aliceblue','white'])


    #-----plot the density and pressure
    f, c1, c2 = double_plot(a1,D.x1,D.x2,np.log10(D.rho),np.log10(D.prs*(1.503)*(10**(-3))),"Density           Pressure",xzoom,yzoom,m1=densitycmap,m2=pressurecmap,showtitle=showt,min=True)
    
    c1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    c2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    if showt:
        c1.ax.set_title(r"log($\mathdefault{cm^{-3}}$)")
        c2.ax.set_title(r"log($\mathdefault{g\, cm^{-1}}$""\n"r"$\mathdefault{s^{-2}}$)")
        c2.ax.titlesize : 40
    
    #-----plot pseudo emissivity
    radiomap=plt.get_cmap('gist_heat')


    f, c = single_plot(a2, D.x1,D.x2,synch,"Pseudo Emissivity",xzoom,yzoom,c=radiomap,radio=True,showtitle=showt)

    fontprops = fm.FontProperties(size=30)
    scalebar = AnchoredSizeBar(a2.transData, scalebar_unit, "1\"",3,color='white',frameon=False,size_vertical=1,pad=1,sep=3,fontproperties=fontprops)

    a2.add_artist(scalebar)


    #-----plot the tracer and gamma
    f, c1, c2 = double_plot(a3,D.x1,D.x2,D.tr1,lorentz,r"     Tracer                    $\mathdefault{\Gamma}$          ",xzoom,yzoom,m1=tracercmap,m2=gammacmap,showtitle=showt,min=False,gamma=True)
    
    
    c1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    c2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    if showt:
        c1.ax.set_title(r'$\mathdefault{n}$')
        c2.ax.set_title(r'$\mathdefault{\Gamma}$')
    
    
    
    #plt.tight_layout(pad=3)
    dbl_frequency = 4
    unit_time_years = 1.05*10**(-3)
    day = i*unit_time_years*365*dbl_frequency
    a2.text((-np.max(D.x1)*xzoom)+(60*xzoom),((np.max(D.x2)+offset)*yzoom)-(10*yzoom),"{:.0f} days".format(day),color='white',fontsize=25)

    return fig


def plot_some_timesteps(system_name,exponent=1.75,show_plots=False,vmin=0,vmax=0,num_min=0,num_max=0):
    """Plots colourmaps of given variable at all time steps, and saves them in a new or 
    existing directory 
    Args:
        var (string): variable to access from pyPLUTO data object
        new_folder_title (string): name of folder to store figures in, also plot title and figure title
        figsize ([int,int]): figure size of plot
        data_folder_name (string): directory name within data_storage to find data
        show_plots (bool, optional): Decides weather to output plots to screen whiel running. Defaults to True.
    """
    data_dir= '/pluto_playtime/data_storage/'+system_name+'/'
    storage_dir = '../data_storage/'+system_name+'/figures'
    max_step = get_max_step(data_dir)
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    if num_max==0:
        num_max = max_step
    elif num_max>max_step:
        num_max = max_step

    fsize = (30,15)
    x_zoom = 0.325
    y_zoom = 0.65
    showt = True

    for i in range(num_min,num_max+1):
        D = load_data_obj(data_dir,i,data_type='flt')
        fig = plot_4plot(D,i,exponent,fsize,showt,xzoom=x_zoom,yzoom=y_zoom)
        fig_title = "multipane_gamfix_{}_{}".format(system_name,i)
        save_fig(storage_dir,fig_title,overwrite=True)
        plt.clf()
        plt.close(fig)
        del D

def plot_some_timesteps_3D(var,exponent,new_folder_title,figsize,data_folder_name,show_plots=False,vmin=0,vmax=0,num_min=0,num_max=0):
    """Plots colourmaps of given variable at all time steps, and saves them in a new or 
    existing directory 

    Args:
        var (string): variable to access from pyPLUTO data object
        new_folder_title (string): name of folder to store figures in, also plot title and figure title
        figsize ([int,int]): figure size of plot
        data_folder_name (string): directory name within data_storage to find data
        show_plots (bool, optional): Decides weather to output plots to screen whiel running. Defaults to True.
    """
    pluto_wdir = '/pluto_playtime/data_storage/'+data_folder_name+'/'
    storage_dir = '../data_storage/'+data_folder_name+'/'+new_folder_title
    max_step = get_max_step(pluto_wdir)
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    if num_max==0:
        num_max = max_step
    elif num_max>max_step:
        num_max = max_step

    for i in range(num_min,num_max+1):
        D = load_data_obj(pluto_wdir,i)
        plot_4plot_3D(D,i,exponent)
        title = storage_dir+"/"+new_folder_title+"_{}.png".format(i)
        plt.savefig(title)
        plt.close()
        #del D , title
        #gc.collect()

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
