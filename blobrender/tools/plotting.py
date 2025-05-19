import matplotlib.pyplot as plt
import os



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