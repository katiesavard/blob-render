import sys
import numpy as np
from astropy.io import fits
import os
import yaml
import argparse

from .config import PLOTS
from . import tools


def deres_array_check(image,verbose,output_string):
    """
    Check if the image needs to be deresolutioned. (blobrender requires square pixels)
    It checks the resolution of the image in both x and y directions, and if they are not equal,
    it will remove every x rows or columns from the image to make them equal.

    Args:
        image (array): image array
        verbose (bool): if True, print to screen
        output_string (string): logging string

    Returns:
        image (array): adjusted array
        output_string (string): logging string
    """
    #check if needs a deres
    column_resolution = image[1][5]-image[1][4]
    row_resolution = image[5][1]-image[4][1]
    if row_resolution==column_resolution:
        output = print_and_save('square elements, no need to de-res',output_string,verbose)
    else:
        if row_resolution>column_resolution:
            output = print_and_save('deres row',output_string,verbose)
            resolution_difference = row_resolution/column_resolution
            image = np.array(deres_array(image,resolution_difference),axis='row')
            output = print_and_save('New shape: '+str(np.shape(image)),output_string,verbose)
        else:
            output = print_and_save('deres column',output_string,verbose)
            resolution_difference = column_resolution/row_resolution
            image = np.array(deres_array(image,resolution_difference),axis='column')
            output = print_and_save('New shape: '+str(np.shape(image)),output_string,verbose)
    return image, output

def even_shape_check(image,verbose,output_string):
    row_len = len(image[0])
    column_len = len(image.T[0])
    even = True
    if row_len%2==1:
        even=False
        output = print_and_save('adding another column',output_string,verbose)
        image = adjust(image,axis='column')
    if column_len%2==1:
        even=False
        output = print_and_save('adding another row',output_string,verbose)
        image = adjust(image,axis='row') 
    if even:
        output = print_and_save("image has even sides, no need to adjust",output_string,verbose)
    return image, output

def adjust(image_array,axis='row'): 
    #adds on an extra row
    #if it needs reshaping for wsclean (odd numbers)
    if axis=='row':
        z = [0]*(len(image_array.T))
        new_image = np.row_stack([z,image_array])
    elif axis=='column':
        z = [0]*(len(image_array))
        new_image = np.column_stack([z,image_array])
    return new_image

def deres_array(image_array,res,axis='row'): 
    #remove every x=res rows from array
    #if its higher resolution on one axis (to make pixels square)
    new_lum_array = []
    if axis=='row':
        for index, row in enumerate(image_array):
            if index%res==1:
                new_lum_array.append(row)
    elif axis=='column':
        for index, column in enumerate(image_array.T):
            if index%res==1:
                new_lum_array.append(column)

    return new_lum_array

def unit_print(xres,yres,L_sim,distance_in_pc,outstring):
    #now check what the resolution size is in real units for wsclean:
    x = tools.m_to_arcseconds(L_sim*xres,distance_in_pc)
    y = tools.m_to_arcseconds(L_sim*yres,distance_in_pc)

    outstring = print_and_save('x-direction res: {:.4f} arcseconds'.format(x),outstring)
    outstring = print_and_save('y-direction res: {:.4f} arcseconds'.format(y),outstring)
    return outstring

def print_and_save(string,output,verbose):
    if verbose: print(string)
    output+=string+'\n'
    return output

def get_arguments(yaml_file):
    with open(yaml_file, 'r') as f:
        defaults = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Converting a simulation data to a fits file")
    parser.add_argument('--system_name', type=str, default=defaults['system_name'], help='PLUTO output folder name')
    parser.add_argument('--image_timestep', type=int, default=defaults['image_timestep'], help='Timestep to analyse')
    parser.add_argument('--nu_observe', type=float, default=defaults['nu_observe'], help='Observing frequency (Hz)')
    parser.add_argument('--L_sim', type=float, default=defaults['L_sim'], help='PLUTO units: length (m)')
    parser.add_argument('--distance_in_pc', type=float, default=defaults['distance_in_pc'], help='Distance in parsecs')
    parser.add_argument('--x_resolution', type=float, default=defaults['x_resolution'], help='Number of pixels per unit length L_sim in x direction')
    parser.add_argument('--y_resolution', type=float, default=defaults['y_resolution'], help='Number of pixels per unit length L_sim in y direction')

    return parser.parse_args()

def main():
    ##############       defining variables       ############################

     # Load defaults from YAML
    yaml_file = 'default_framelum.yaml'

    args = get_arguments(yaml_file)

    # Unpack arguments
    system_name = args.system_name
    image_timestep = args.image_timestep
    nu_observe = args.nu_observe
    L_sim = args.L_sim
    distance_in_pc = args.distance_in_pc
    xres = args.x_resolution
    yres = args.y_resolution


    verbose = True #prints to screen as well as to file

    ##############       setup       ############################
    #filenames
    results_folder = os.path.join(PLOTS, system_name)

    ez_filename = 'pixel_lum_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/1e9)+'GHz'
    output_string = '' #logging string

    #user checks
    output_string = print_and_save(" Image timestep = "+str(image_timestep)+"\n System = "+system_name,output_string)

    if verbose:
        inp = input("Continue with current setup? (y/n) : ")
        if inp=='y':
            print("")
        else:
            return



    #load in data

    image_array = tools.pyplutplot.load_list(results_folder,ez_filename)
    image_array_flip = np.concatenate((np.flip(image_array,axis=1),np.array(image_array)),axis=1)

    ##############       reshape array       ############################
    output_string = print_and_save('Array shape: '+str(np.shape(image_array_flip)),output_string)


    #deres if needs
    image_array, output_string = deres_array_check(image_array_flip,verbose,output_string)

    #make even if needs
    image_array, output_string = even_shape_check(image_array,verbose,output_string)

    output_string = print_and_save('Array shape after adjustments: '+str(np.shape(image_array)),output_string,verbose) #input to wsclean


    ##############       save to fits       ############################
    hdu = fits.PrimaryHDU(image_array)
    fits_name = 'ffile_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/1e9)+'GHz'

    exta_descriptors=''

    save_to = results_folder+'/'+fits_name+exta_descriptors+'.fits'
    hdu.writeto(save_to,overwrite=True)

    #now check what the resolution size is in real units for wsclean:
    output_string = unit_print(xres,yres,L_sim,distance_in_pc, output_string) #input to wsclean


    ##################       make an output file       ############################
    output_file = 'fits_info_'+str(image_timestep)+'.txt'
    f = open(results_folder+'/'+output_file,'w')
    print(output_string)
    f.write(output_string)
    f.close()


if __name__ == "__main__":
    main()