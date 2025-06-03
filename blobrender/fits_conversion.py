import sys
import numpy as np
from astropy.io import fits
import os
import yaml
import argparse
from PIL import Image

from .paths import PLOTS, CONFIGS, SIM_DAT
from . import tools
from blobrender.help_strings import HELP_DICT


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
            output_temp = print_and_save('deres row',output_string,verbose)
            resolution_difference = row_resolution/column_resolution
            image = np.array(deres_array(image,resolution_difference,axis='row'))
            output = print_and_save('New shape: '+str(np.shape(image)),output_temp,verbose)
        else:
            output_temp = print_and_save('deres column',output_string,verbose)
            resolution_difference = column_resolution/row_resolution
            image = np.array(deres_array(image,resolution_difference,axis='column'))
            output = print_and_save('New shape: '+str(np.shape(image)),output_temp,verbose)
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

def unit_print(xres,yres,L_sim,distance_in_pc,outstring,verbose):
    #now check what the resolution size is in real units for wsclean:
    x = tools.m_to_arcseconds(L_sim*xres,distance_in_pc)
    y = tools.m_to_arcseconds(L_sim*yres,distance_in_pc)

    outstring = print_and_save('x-direction res: {:.4f} arcseconds'.format(x),outstring,verbose)
    outstring = print_and_save('y-direction res: {:.4f} arcseconds'.format(y),outstring,verbose)
    return outstring, x

def print_and_save(string,output,verbose):
    if verbose: print(string)
    output+=string+'\n'
    return output

def load_image_or_npy(data_folder,filename):
    filepath = os.path.join(data_folder,filename)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        image_array = tools.load_list(data_folder,filename)
        return image_array
    else:
        image_array = np.flipud(tools.rgb2gray(np.array(Image.open(filepath)))) 
        return image_array
    
def main():

    ##############       defining variables       ############################

    verbose = True #prints to screen as well as to file
    update_yaml = True #update the yaml file with the new fits name
    reflect_image = False
    adjust_image = False

     # Load defaults from YAML

    yaml_file = os.path.join(CONFIGS,'default_simulation.yaml')
    description = "Convert simulation outputs (numpy or image format) to FITS images for use with blobrender." \
    "   Handles reshaping and resolution checks to make appropriate input to the prediction stafe." \
    "   Default values from default_prediction.yaml, unless otherwise specified with --config. Updates default_prediction.yaml accordingly." \

    args = tools.get_arguments(yaml_file,HELP_DICT,description)
    

    # Unpack arguments
    system_name = args.system_name
    nu_observe = args.nu_observe
    L_sim = args.L_sim
    distance_in_pc = args.distance_in_pc
    xres = args.xresolution
    yres = args.yresolution
    image_filename = args.image_filename

    
    

    ##############       setup       ############################
    #filenames
    
    data_folder = SIM_DAT
    output_string = '' #logging string

    #user checks
    output_string = print_and_save("System = "+system_name,output_string,verbose)

    if verbose:
        inp = input("Continue with current setup? (y/n) : ")
        if inp=='y':
            print("")
        else:
            return



    #load in data
    image_array = load_image_or_npy(SIM_DAT,image_filename)

    if reflect_image:
        image_array_flip = np.concatenate((np.flip(image_array,axis=1),np.array(image_array)),axis=1)
    else:
        image_array_flip = image_array 

    ##############       reshape array       ############################
    output_string = print_and_save('Array shape: '+str(np.shape(image_array_flip)),output_string,verbose)

    if adjust_image:
        #deres if needs
        image_array, output_string = deres_array_check(image_array_flip,verbose,output_string)

        #make even if needs
        image_array, output_string = even_shape_check(image_array,verbose,output_string)

    output_string = print_and_save('Array shape after adjustments: '+str(np.shape(image_array)),output_string,verbose) #input to wsclean
    nxpix = image_array.shape[0]
    nypix = image_array.shape[1]

    ##############       save to fits       ############################
    hdu = fits.PrimaryHDU(image_array)

    exta_descriptors=''
    fits_name = 'fits_'+system_name+'_'+str(nu_observe/1e9)+'GHz'+exta_descriptors+'.fits'
    save_to = data_folder+'/'+fits_name
    hdu.writeto(save_to,overwrite=True)


    #now check what the resolution size is in real units for wsclean:
    output_string, xasec = unit_print(xres,yres,L_sim,distance_in_pc, output_string,verbose) #input to wsclean

    if update_yaml:
        #update the yaml file with the new fits name
        yaml_path = os.path.join(CONFIGS,'default_prediction.yaml')
        tools.update_yaml('fitsfile_name',fits_name,yaml_path)
        tools.update_yaml('ypix',nxpix,yaml_path)
        tools.update_yaml('xpix',nypix,yaml_path)
        tools.update_yaml('scale',round(xasec,6),yaml_path)
   

    ##################       make an output file       ############################
    output_file = 'fits_info_'+system_name+'.txt'
    f = open(data_folder+'/'+output_file,'w')
    print(output_string)
    f.write(output_string)
    f.close()


if __name__ == "__main__":
    main()