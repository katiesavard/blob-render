import sys
import tools
import numpy as np
from astropy.io import fits

def print_if(text,verbose):
    if verbose: print(text)


def deres_array_check(image,verbose):
    #check if needs a deres
    column_resolution = image[1][5]-image[1][4]
    row_resolution = image[5][1]-image[4][1]
    if row_resolution==column_resolution:
        print_if('square elements, no need to de-res',verbose)
    else:
        if row_resolution>column_resolution:
            print_if('deres row',verbose)
            resolution_difference = row_resolution/column_resolution
            image = np.array(deres_array(image,resolution_difference),axis='row')
            print_if('New shape: '+str(np.shape(image)),verbose)
        else:
            print_if('deres column',verbose)
            resolution_difference = column_resolution/row_resolution
            image = np.array(deres_array(image,resolution_difference),axis='column')
            print_if('New shape: '+str(np.shape(image)),verbose)
    return image

def even_shape_check(image,verbose):
    row_len = len(image[0])
    column_len = len(image.T[0])
    even = True
    if row_len%2==1:
        even=False
        print_if('adding another column',verbose)
        image = adjust(image,axis='column')
    if column_len%2==1:
        even=False
        print_if('adding another row',verbose)
        image = adjust(image,axis='row') 
    if even:
        print_if("image has even sides, no need to adjust",verbose)
    return image

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

def m_to_arcseconds(m,distance_in_pc):
    acs = (m*100)/(1.496*10**(13))*(1/distance_in_pc)
    return acs

def unit_print(D,L_sim,image,distance_in_pc,outstring):
    #now check what the resolution size is in real units for wsclean:
    ax1_check = D.x1[:len(image[0])]
    ax2_check = D.x2[:len(image.T[0])]
    x = m_to_arcseconds(ax1_check*L_sim,distance_in_pc)
    y = m_to_arcseconds(ax2_check*L_sim,distance_in_pc)

    outstring = print_and_save('x-direction res: {:.4f} arcseconds'.format(x[2]-x[1]),outstring)
    outstring = print_and_save('y-direction res: {:.4f} arcseconds'.format(y[2]-y[1]),outstring)
    return outstring

def print_and_save(string,output,verbose):
    print_if(string,verbose)
    output+=string+'\n'
    return output

def main():
    ##############       defining variables       ############################

    image_timestep = 150
    nu_observe = (1.28* 1e9) #1.28 GHz (Bright2020)
    L_sim = 1*10**13 #m
    distance_in_pc = 2960
    dtype = 'flt'
    verbose = True

    ##############       setup       ############################
    #filenames
    system_name = 'ri0.001_rb0.02_lz2'
    data_dir = '/pluto_playtime/data_storage/'+system_name+'/' #set up working directory where data is stored
    results_folder = '/Users/savard/PLUTO/pluto_playtime/plotting_analysis/sim_results/{}'.format(system_name)
    ez_filename = 'pixel_lum_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/1e9)+'GHz'
    output_string = ''

    #user checks
    output_string = print_and_save(" Image timestep = "+str(image_timestep)+"\n System = "+system_name,output_string)
    inp = input("Continue with current setup? (y/n) : ")
    if inp=='y':
        print("")
    else:
        return



    #load in data
    D = tools.pyplutplot.load_data_obj(data_dir,image_timestep,data_type=dtype)
    image_array = tools.pyplutplot.load_list(results_folder,ez_filename)
    image_array_flip = np.concatenate((np.flip(image_array,axis=1),np.array(image_array)),axis=1)

    ##############       reshape array       ############################
    output_string = print_and_save('Array shape: '+str(np.shape(image_array_flip)),output_string)


    #deres if needs
    image_array = deres_array_check(image_array_flip,verbose)

    #make even if needs
    image_array = even_shape_check(image_array,verbose)

    output_string = print_and_save('Array shape after adjustments: '+str(np.shape(image_array)),output_string,verbose) #input to wsclean


    ##############       save to fits       ############################
    hdu = fits.PrimaryHDU(image_array)
    fits_name = 'ffile_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/1e9)+'GHz'

    exta_descriptors=''

    save_to = results_folder+'/'+fits_name+exta_descriptors+'.fits'
    hdu.writeto(save_to,overwrite=True)

    #now check what the resolution size is in real units for wsclean:
    
    output_string = unit_print(D,L_sim,image_array,distance_in_pc, output_string) #input to wsclean


    ##################       make an output file       ############################
    output_file = 'fits_info_'+str(image_timestep)+'.txt'
    f = open(results_folder+'/'+output_file,'w')
    print(output_string)
    f.write(output_string)
    f.close()


if __name__ == "__main__":
    main()