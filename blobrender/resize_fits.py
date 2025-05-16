from astropy.io import fits
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
sys.path.append("/Users/savard/PLUTO/pluto_playtime/plotting_analysis/")
from pyplutplot import * 
from fits_conversion import *
import aplpy

def header_adjust(fits_name,image_array_flip,RA_cent,DEC_cent,xres,yres):
    hdu = fits.PrimaryHDU(image_array_flip)
    hdu.writeto(fits_name,overwrite=True)
    hdul = fits.open(fits_name,mode='update')
    hdu = hdul[0]
    headers = hdu.header
    headers = hdu.header
    cent_pix1 = len(image_array_flip[0])/2
    cent_pix2 = len(image_array_flip.T[0])/2
    headers.set('BUNIT','JY/PIXEL')
    headers.set('BSCALE',1.)
    headers.set('BZERO',0.)
    headers.set('BTYPE','Intensity')
    headers.set('CRPIX1',cent_pix1)
    headers.set('CRPIX2',cent_pix2)
    headers.set('CUNIT1','deg     ')
    headers.set('CUNIT2','deg     ')

    headers.set('CRVAL1',RA_cent)
    headers.set('CRVAL2',DEC_cent)

    headers.set('CTYPE1','RA---SIN')
    headers.comments['CTYPE1'] = 'Right ascension angle cosine'
    headers.set('CTYPE2','DEC--SIN')
    headers.comments['CTYPE2'] = 'Declination angle cosine'
    headers.set('CDELT1',xres)
    headers.set('CDELT2',yres)
    hdul.flush()

def plot_fits(eht_fits,frame_flux,time,results_folder,imtstep):
    fig = plt.figure(figsize=(7, 7))
    f1 = aplpy.FITSFigure(eht_fits,figure=fig)
    f1.show_colorscale(cmap='inferno')
    f1.add_colorbar()
    f1.colorbar.show(pad=0.1)
    f1.colorbar.set_location('right')
    f1.colorbar.set_axis_label_text('Jy/beam')
    f1.add_scalebar(20*1e-6 * u.arcsecond)
    f1.scalebar.set_corner('top right')
    f1.scalebar.set_color('white')
    f1.scalebar.set_label(r'20$\mu$as')
    f1.add_label(x=0.1,y=0.1,text='{} mJy'.format(frame_flux),relative=True,color='pink')
    f1.set_title('{} seconds'.format(time))
    save_fig(results_folder,'plotfits_{}'.format(imtstep),overwrite=True)

def unit_print(D,image,L_sim,distance_in_pc,verbose):
    #now check what the resolution size is in real units for wsclean:
    ax1_check = D.x1[:len(image[0])]
    ax2_check = D.x2[:len(image.T[0])]
    x = m_to_arcseconds(ax1_check*L_sim,distance_in_pc)
    y = m_to_arcseconds(ax2_check*L_sim,distance_in_pc)
    xres = x[2]-x[1]
    yres = y[2]-y[1]
    if verbose:
        print('x-direction res: {:.8f} arcseconds'.format(xres))
        print('y-direction res: {:.8f} arcseconds'.format(yres))
    return xres/3600,yres/3600

def main():

    ######### system parameters
    system_name = 'ri0.001_rb0.02_lz2'

    args = sys.argv[1:]
    if len(args)==1:
        image_timestep = int(args[0])
    else:
        print('specify timestep')
        return
    
    L_sim = 465841907.0462244 #m
    nu_observe = (86.0 * 1e9) #GHz 
    distance_in_pc = 2960

    verbose = True
    print_result = True

    ######## define filenames
    data_dir = '/pluto_playtime/data_storage/'+system_name+'/' #set up working directory where data is stored
    results_folder = '/Users/savard/PLUTO/pluto_playtime/plotting_analysis/sim_results/{}'.format(system_name)
    eht_results_folder = '/Users/savard/PLUTO/pluto_playtime/plotting_analysis/sim_results/eht_scaled_sims'
    ez_filename = 'pixel_lum_'+system_name+'_'+str(image_timestep)+'_'+str(nu_observe/1e9)+'GHz'

    ####### load in the image
    image_array = load_list(eht_results_folder,ez_filename)
    image_array_flip = np.concatenate((np.flip(image_array,axis=1),np.array(image_array)),axis=1)

    #### ----- adjust luminosity array ---- ######

    #check if needs a deres
    image_array_flip = deres_array_check(image_array_flip,verbose) 

    #check if the shape is even
    image_array_flip = even_shape_check(image_array_flip,verbose)
    if verbose: print(np.shape(image_array_flip))

    #scale up in luminosity
    lum_scaler = 3e15
    eht_image_array = image_array*lum_scaler
    eht_image_array_flip = np.concatenate((np.flip(eht_image_array,axis=1),np.array(eht_image_array)),axis=1)
    frame_flux = np.sum(eht_image_array_flip*10**3)

    #create name for fits file 
    fits_name = 'ffile_'+system_name+'_'+str(image_timestep)
    exta_descriptors='_EHTscaled'
    eht_fits = eht_results_folder+'/'+fits_name+exta_descriptors+'.fits'

    D = load_data_obj(data_dir,image_timestep,data_type='flt')
    xres, yres = unit_print(D,image_array,L_sim,distance_in_pc,verbose) #get resolution
    del D

    #### ----- calculate the position of the central pixel ---- ######

    #position of the core and direction of propagation
    ra_core_deg = -84.9085916666667
    dec_core_deg = 7.18532694444445
    c1 = SkyCoord(ra_core_deg,dec_core_deg,unit=(u.deg, u.deg),frame='icrs')
    position_angle = 0.0 * u.deg

    #now load in what the separation should be 
    time = load_list(eht_results_folder,'timearray_projected_scaled') #seconds
    disp = load_list(eht_results_folder,'disparray_projected_scaled') #uas

    time_current = time[image_timestep] #seconds
    disp_current = disp[image_timestep] #uas

    separation = disp_current * u.microarcsecond
    new_pos = c1.directional_offset_by(position_angle, separation)
    RA_cent = new_pos.ra.degree
    DEC_cent = new_pos.dec.degree

    ###### save as a fits file with the correct headers 
    header_adjust(eht_fits,eht_image_array_flip,RA_cent,DEC_cent,xres,yres)

    #### plot the fits file 
    if print_result:
        hdul = fits.open(eht_fits,mode='readonly')
        hdu = hdul[0]
        headers = hdu.header
        if verbose: print(headers)
        plot_fits(eht_fits,frame_flux, time_current,eht_results_folder,image_timestep)



    




if __name__ == "__main__":
    main()