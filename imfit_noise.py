from casatasks import importfits, imfit
import numpy as np
#for example:
#NOISY DATA: results/yesnoise/brender_e-MERLIN_modiamge_#-MFS-image.fits
# NO NOISE DATA: results/nonoise/brender_e-MERLIN_modimage_#_image.fits
time_files=np.array([0, 53])
for file in time_files: #use f strings for the file to be read as time_files:
    fits_file= f'results/yesnoise/final_reced/brender_e-MERLIN_modimage_{file}-MFS-image.fits'
    casaimage = f'results/yesnoise/final_reced/brender_e-MERLIN_modimage_{file}_RECED-MFS-image.im' #for blob>
# specification, remember to use for example 69_a or 69_r respectively from her>
    region_file= f'results/yesnoise/final_reced/ellipse_eMERLIN_{file}_RECED.crtf'
    log_file= f'eMERLIN_{file}_RECED.log'

    importfits(fitsimage=fits_file,imagename=casaimage,overwrite= True)

    result= imfit(imagename=casaimage, region=region_file, 
    logfile=log_file) #logs will be spit in blob-render, read using cat e-MERLIN_#.>

    print(f"successfully created {log_file}")
