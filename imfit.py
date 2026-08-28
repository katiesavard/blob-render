from casatasks import importfits, imfit
#for example:
#NOISY DATA: results/yesnoise/brender_e-MERLIN_modiamge_#-MFS-image.fits
# NO NOISE DATA: results/nonoise/brender_e-MERLIN_modimage_#_image.fits
fits_file='results/nonoise/approach/brender_MeerKAT_modimage_77-MFS-image.fits'
casaimage = 'results/nonoise/approach/brender_MeerKAT_modimage_77_a-MFS-image.im' #for blobs that need a (approach) or receeding (r)
# specification, remember to use for example 69_a or 69_r respectively from here onwards.
region_file='results/nonoise/approach/ellipse_MeerKAT_77_a.crtf'
log_file='MeerKAT_77_a.log'

importfits(fitsimage=fits_file,imagename=casaimage,overwrite= True)

result= imfit(imagename=casaimage, region=region_file, 
logfile=log_file) #logs will be spit in blob-render, read using cat e-MERLIN_#.log (include _NOISE for noisy data)

print(f"successfully created {log_file}")

