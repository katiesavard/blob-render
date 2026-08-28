import subprocess
import os
import stat
from . import tools
import numpy as np


from .paths import CONFIGS, CONTAINERS, TEL_INFO, RESULTS
from blobrender.help_strings import HELP_DICT
from blobrender.tools.image_checks import (
	baseline_range_from_ms,
	check_image_fov,
	check_image_pixelsize,
	get_min_max_frequency,
)


#assumes you have fitsfile and split-MS in current directory 

#populate the scale and pix from the format of the fits file

#timestep is just for naming conventions 



def main():

	
	# Load defaults from YAML

	#yaml_file = os.path.join(CONFIGS,'default_prediction.yaml')
	yaml_file = os.path.join('default_prediction_53_r.yaml')
	args = tools.get_arguments(yaml_file,HELP_DICT)

	# Unpack arguments
	reposition_model = args.reposition_model
	rephase_real = args.rephase_real
	add_noise = args.add_noise
	newRA = args.newRA
	newDEC = args.newDEC
	xpix = args.xpix
	ypix = args.ypix
	scale = args.scale
	split_ms_name = args.ms_name
	fitsfile_name = args.fitsfile_name
	telescopename = args.telescopename
	timestep = str(args.image_timestep)
	container_name = args.container_name
	container_type = args.container_type

	# these are from the MS builder YAML file
	nchannels = '8'
	delta_nu = '73.07e6' #'8.12698e6' #'27.74e6' # '122.857e6'  #'73.14e6' #'73.14e6' #Compute as bandwidth/(nchannels-1) '62.5e6'  # Bandwidth per channel in Hz
	t_int = '4.0'  # Integration time in seconds
	pixels_per_beam = '10'  # Number of pixels per beam
	add_field = False

	### check that the pixel size and FOV of your model image are appropriate for the telescope you are using
	
	min_frequency, max_frequency = get_min_max_frequency(split_ms_name)
	min_b, max_b = baseline_range_from_ms(split_ms_name)
	good_pixel_size, max_pix = check_image_pixelsize(float(scale), max_b, max_frequency)
	if not good_pixel_size:
		raise ValueError(f"HONQUE HONQUE! Pixel size {scale} arcsec is too large for the maximum baseline {max_b:.2f} m at frequency {max_frequency/1e6:.3f} MHz. Reduce pixel size of your image to {max_pix:.2f}\" .. or else.")
	fov = min(float(xpix), float(ypix))*float(scale)  # field of view in arcseconds
	good_fov, min_fov = check_image_fov(fov, min_b, min_frequency)
	print(fov, min_fov)
	if not good_fov:
#Do the following for MeerKAT:
		print (f"WARNING: Continuing with a smaller than recommended FOV of {fov:.3f} arcesec (recommended is {min_fov:.3f} arcsec)! ")
#This next code could work with e-MERLIN for example
#raise ValueError(f"HONQUE HONQUE! Field of view {fov} arcsec is too small for the minimum baseline {min_b:.2f} m at frequency {min_frequency/1e6:.3f} MHz. Increase field of view of your image to at least {min_fov:.3f}\" .. or else.")
#I have changed the raise value to simply printing, so the code continues
	

	#some specific requirements that I need to figure out how they depend on the telescope 
	reorder='-reorder' #blank for no reorder
	field = '' #blank for split dataset 
	mem=str(50) #memory for wsclean
	
	cont = os.path.join(CONTAINERS,container_name)
	container_type_lower = str(container_type).lower()

	if container_type_lower == 'singularity':
		container_setup = f'singularity exec --bind {os.getcwd()},{RESULTS} ' + cont + ' '
	elif container_type_lower == 'docker':
		container_setup = f'docker run --rm -v {os.getcwd()}:/home/user -v {RESULTS}:{RESULTS} -w /home/user ' + container_name + ' '
	elif container_type_lower == 'none':
		container_setup = ''
	else:
		raise ValueError(f"Unknown container type: {container_type}. Use 'singularity', 'docker', or 'none'")

	bash_runfile = 'run_predict.sh'
	predict_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_inpmodel_'+timestep) #where you instert the simulation
	image_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_modimage_'+timestep) #images from cleaning process
	imagesum_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_sumimage_'+timestep)


	f = open(bash_runfile,'w')
	f.write('#!/usr/bin/env bash\n')
	f.write('set -e\n')
	#this is confusing me so i will comment the entire if statement. the phase centre in this way begins as the MS RA and DEC coordinates.
	#rephase real visibilities to where you want to add the sim data to
#	if rephase_real:
#		core_python="/home/rosalesvargas/blobrender-env/bin/python3"
#		f.write('printf "rephasing real data\n" \n')
		#chgcentre="/home/rosalesvargas/blob-render/chgcentre"
#		f.write(container_setup+'chgcentre '+split_ms_name+' 18h20m21.938s 07d11m07.177s\n') #18h20m21.6185s +07d11m00.6386s\n')
		#f.write('printf "injecting core at phase centre\n"\n')
		#f.write(core_python+' -m inject_core --ms_path '+split_ms_name+' --column MODEL_DATA\n') 
	
	#clean with 0 iterations: create fits file images with dirty visibilities but the right dimensions
	f.write('printf "creating model\n" \n')
	f.write(container_setup+'wsclean -size '+xpix+' '+ypix+' -scale '+scale+'asec -niter 0 -make-psf -channels-out ' + nchannels +' '+reorder+' -name '+predict_file_name+' -data-column DATA -use-wgridder -mem '+mem+' '+split_ms_name+'\n')
	
	#create a -model fits file with the simulated data in it with the same format as the -image file produced from the previous cleaning step 
	f.write(
    	f"python3 -m blobrender.tools.populatefits --model_fits {fitsfile_name} "
    	f"--wsclean_fits {predict_file_name}-image.fits "
    	f"--op_fits {predict_file_name}-model.fits "
		f"--nchan {nchannels}\n"
	)
	
	#change the RA and DEC of the model fits files to the desired position
	if reposition_model:
#change here python3 to the environment where python is:
		#reposition_python="/home/rosalesvargas/blobrender-env/bin/python"
#do a for loop for all considered channels, so that the RA and DEC of the blob are propperly updated:
		f.write('printf "changing RA and DEC of model\n" \n')
		#f.write('for fitsfile in '+predict_file_name+'*-model.fits; do\n python3 -m blobrender.tools.change_RADEC_fits "$fitsfile" '+newRA+' '+newDEC+' \ndone\n')
		#f.write('for fitsfile in '+predict_file_name+'*-image.fits; do\n python3 -m blobrender.tools.change_RADEC_fits "$fitsfile" '+newRA+' '+newDEC+' \ndone\n')
		#f.write('for fitsfile in '+predict_file_name+'*-dirty.fits; do\n python3 -m blobrender.tools.change_RADEC_fits "$fitsfile" '+newRA+' '+newDEC+' \ndone\n')
#THIS IS A SIMPLIFICATION USING THE MFS FITS FILES:
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-MFS-model.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-MFS-image.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-MFS-dirty.fits '+newRA+' '+newDEC+'\n')
	
	#predict model visibilities
	f.write('printf "predicting model visibilities\n" \n')
	f.write(container_setup+'wsclean -predict -mem '+mem+' -size '+xpix+' '+ypix+' -scale '+scale+'asec -channels-out '+nchannels+' '+reorder+' -name '+predict_file_name+' -use-wgridder '+split_ms_name+'/\n')
	
	#inject core here????
#	if rephase_real:
#		core_python = "/home/rosalesvargas/blobrender-env/bin/python3"
#		f.write(container_setup+'chgcentre -datacolumn DATA_MODEL_SUM '+split_ms_name+' 18h20m21.939s 07d11m07.17s\n') #rephase to where we want the core (usually same RA an>
#		f.write('printf "injecting core at phase centre\n"\n') #inject the core using the defined inject_core.py function, do not call with .py
#		f.write(core_python+' -m inject_core --ms_path '+split_ms_name+' --column DATA_MODEL_SUM --core_flux 5.26e-3\n') #the --core_flux call is in Jy!!

	#estimate the beam size from the PSF fits file
	f.write('pixscale=$(python3 -m blobrender.tools.calc_beamsize --fitsfile '+predict_file_name+' --pixels_per_beam '+pixels_per_beam+' --nchan '+nchannels+')\n')
	f.write('echo "Pixel scale is $pixscale arcseconds"\n')

	###add noise to the model visibilities
	if add_noise:
		noise_python= "/home/rosalesvargas/blobrender-env/bin/python"
		f.write('printf "adding noise to model visibilities\n" \n')
		#load the sefd config file
		sefd_dict = os.path.join(TEL_INFO,'{}_SEFD.yaml'.format(telescopename))
		#f.write(noise_python+' blobrender/tools/add_MS_column.py --ms_path '+split_ms_name+' --modelcol MODEL_DATA --newcol NOISY_MODEL2\n')
		f.write(noise_python+' blobrender/tools/add_MS_column.py --ms_path '+split_ms_name+' --modelcol MODEL_DATA\n')
#after this step choose to use MODEL_DATA or NOISY_DATA, the latter will reproduce the noisy data with the injected core!
#change here python3 to /home/rosalesvargas/blobrender-env/bin/python to run the correct python that will read the modules accordingly
		f.write(noise_python+' blobrender/tools/add_noise_pyrap.py --ms_path '+split_ms_name+' --sefd_dict_filename '+sefd_dict+' --delta_nu '+delta_nu+' --t_int '+t_int+' --column NOISY_MODEL\n')
		f.write('printf "noise added to model visibilities\n" \n')
	#add this new statement that defines what type of data we use (noisy or noiseless) in the next commands:
	if add_noise:
		model_column='NOISY_MODEL'
	else:
		model_column='MODEL_DATA'

	#image the model visibilities. -weight natural for e-MERLIN, and -weight uniform for MeerKAT,
#for e-MERLIN, gain is 0.15. For MeerKAT, gain is 0.1
#for e-MERLIN, auto-mask 3. For MeerKAT, auto-mask 4.5
#for e-MERLIN, DO NOT USE -local-rms. For MeerKAT, use -local-rms !!!!! Add this after niter :)))
	f.write('printf "imaging model data \n" \n')
	f.write(container_setup+'wsclean -mem 80 -mgain 0.9 -gain 0.15 -size 1024 1024 -scale ${pixscale}asec -niter 10000 -auto-mask 3 -auto-threshold 0.3 -channels-out '+nchannels+ ' -join-channels -no-update-model-required -weight natural '+reorder+' -name '+image_file_name+' -data-column '+model_column+' '+field+' -use-wgridder '+split_ms_name+'\n')
	
	#add together model and real data
	if add_field:
		field_python="/home/rosalesvargas/blobrender-env/bin/python3"
		f.write('printf "adding model to real data\n" \n')
		f.write(field_python+' -m blobrender.tools.add_MS_column --ms_path '+split_ms_name+' --modelcol '+model_column+' --newcol DATA_MODEL_SUM\n')
		#f.write(field_python+' -m blobrender.tools.add_MS_column --ms_path '+split_ms_name+' --newcol DATA_MODEL_SUM\n')
		f.write(field_python+' -m blobrender.tools.copy_MS_column '+split_ms_name+' --fromcol CORRECTED_DATA --tocol DATA_MODEL_SUM\n')
		f.write(field_python+' -m blobrender.tools.sum_MS_columns '+split_ms_name+' --src '+model_column+' --dest DATA_MODEL_SUM\n')

	#rephase the visibilities back to the original phase centre 
	if rephase_real:
		#chgcentre_container_setup="/home/rosalesvargas/blob-render/chgcentre"
		core_python = "/home/rosalesvargas/blobrender-env/bin/python3"
		f.write('printf "rephasing real data back to original phase centre\n" \n')
		f.write(container_setup+'chgcentre '+split_ms_name+' 18h20m22.10s +07d11m12.3s\n') #rephase to where blob lies.  #DATA, MODEL_DATA and CORRECTED_DATA 
		#rephase to where the black hole lives, I will keep this RA and DEC constant as this is our reference point.
		#if we change the core RA and DEC, we would have to take into account an apparent motion.
		f.write(container_setup+'chgcentre -datacolumn DATA_MODEL_SUM '+split_ms_name+' 18h20m21.93s 07d11m07.1s\n') #rephase to where we want the core (usually same RA and DEC as MS)  #DATA_MODEL_SUM 
		f.write('printf "injecting core at phase centre\n"\n') #inject the core using the defined inject_core.py function, do not call with .py
		f.write(core_python+' -m inject_core --ms_path '+split_ms_name+' --column DATA_MODEL_SUM --core_flux 0.153e-3\n') #the --core_flux call is in Jy!!
	
	#image the model+data according to emerlin recommended params
	if add_field:
		f.write('printf "imaging model + data\n" \n')
		f.write(container_setup+'wsclean -mem 80 -mgain 0.8 -gain 0.15 -size 5000 5000 -scale 5masec -niter 10000 -channels-out '+nchannels+' -join-channels -no-update-model-required -reorder -name '
		+imagesum_file_name+' -weight natural -data-column DATA_MODEL_SUM -use-wgridder '+split_ms_name+'\n')
#previously had in this if statement a -weight briggs 0.8, changed it to -weight natural
	f.close()

	os.chmod(bash_runfile,stat.S_IRWXU)
	#run bash file 
	subprocess.call("./"+bash_runfile)
	

if __name__ == "__main__":
    main()
