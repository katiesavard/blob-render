import subprocess
import os
import stat
from . import tools

from .paths import CONFIGS, CONTAINERS, RESULTS
from blobrender.help_strings import HELP_DICT


#assumes you have fitsfile and split-MS in current directory 

#populate the scale and pix from the format of the fits file

#timestep is just for naming conventions 

def singularity_setup(use_remote=True):
    wsclean_sif = os.path.join(CONTAINERS, "wsclean-1.6.3.sif")
    if not os.path.exists(wsclean_sif):
        print(f"Singularity container not found at {wsclean_sif}. Pulling from DockerHub...")
        os.makedirs(CONTAINERS, exist_ok=True)
        if use_remote:
            # Remote build: output file must be in CONTAINERS
            build_cmd = [
                "singularity", "build", "--remote", wsclean_sif, "docker://stimela/wsclean:1.6.3"
            ]
            subprocess.run(build_cmd, check=True)
        else:
            pull_cmd = [
                "singularity", "pull", f"--dir={CONTAINERS}", "docker://stimela/wsclean:1.6.3"
            ]
            subprocess.run(pull_cmd, check=True)
        print("Singularity container downloaded.")
    singularity = 'singularity exec --bind $HOME '+wsclean_sif+' '
    return singularity

def main():

	
	# Load defaults from YAML

	yaml_file = os.path.join(CONFIGS,'default_prediction.yaml')
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


	#some specific requirements that I need to figure out how they depend on the telescope 
	column ='CORRECTED_DATA'
	reorder='-reorder' #blank for no reorder
	field = '' #blank for split dataset 
	imscale='5mas' #imaging scale for cleaning (some fraction of the beam)
	mem=str(50) #memory for wsclean
	
	container_type_lower = str(container_type).lower()
	if container_type_lower == 'singularity':
		container_setup = 'singularity exec --bind $HOME ' + container_name + ' '
	elif container_type_lower == 'docker':
		container_setup = 'docker run --rm -v $HOME:/home/user -w /home/user ' + container_name + ' '
	elif container_type_lower == 'none':
		container_setup = ''
	else:
		raise ValueError(f"Unknown container type: {container_type}. Use 'singularity', 'docker', or 'none'")

	bash_runfile = 'run_predict.sh'
	predict_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_inpmodel_'+timestep)
	image_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_modimage_'+timestep)
	imagesum_file_name = os.path.join(RESULTS,'brender_'+telescopename+'_sumimage_'+timestep)


	f = open(bash_runfile,'w')
	f.write('#!/usr/bin/env bash\n')
	
	#rephase real visibilities to where you want to add the sim data to
	if rephase_real:
		f.write('printf "rephasing real data\n" \n')
		f.write(container_setup+'chgcentre '+split_ms_name+' 18h20m21.6185s +07d11m00.6386s\n') 
	
	#clean with 0 iterations: create fits file images with dirty visibilities but the right dimensions
	f.write('printf "creating model\n" \n')
	f.write(container_setup+'wsclean -size '+xpix+' '+ypix+' -scale '+scale+'asec -niter 0 -channels-out 1 '+reorder+' -name '+predict_file_name+' -data-column '+column+' -use-wgridder -mem '+mem+' '+split_ms_name+'\n')
	
	#write in the correct filenames into populatefits.py
	#f.write(f'{sed_inplace} \"s/model_fits.*fits\'/model_fits = \''+fitsfile_name+'\'/g" populatefits.py\n')
	#f.write(f'{sed_inplace} \"s/wsclean_fits.*fits\'/wsclean_fits = \''+predict_file_name+'-image.fits\'/g" populatefits.py\n')
	#f.write(f'{sed_inplace} \"s/op_fits.*fits\'/op_fits = \''+predict_file_name+'-model.fits\'/g" populatefits.py\n')

	f.write(
    	f"python3 -m blobrender.tools.populatefits --model_fits {fitsfile_name} "
    	f"--wsclean_fits {predict_file_name}-image.fits "
    	f"--op_fits {predict_file_name}-model.fits \n"
	)
	#bulldoze the dirty fits files with the data from simulation
	#f.write("python3 populatefits.py\n")
	
	#change the RA and DEC of the model fits files to the desired position
	if reposition_model:
		f.write('printf "changing RA and DEC of model\n" \n')
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-model.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-image.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 -m blobrender.tools.change_RADEC_fits "+predict_file_name+'-dirty.fits '+newRA+' '+newDEC+'\n')
	
	#predict model visibilities
	f.write('printf "predicting model visibilities\n" \n')
	f.write(container_setup+'wsclean -predict -mem '+mem+' -size '+xpix+' '+ypix+' -scale '+scale+'asec -channels-out 1 '+reorder+' -name '+predict_file_name+' -use-wgridder '+split_ms_name+'/\n')
		
	#image the model visibilities
	f.write('printf "imaging model data with no noise\n" \n')
	f.write(container_setup+'wsclean -mem 80 -mgain 0.9 -gain 0.15 -size 1024 1024 -scale '+imscale+' -niter 1000 -channels-out 1 -no-update-model-required '+reorder+' -name '+image_file_name+' -data-column MODEL_DATA '+field+' -use-wgridder '+split_ms_name+'\n')
	
	#add together model and real data
	if add_noise:
		f.write('printf "adding model to real data\n" \n')
		f.write(container_setup+'python3 -m blobrender.tools.add_MS_column '+split_ms_name+' --colname DATA_MODEL_SUM\n')
		f.write(container_setup+'python3 -m blobrender.tools.copy_MS_column '+split_ms_name+' --fromcol CORRECTED_DATA --tocol DATA_MODEL_SUM\n')
		f.write(container_setup+'python3 -m blobrender.tools.sum_MS_columns '+split_ms_name+' --src MODEL_DATA --dest DATA_MODEL_SUM\n')

	#rephase the visibilities back to the original phase centre 
	if rephase_real:
		f.write('printf "rephasing real data back to original phase centre\n" \n')
		f.write(container_setup+'chgcentre '+split_ms_name+' 18h20m21.938s 07d11m07.177s\n')  #DATA, MODEL_DATA and CORRECTED_DATA 
		f.write(container_setup+'chgcentre -datacolumn DATA_MODEL_SUM '+split_ms_name+' 18h20m21.938s 07d11m07.177s\n')  #DATA_MODEL_SUM 
	
	#image the model+data according to emerlin recommended params
	if add_noise:
		f.write('printf "imaging model + data\n" \n')
		f.write(container_setup+'wsclean -mem 80 -mgain 0.8 -gain 0.15 -size 5000 5000 -scale 5masec -niter 10000 -channels-out 1 -no-update-model-required -reorder -name '
		+imagesum_file_name+' -weight briggs 0.8 -data-column DATA_MODEL_SUM -use-wgridder '+split_ms_name+'\n')

	f.close()

	os.chmod(bash_runfile,stat.S_IRWXU)
	#run bash file 
	subprocess.call("./"+bash_runfile)


if __name__ == "__main__":
    main()

