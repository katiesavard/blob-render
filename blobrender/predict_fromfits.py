import sys
import subprocess
import os
import stat
sys.path.append("../")

#assumes you have fitsfile and split-MS in current directory 

#populate the scale and pix from the format of the fits file

#timestep is just for naming conventions 


def main():
	args = sys.argv[1:]

	default_fitsfile_name = 'ffile_ri0.001_rb0.02_k0.1_lz25_6hhres_newprs2_0_1.7GHz.fits'
	default_split_ms_name = 'split_data_1820.ms'
	xpix = '1300' #from fits_conversion.py
	ypix = '1300' #from fits_conversion.py
	scale = '0.0038' #arcseconds 
	timestep = '0'
	reposition_model = False
	rephase_real = False
	add_noise = False
	newRA = '-84.9099229'
	newDEC = '7.1835107'

	if len(args)==2:
		fitsfile_name = args[0] 
		split_ms_name = args[1]
	else:
		default = input('need 2 args: use default arguments? '+'\n'+'default arguments are: '
		+'\n'+default_fitsfile_name+'\n'+'MS to split: '+str(default_split_ms_name)+'\n'+
		'Reposition: '+str(reposition_model)+' '+newRA+' '+newDEC+'\n'+'(y/n)')
	if default=='y':
		fitsfile_name = default_fitsfile_name
		split_ms_name = default_split_ms_name
	else:
		return
	bash_runfile = 'run_predict.sh'
	predict_file_name = 'brender_meerkat_inpmodel_'+timestep
	image_file_name = 'brender_meerkat_modimage_'+timestep
	imagesum_file_name = 'brender_meerkat_sumimage_'+timestep

	f = open(bash_runfile,'w')
	f.write('#!/usr/bin/env bash\n')
	singularity = 'singularity exec /home/ianh/containers/oxkat-0.41.sif '
	
	#rephase real visibilities to where you want to add the sim data to
	if rephase_real:
		f.write('printf "rephasing real data\n" \n')
		f.write(singularity+'chgcentre '+split_ms_name+' 18h20m21.6185s +07d11m00.6386s\n') 
	
	#clean with 0 iterations: create fits file images with dirty visibilities but the right dimensions
	f.write('printf "creating model\n" \n')
	f.write(singularity+'wsclean -size '+xpix+' '+ypix+' -scale '+scale+'asec -niter 0 -channels-out 1 -reorder -name '+predict_file_name+' -data-column CORRECTED_DATA -use-wgridder -mem 50 '+split_ms_name+'\n')
	
	#write in the correct filenames into populatefits.py
	f.write('sed -i "s/model_fits.*fits\'/model_fits = \''+fitsfile_name+'\'/g" populatefits.py\n')
	f.write('sed -i "s/wsclean_fits.*fits\'/wsclean_fits = \''+predict_file_name+'-image.fits\'/g" populatefits.py\n')
	f.write('sed -i "s/op_fits.*fits\'/op_fits = \''+predict_file_name+'-model.fits\'/g" populatefits.py\n')

	#bulldoze the dirty fits files with the data from simulation
	f.write("python3 populatefits.py\n")
	
	#change the RA and DEC of the model fits files to the desired position
	if reposition_model:
		f.write('printf "changing RA and DEC of model\n" \n')
		f.write("python3 change_RADEC_fits.py "+predict_file_name+'-model.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 change_RADEC_fits.py "+predict_file_name+'-image.fits '+newRA+' '+newDEC+'\n')
		f.write("python3 change_RADEC_fits.py "+predict_file_name+'-dirty.fits '+newRA+' '+newDEC+'\n')
	
	#predict model visibilities
	f.write('printf "predicting model visibilities\n" \n')
	f.write(singularity+'wsclean -predict -mem 50 -size '+xpix+' '+ypix+' -scale '+scale+'asec -channels-out 1 -reorder -name '+predict_file_name+' -use-wgridder '+split_ms_name+'/\n')
		
	#image the model visibilities
	f.write('printf "imaging model data with no noise\n" \n')
	f.write(singularity+'wsclean -mem 80 -mgain 0.9 -gain 0.15 -size 1024 1024 -scale 1.1asec -niter 1000 -channels-out 1 -no-update-model-required -name '+image_file_name+' -data-column MODEL_DATA -use-wgridder '+split_ms_name+'\n')
	
	#add together model and real data
	if add_noise:
		f.write('printf "adding model to real data\n" \n')
		f.write(singularity+'python3 add_MS_column.py '+split_ms_name+' --colname DATA_MODEL_SUM\n')
		f.write(singularity+'python3 copy_MS_column.py '+split_ms_name+' --fromcol CORRECTED_DATA --tocol DATA_MODEL_SUM\n')
		f.write(singularity+'python3 sum_MS_columns.py '+split_ms_name+' --src MODEL_DATA --dest DATA_MODEL_SUM\n')

	#rephase the visibilities back to the original phase centre 
	if rephase_real:
		f.write('printf "rephasing real data back to original phase centre\n" \n')
		f.write(singularity+'chgcentre '+split_ms_name+' 18h20m21.938s 07d11m07.177s\n')  #DATA, MODEL_DATA and CORRECTED_DATA 
		f.write(singularity+'chgcentre -datacolumn DATA_MODEL_SUM '+split_ms_name+' 18h20m21.938s 07d11m07.177s\n')  #DATA_MODEL_SUM 
	
	#image the model+data according to emerlin recommended params
	if add_noise:
		f.write('printf "imaging model + data\n" \n')
		f.write(singularity+'wsclean -mem 80 -mgain 0.8 -gain 0.15 -size 5000 5000 -scale 5masec -niter 10000 -channels-out 1 -no-update-model-required -reorder -name '
		+imagesum_file_name+' -weight briggs 0.8 -data-column DATA_MODEL_SUM -use-wgridder '+split_ms_name+'\n')

	f.close()

	os.chmod(bash_runfile,stat.S_IRWXU)
	#run bash file 
	subprocess.call("./"+bash_runfile)

if __name__ == "__main__":
    main()

