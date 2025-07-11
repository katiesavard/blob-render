import numpy
import shutil
import argparse
from astropy.io import fits
from blobrender.paths import SIM_DAT, RESULTS
import os

def get_image(fitsfile):
        input_hdu = fits.open(fitsfile)[0]
        if len(input_hdu.data.shape) == 2:
                image = numpy.array(input_hdu.data[:,:])
        elif len(input_hdu.data.shape) == 3:
                image = numpy.array(input_hdu.data[0,:,:])
        else:
                image = numpy.array(input_hdu.data[0,0,:,:])
        return image


def flush_fits(newimage,fitsfile):
        """
        Update the data of a FITS file with a new image array.

        Parameters
        ----------
        newimage : numpy.ndarray
                The new image data to write into the FITS file. Its shape must be compatible with the target HDU's data shape.
        fitsfile : str
                Path to the FITS file to be updated.
        """

        f = fits.open(fitsfile,mode='update')
        input_hdu = f[0]
        if len(input_hdu.data.shape) == 2:
                input_hdu.data[:,:] = newimage
        elif len(input_hdu.data.shape) == 3:
                input_hdu.data[0,:,:] = newimage
        else:
                input_hdu.data[0,0,:,:] = newimage
        f.flush()

def copy_and_paste_fits(model_fits, wsclean_fits, op_fits):
        """
        Copy the wsclean fits file into the op_fits file and populate it with data from model_fits.

        Parameters
        ----------
        model_fits : str
                Path to the original FITS file containing simulated data.
        wsclean_fits : str
                Path to the wsclean FITS file to be copied.
        op_fits : str
                Path to the output FITS file where data will be populated.
        """
        
        shutil.copyfile(wsclean_fits, op_fits)  # Copy -image into -model fits
        img = get_image(model_fits)  # Original fits with your simulated data
        flush_fits(img, op_fits)  # Put data into the -model fits file
        outhdu = fits.open(op_fits,mode='update')
        outhdr = outhdu[0].header
        outhdr.set('BUNIT','JY/PIXEL') #set the units to Jy/pixel, this is what wsclean expects
        hh = outhdr.get('HISTORY') 
        for i in range(0,len(hh)):
                outhdr.remove('HISTORY')
                outhdu.flush() #destroy the evidence of this mess we made 

def add_chan_suffix(fname, ch_str):
    base, ext = os.path.splitext(fname)
    # Find the last '-word' in the base
    parts = base.split('-')
    if len(parts) > 1:
        # Insert channel string before the last part
        new_base = '-'.join(parts[:-1]) + f'-{ch_str}' + '-' + parts[-1]
    else:
        # No dash found, just append
        new_base = base + f'-{ch_str}'
    return f"{new_base}{ext}"

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_fits', required=True)
        parser.add_argument('--wsclean_fits', required=True)
        parser.add_argument('--op_fits', required=True)
        parser.add_argument('--nchan', required=True)
        args = parser.parse_args()
        nchan = int(args.nchan)

        if nchan == 1:
                model_fits = os.path.join(SIM_DAT, os.path.basename(args.model_fits)) #'fitsfile_name.fits'
                wsclean_fits = args.wsclean_fits #'predict_file_name-image.fits' (correct format, exists)
                op_fits = args.op_fits #'predict_file_name_0-model.fits' (doesnt yet exist, we will put sim data in here)
                copy_and_paste_fits(model_fits,wsclean_fits,op_fits)
        elif nchan > 1:
                for ch in range(nchan):
                       ch_str = f"{ch:04d}"
                       model_fits = os.path.join(SIM_DAT, os.path.basename(args.model_fits))
                       wsclean_fits = add_chan_suffix(args.wsclean_fits,ch_str)
                       op_fits = add_chan_suffix(args.op_fits,ch_str)
                       copy_and_paste_fits(model_fits,wsclean_fits,op_fits)
                ch_str = "MFS"
                model_fits = os.path.join(SIM_DAT, os.path.basename(args.model_fits))
                wsclean_fits = add_chan_suffix(args.wsclean_fits,ch_str)
                op_fits = add_chan_suffix(args.op_fits,ch_str)
                copy_and_paste_fits(model_fits,wsclean_fits,op_fits)
                       
if __name__ == "__main__":
        main()

