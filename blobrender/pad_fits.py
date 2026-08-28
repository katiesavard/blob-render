from . import tools
import os
from .paths import CONFIGS, CONTAINERS, RESULTS, SIM_DAT
from blobrender.help_strings import HELP_DICT
from astropy.io import fits
import numpy as np
from math import ceil
import sys
from blobrender.tools.image_checks import (
    baseline_range_from_ms,
    check_image_fov,
    check_image_pixelsize,
    get_min_max_frequency,
)

def pad_fits(input_fits, output_fits, target_x, target_y):
    """
    Pad ya FITS image with zeros to reach (target_y, target_x) shape.
    Pads equally on both sides if possible.

    Parameters:
        input_fits (str): Path to input FITS file.
        output_fits (str): Path to output padded FITS file.
        target_x (int): Desired number of pixels in X (axis 1).
        target_y (int): Desired number of pixels in Y (axis 0).
    """
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        # Ensure target_x and target_y are even numbers
        if target_x % 2 != 0:
            target_x += 1
        if target_y % 2 != 0:
            target_y += 1
        # Assume 2D image (can be generalized)
        y, x = data.shape[-2:]
        # Calculate padding to keep the central pixel at the center
        orig_cx = (x - 1) / 2
        orig_cy = (y - 1) / 2
        new_cx = (target_x - 1) / 2
        new_cy = (target_y - 1) / 2
        pad_left = int(np.floor(new_cx - orig_cx))
        pad_right = int(np.ceil(new_cx - orig_cx))
        pad_top = int(np.floor(new_cy - orig_cy))
        pad_bottom = int(np.ceil(new_cy - orig_cy))

        pad_width = [(0, 0)] * (data.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
        padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)

        # Update header for new size (CRPIX1/2 should be shifted)
        if 'CRPIX1' in header:
            header['CRPIX1'] += pad_left
        if 'CRPIX2' in header:
            header['CRPIX2'] += pad_top
        header['NAXIS1'] = target_x
        header['NAXIS2'] = target_y

        # Copy all other extensions (if any) to preserve full FITS structure
        hdul_out = fits.HDUList()
        # Update the primary HDU with padded data and updated header
        hdul_out.append(fits.PrimaryHDU(data=padded_data, header=header))
        # Copy any additional HDUs unchanged
        for hdu in hdul[1:]:
            hdul_out.append(hdu.copy())

        hdul_out.writeto(output_fits, overwrite=True)
        return padded_data.shape[-1], padded_data.shape[-2]

def padded_fits_name(fname):
    base, ext = os.path.splitext(fname)
    # Check if the base already ends with '_padded'
    if base.lower().endswith('_padded'):
        sys.exit(f"HONQUE!! FITS file {fname} already padded. Remove '_padded' suffix to re-pad.") 
    if ext.lower() == '.fits':
        return f"{base}_padded{ext}"
    else:
        return f"{fname}_padded"

def main():
    should_update_yaml = True

    yaml_file = os.path.join(CONFIGS,'default_prediction.yaml')
    args = tools.get_arguments(yaml_file,HELP_DICT)

    split_ms_name = args.ms_name
    fitsfile_name = args.fitsfile_name
    xpix = int(args.xpix)
    ypix = int(args.ypix)
    scale = float(args.scale)
    fov = min(xpix, ypix)*scale

    padded_fits = padded_fits_name(fitsfile_name)

    fits_path = os.path.join(SIM_DAT, os.path.basename(fitsfile_name))
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        fits_ypix, fits_xpix = data.shape[-2:]
    if xpix != fits_xpix or ypix != fits_ypix:
        sys.exit(f"HONQUE HONQUE!: xpix/ypix from args ({xpix}, {ypix}) do not match FITS file dimensions ({fits_xpix}, {fits_ypix})")



    min_frequency, max_frequency = get_min_max_frequency(split_ms_name)
    min_b, max_b = baseline_range_from_ms(split_ms_name)

    good_pix, max_pix = check_image_pixelsize(scale, max_b, max_frequency)
    print(f"Pixel size {scale} arcsec is {'OK' if good_pix else 'TOO LARGE'} for max baseline {max_b:.2f} m at frequency {max_frequency/1e6:.3f} MHz. Max allowed pixel size is {max_pix:.2f} arcsec.")

    good_fov, min_fov = check_image_fov(fov, min_b, min_frequency)
    print(good_fov, min_fov)
    new_npix = int(ceil(min_fov / scale)*1.1) # 10% extra padding for those wanting to use uniform weighting when imaging
    if new_npix>5000:
        print (f"{new_npix}x{new_npix} is TOO large. Will continue anygays..")
        print (f"the good FOV value is {good_fov}, the minimum FOV value is {min_fov}")
        print (f"Current pixel scale: {scale:.6f} arcsec/pixel")
        print (f"Current FOV: {fov:.3f} arcsec")
        print (f"Minimum required FOV: {min_fov:.3f} arcsec")
        print (f"Required pixels at current scale: {new_npix}")
        print (f"Maximum allowed pixel scale: {max_pix:.3f} arcsec/pixel")
#set a value for new_npix for padding if its too large (capt new_npix):
       # new_npix=5000
 #this is the previous code:
   # if new_npix*2 > 100000:
    #    sys.exit(f"HONQUE HONQUE! New pixel size {new_npix}x{new_npix} is too large. Reduce pixel scale to fit within 100,000 pixels.")
    output_fits_path = os.path.join(SIM_DAT, os.path.basename(padded_fits))
    if not good_fov:
        newxpix, newypix = pad_fits(fits_path, output_fits_path, new_npix, new_npix)
        print(f"FITS image padded to {new_npix}x{new_npix} pixels to meet FOV*1.1 requirements.")
    else:
        print(f"FITS image already meets FOV requirements: {fov} arcsec >= {min_fov:.3f} arcsec.")

    if not good_fov and should_update_yaml:
        tools.update_yaml('xpix',newxpix,yaml_file)
        tools.update_yaml('ypix',newypix,yaml_file)
        tools.update_yaml('fitsfile_name',padded_fits,yaml_file)
    return 

if __name__ == "__main__":
    main()
