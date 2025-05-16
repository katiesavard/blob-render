
import sys
from astropy.io import fits

def update_fits_header(file_path, new_ra, new_dec):
    # Open the FITS file in update mode
    with fits.open(file_path, mode='update') as hdul:
        # Access the primary header
        header = hdul[0].header
        
        # Update the RA and DEC values in decimal degrees
        header['CRVAL1'] = new_ra
        header['CRVAL2'] = new_dec
        
        # Save the changes
        hdul.flush()

# Example usage
file_path = sys.argv[1] #input filename when calling python script
new_ra = sys.argv[2]  # New RA value in decimal degrees
new_dec = sys.argv[3]  # New DEC value in decimal degrees

update_fits_header(file_path, new_ra, new_dec)
