import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

def twoD_gaussian(coords, amp, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amp * np.exp(-(a*(x - xo)**2 + 2*b*(x - xo)*(y - yo) + c*(y - yo)**2))
    return g.ravel()

def fit_psf_beam(fitsfile):
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data.squeeze()
        header = hdul[0].header
        wcs = WCS(header)
        pixscale_deg = np.abs(wcs.wcs.cdelt[0])
        pixscale_arcsec = pixscale_deg * 3600

    # Find peak
    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    size = 20
    cutout = data[y0-size//2:y0+size//2, x0-size//2:x0+size//2]
    x = np.linspace(0, cutout.shape[1]-1, cutout.shape[1])
    y = np.linspace(0, cutout.shape[0]-1, cutout.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = (cutout.max(), size//2, size//2, 3, 3, 0, np.median(cutout))
    popt, _ = curve_fit(twoD_gaussian, (x, y), cutout.ravel(), p0=initial_guess,maxfev=10000)

    # Convert sigma to FWHM
    fwhm_x = 2.355 * popt[3] * pixscale_arcsec
    fwhm_y = 2.355 * popt[4] * pixscale_arcsec
    pa_deg = np.degrees(popt[5])%180

    return fwhm_x, fwhm_y, pa_deg

def recommend_pixel_scale(beam_major_arcsec, beam_minor_arcsec=None, pixels_per_beam=4):
    """
    Recommend a pixel scale for imaging based on the synthesized beam size.
    
    Parameters:
        beam_major_arcsec (float): Major axis of the beam in arcseconds.
        beam_minor_arcsec (float or None): Minor axis of the beam in arcseconds (optional).
        pixels_per_beam (int): Number of pixels to sample the beam across (default: 4).
    
    Returns:
        float: Recommended pixel scale in arcseconds per pixel.
    """
    if beam_minor_arcsec is None:
        scale = beam_major_arcsec / pixels_per_beam
    else:
        scale = (beam_major_arcsec * beam_minor_arcsec)**0.5 / pixels_per_beam
    return scale

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate the beam size from a FITS file.')
    parser.add_argument('--fitsfile', type=str, required=True, help='Path to the FITS file containing the PSF.')
    parser.add_argument('--pixels_per_beam', type=int, default=4, help='Number of pixels per beam (default: 4).')

    args = parser.parse_args()

    fitsfile = args.fitsfile
    pixels_per_beam = args.pixels_per_beam
    fwhm_x, fwhm_y, pa_deg = fit_psf_beam(fitsfile)
    major = max(fwhm_x, fwhm_y)
    minor = min(fwhm_x, fwhm_y)
    pixscale = recommend_pixel_scale(major,minor,pixels_per_beam)
    print(pixscale)


    return



if __name__ == '__main__':

    main()