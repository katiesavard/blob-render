from scipy.spatial.distance import pdist
from casatools import table
import numpy as np

def baseline_range(positions):
    """
    Compute min and max baseline lengths from Nx3 antenna position array.
    
    Parameters:
        positions (np.ndarray): shape (N_antennas, 3)
        
    Returns:
        tuple: (min_distance, max_distance) in meters
    """
    dists = pdist(positions)  # fast condensed distance matrix
    return np.min(dists), np.max(dists)

def baseline_range_from_ms(ms_path):
    tb = table()
    tb.open(ms_path + "/ANTENNA")
    positions = tb.getcol("POSITION").T  # shape: (N_antennas, 3)
    tb.close()
    return baseline_range(positions)

def check_image_pixelsize(pixel_size_arcsec, max_baseline_m, freq_hz):
    """
    Check if the input pixel size (arcsec) is greater than 1/(2 * (max_baseline / wavelength)) in radians.
    Returns True if pixel size is OK, False if too lorge and in chorge.

    Parameters:
        pixel_size_arcsec (float): Pixel size in arcseconds.
        max_baseline_m (float): Maximum baseline in meters.
        freq_hz (float): Observing frequency in Hz.

    Returns:
        bool: True if pixel size is >= 1/(2 * (max_baseline / wavelength)) in radians, else False.
    """
    c = 299792458.0  # speed of light in m/s
    wavelength = c / freq_hz    
    u_max = max_baseline_m / wavelength
    min_pixel_rad = 1 / (2 * u_max)  # radians
    pixel_rad = np.deg2rad(pixel_size_arcsec / 3600.0)
    return pixel_rad <= min_pixel_rad, np.degrees(min_pixel_rad) * 3600.0  # return pixel size in arcseconds for convenience

def check_image_fov(fov_arcsec, min_baseline_m, freq_hz):
    """
    Check if the input field of view (arcsec) is greater than 1/(2 * (min_baseline / wavelength)) in radians.
    Returns True if FOV is OK, False if too smol.

    Parameters:
        fov_arcsec (float): Field of view in arcseconds.
        min_baseline_m (float): Minimum baseline in meters.
        freq_hz (float): Observing frequency in Hz.

    Returns:
        bool: True if FOV is >= 1/(2 * (min_baseline / wavelength)) in radians, else False.
    """
    c = 299792458.0  # speed of light in m/s
    wavelength = c / freq_hz    
    u_min = min_baseline_m / wavelength
    min_fov_rad = 1 / (2 * u_min)  # radians
    fov_rad = np.deg2rad(fov_arcsec / 3600.0)
    return fov_rad >= min_fov_rad, np.degrees(min_fov_rad) * 3600.0  # return FOV in arcseconds for convenience

def get_min_max_frequency(ms_path):
    """
    Extract the minimum and maximum frequency (in Hz) from the SPECTRAL_WINDOW table of a Measurement Set.
    """
    from casatools import table
    tb = table()
    tb.open(ms_path + "/SPECTRAL_WINDOW")
    chanfreq = tb.getcol("CHAN_FREQ")  # shape: (nchan, nspw) or (nchan,)
    tb.close()
    freqs = np.array(chanfreq).flatten()
    return freqs.min(), freqs.max()


