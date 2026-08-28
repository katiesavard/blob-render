import os
from datetime import datetime

import numpy as np
import sys
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
import astropy.units as u


from blobrender.help_strings import HELP_DICT
from blobrender.paths import TEL_INFO, CONFIGS
from blobrender.tools.basics import get_arguments, update_yaml
from blobrender.tools.casa_setup import check_casa_observatory, import_casa_tools


def wgs84_to_ecef(lon_deg, lat_deg, height_m):
    """
    Convert lat and long (degrees) and height / elevation (metres)
    to Earth-centred XYZ coordinates based on the WGS84 geoid.
    """

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # WGS84 ellipsoid constants
    a = 6378137.0           # semi-major axis in metres
    f = 1 / 298.257223563   # flattening
    e2 = f * (2 - f)        # eccentricity squared

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + height_m) * np.cos(lat) * np.cos(lon)
    Y = (N + height_m) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + height_m) * np.sin(lat)

    return X, Y, Z


def ecef_to_wgs84(X, Y, Z):
    """
    Convert ECEF XYZ (meters) to latitude (deg), longitude (deg), and height (meters)
    using WGS84 ellipsoid. Vectorized and supports scalar or array input.
    """
    # WGS84 constants
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    b = a * (1 - f)

    # Ensure arrays
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    Z = np.atleast_1d(Z)

    # Compute longitude
    lon = np.arctan2(Y, X)

    # Compute intermediate values
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - e2))
    lat_prev = lat + 2  # force entry into loop

    tol = 1e-12
    while np.max(np.abs(lat - lat_prev)) > tol:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(Z, p * (1 - e2 * N / (N + h)))

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N

    # Convert to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    # Return scalars if scalar input
    if lat_deg.size == 1:
        return lon_deg[0], lat_deg[0], h[0]
    return lon_deg, lat_deg, h

def geometric_mean_antenna(antennas_xyz):
    """
    Given a list or array of antenna ECEF XYZ coordinates (meters),
    return the geometric mean as (longitude_deg, latitude_deg, altitude_m).
    """
    antennas_xyz = np.array(antennas_xyz)
    mean_xyz = np.mean(antennas_xyz, axis=0)
    lon, lat, alt = ecef_to_wgs84(*mean_xyz)
    return lon, lat, alt 


def compute_obs_time(ra_str, start_ha_str, obs_date, site_lon):
    """
    Compute the UTC observation time given RA, start HA, observing date, and observatory longitude.

    Parameters:
        ra_str (str): Right Ascension (e.g., '18h20m21.938s')
        start_ha_str (str): Start Hour Angle (e.g., '-0.2h')
        obs_date (str): Observation date 'YYYY-MM-DD'
        site_lon (float): Observatory longitude in radians 

    Returns:
        str: UTC observation time (YYYY/MM/DD/HH:MM:SS)
    """
    # Convert inputs
    site_lon_deg = np.degrees(site_lon)
    ra = Angle(ra_str)
    start_ha = Angle(start_ha_str)
    lst_target = (ra + start_ha).wrap_at(12 * u.hourangle)

    # Observatory location (latitude is not needed for LST)
    location = EarthLocation(lat=0*u.deg, lon=site_lon_deg*u.deg)  # lat dummy here

    # Initial time guess near midnight
    t_guess = Time(f"{obs_date} 00:00:00", scale='utc', location=location)

    # Compute time offset to reach desired LST
    delta_lst = (lst_target - t_guess.sidereal_time('apparent')).wrap_at(12 * u.hourangle)
    delta_sec = delta_lst.hour * 3600
    t_start = t_guess + delta_sec * u.s

    return t_start.utc.strftime("%Y/%m/%d/%H:%M:%S")


def main():

    #--------------------------------------------------------------
    # Observational setup
    # Use caution when testing as realistic dump times and channel widths
    # will generate an SKA-Mid-sized Measurement Set

    overwrite = True # Remove any existing MS with the same name
    should_update_yaml = True # Update the YAML file with the new MS name


    yaml_file = os.path.join(CONFIGS,'default_MSbuilder.yaml')
    args = get_arguments(yaml_file,HELP_DICT)
    simulator, measures, _table = import_casa_tools(fix_data=args.fix_casa_data)
    sm = simulator()
    me = measures()
    
    telescopename = args.telescopename # 'SKA-Mid'

    ms_name = args.new_ms_name

    # Sky direction (J2000)
    source_name = 'BLOB'
    source_ra = args.source_ra #'12h00m00.0s'
    source_dec = args.source_dec #'-40d00m0.0s'

    # Frequency setup
    f0 = args.f0 #'856MHz' # lowest band frequency, or central frequency??
    bandwidth = args.bandwidth #total bandwidth in MHz
    nchan = args.nchan #8 # number of channels per spectral window
    df = (bandwidth)/ (nchan-1.0) # channel width in MHz
    #nspw= 8 #number of spectral windows.
    #spw_bandwidth=bandwidth/nspw #total bandwidth per spectral window
    #df=spw_bandwidth/nchan #bandwidth for each channel
    #lowest_band_edge=f0-(bandwidth/2)
    # Start time and track length


    #####now_utc = datetime.utcnow() #this can be changed to a specific time if desired
    #####obs_time = now_utc.strftime("%Y/%m/%d/%H:%M:%S") #removed -m and -d for windows compatibility
    
    obs_date=args.obs_date
    #obs_date = datetime.utcnow().strftime("%Y-%m-%d") # Observation date in YYYY-MM-DD format
    #changing the start reference time so that it corresponds to the correct transit time, this requires knowing
    #location of the telescope so will do later on. 
    #for now just specifying the date and working out the compatible time later
    #will include this as a yaml option later




    t_int = args.t_int #'120s' # Note that this is the correlator dump time, not total track length
    start_ha = args.start_ha #'-0.2h' # Start and end times are relative to transit
    end_ha = args.end_ha #'+0.2h'
    elevation = 1000.0 # Elevation above sea level in metres, used for antenna positions, assuming 1km for now 
    mount_type = 'alt-az'

    ### this will ONLY be used if overtly specified
    override_reference_location = False
    # Inferred emerlin reference point 
    lon_deg = -2.596399       # Longitude in degrees
    lat_deg = 53.041153       # Latitude in degrees
    alt_m   = -6193.071004    # Ellipsoidal height in meters
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    reference_pos = me.position('ITRF', lon_rad, lat_rad, alt_m)

    #--------------------------------------------------------------
    # Read the antenna names and coordinates (name,lon,lat,elevation,diameter)

#seems like it should be name, lon, lat, elevation, diameter...
    antenna_filename = telescopename + '.txt'
    layout_file = os.path.join(TEL_INFO, antenna_filename)
    f = open(layout_file)
    antenna_positions = []
    antenna_names = []
    antenna_diameters = []

#USE THIS PART OF THE CODE in the following way, first array is for e-MERLIN, second is for MeerKAT.

#e-MERLIN takes lat as cols[1] and lon as cols[2].

#MeerKAT takes lat as cols[2] and lon as cols[1]. Careful with capital letters for MeerKAT antenna names.

    selected_antennas = ['Mk2', 'Kn', 'De', 'Da', 'Cm']
#    selected_antennas = ['M000', 'M002', 'M003', 'M004', 'M005', 'M006', 'M009', 'M010', 'M011', 'M012', 'M013',
#'M014', 'M015', 'M016', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M026', 'M027', 'M028', 'M029',
#'M030', 'M031', 'M034', 'M035', 'M036', 'M038', 'M039', 'M040', 'M041', 'M042', 'M043', 'M044', 'M045', 'M046',
#'M047', 'M048', 'M050', 'M052', 'M053', 'M054', 'M055', 'M056', 'M057', 'M058', 'M059', 'M060', 'M061', 'M062', 'M063']
    for line in f:
        cols = line.split()

        ant_name = cols[0]

        if ant_name not in selected_antennas:
            continue #skip this step for antennas that are not in the selected_antenna array.
        lat=float(cols[1]) #1 for e-MERLIN. 2 for MeerKAT
        lon=float(cols[2]) #2 for e-MERLIN. 1 for MeerKAT
        elev=float(cols[3])
        antenna_positions.append(
            (lon, lat, elev)        )
        antenna_names.append(ant_name)
        antenna_diameters.append(float(cols[4]))

    f.close()
    print("Using antennas:", antenna_names)

#THIS PART OF THE CODE HAS BEEN REPLACED BY THE ABOVE.
#    line = f.readline()
#    while line:
#        cols = line.split()
#        lat=float(cols[2])
#        lon=float(cols[1])
#        elev=float(cols[3])
#        antenna_positions.append((lon,lat,elev)) 
#        antenna_names.append(cols[0])
#        antenna_diameters.append(float(cols[4])) # Diameter in metres
#        line = f.readline()
#    f.close()


    #--------------------------------------------------------------
    # Convert lon,lat to XYZ

    antennas_xyz = []
    ant_x = []
    ant_y = []
    ant_z = []

    for i, (lon, lat, elev) in enumerate(antenna_positions):
        X, Y, Z = wgs84_to_ecef(lon, lat, elev)
        ant_x.append(X)
        ant_y.append(Y)
        ant_z.append(Z)
        antennas_xyz.append((X, Y, Z))


    #--------------------------------------------------------------
    # Start the sim

    if os.path.exists(ms_name):
        if overwrite:
            print(f'Replacing existing {ms_name}')
            os.system(f'rm -rf {ms_name}')
        else:
            print(f'{ms_name} exists, will not overwrite')
            sys.exit()

    sm.open(ms = ms_name)

    # Antenna configuration

    if override_reference_location:
            array_center = reference_pos
    else:
        # Use MeerKAT as the reference location since CASA knows about it
        if telescopename == 'SKA-Mid' or telescopename == 'MeerKAT':
            # For SKA-Mid, we assume the MeerKAT configuration
            array_center = check_casa_observatory(me, 'MeerKAT')
        elif telescopename == 'e-MERLIN':
            array_center = check_casa_observatory(me, 'e-MERLIN')
        elif telescopename in [obs.upper for obs in me.obslist()]:
            #if its not emerlin or meerkat, but it is a known observatory 
            array_center = check_casa_observatory(me, telescopename)
        else:
            # If the telescope is not a known observatory, we use the geometric mean of the antenna positions
            array_mean = geometric_mean_antenna(antennas_xyz)
            array_center = me.position('ITRF', *array_mean)
    # Set the antenna configuration
    sm.setconfig(telescopename = telescopename,
                        x = ant_x,
                        y = ant_y,
                        z = ant_z,                 
                        dishdiameter = antenna_diameters,
                        mount = [mount_type],
                        antname = antenna_names,
                        coordsystem = 'global',
                        referencelocation = array_center)


    # Polarisation mode, assuming perfect linear feeds
    sm.setfeed(mode = 'perfect X Y', pol = [''])

    # Spectral window and associated pol config
    # Can be called repeatedly for multiple SPWs

    f0 =f"{f0}MHz" # lowest band frequency in MHz
    df = f"{df}MHz" # channel width in MHz
    #Define the 8 e-MERLIN spectral windows:
    #spw_names=[]
    #for i in range(nspw):
    #    spw_name= f"SPW{i}"
    #    spw_names.append(spw_name)
    #    #Centre frequency of the first 0.5 MHz channel in this SPW
    #    first_chan_freq=(lowest_band_edge+i*spw_bandwidth+df/2)
    #    print(f"{spw_name}: first channel = {first_chan_freq:.3f} MHz, "
    #    f"nchan={nchan}, channel width = {df:.3f} MHz")
    #    sm.setspwindow(spwname=spw_name,freq=f"{first_chan_freq} MHz", deltafreq= f"{df}MHz", freqresolution=f"{df}MHz", nchannels= nchan, stokes='XX YY')

#Define 1 spectral window use:
    sm.setspwindow(spwname = "SPW0",
                freq = f0,
                deltafreq = df,
                freqresolution = df,
                nchannels = nchan,
                stokes = 'XX YY')

    # Source / field info (i.e. phase centre definition)
    # Can also be called multiple times for multiple fields
    sm.setfield( sourcename = source_name,
                sourcedirection = me.direction(rf = 'J2000', v0 = source_ra,v1 = source_dec))

    # Flag based on shadowing or elevation limits
    sm.setlimits(shadowlimit = 0.01, elevationlimit = '12deg')
    #currently these are set somewhat arbitrarily. should be set to realistic values for the telescope in question

    # Ditch the autocorrelations
    sm.setauto(autocorrwt = 0.0);

    # Set the reference time and integration time

    #to set the reference time we need to know the telescope longitude
    tel_lon = array_center['m0']['value']
#APPLY A CHANGE HERE AS HOUR ANGLE POSITIONING IS HAPPENING TWICE. DATE SHOULD HAVE BEEN FOR EXAMPLE 2018-07-10, BUT INSTEAD CODE IS RUNNING 2018-06-10!!
   # obs_time = compute_obs_time(source_ra, start_ha, obs_date, tel_lon)
    obs_time= args.obs_time





    sm.settimes(integrationtime = t_int,
                usehourangle = True,
                referencetime = me.epoch('UTC',obs_time))
    #reference time refers to the time at which the source will transit the local meridian
    #usehourangle = True means that the start and end times are relative to the source transit time

    # Generate u,v,w tracks for all spectral windows:
#    for spw_name in spw_names:
#        print(f"Observing {spw_name}")
#        sm.observe(sourcename=source_name, spwname=spw_name, starttime=start_ha, stoptime=end_ha)
#    sm.close()
#GENERATE u,v,w tracks for only 1 spectral window:
    sm.observe(sourcename = source_name,
            spwname = 'SPW0',
            starttime = start_ha,
            stoptime = end_ha)

    sm.close()


    if should_update_yaml:
        yaml_path = os.path.join(CONFIGS,'default_prediction.yaml')
        update_yaml('ms_name',ms_name,yaml_path)
        update_yaml('telescopename',telescopename,yaml_path)

if __name__ == "__main__":
    main()
