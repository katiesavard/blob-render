from casatools import simulator, measures, table
from datetime import datetime

import numpy as np
import os
import sys

sm = simulator()
me = measures()


def wgs84_to_ecef(lat_deg, lon_deg, height_m):
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


#--------------------------------------------------------------
# Observational setup
# Use caution when testing as realistic dump times and channel widths
# will generate an SKA-Mid-sized Measurement Set

ms_name = 'my_ska-mid_sim.ms'
overwrite = True # Remove any existing MS with the same name

# Sky direction (J2000)
source_name = 'BLOB'
source_ra = '12h00m00.0s'
source_dec = '-40d00m0.0s'

# Frequency setup
f0 = '856MHz' # lowest band frequency
df = '107MHz' # channel width
nchan = 8 # number of channels

# Start time and track length
now_utc = datetime.utcnow()
obs_time = now_utc.strftime("%Y/%-m/%-d/%H:%M:%S")
t_int = '120s' # Note that this is the correlator dump time, not total track length
start_ha = '-0.2h' # Start and end times are relative to transit
end_ha = '+0.2h'


#--------------------------------------------------------------
# Read the antenna names and coordinates (lon,lat)
# This could be readily modified to select only names that begin with 'M'
# (for a MeerKAT-only sim), or 'S' to simulate an array using only the 
# SKA-Mid dishes. Default is to read all.

f = open('SKA-Mid_antenna_layout.txt')
antenna_positions = []
antenna_names = []
antenna_diameters = []
line = f.readline()
while line:
    cols = line.split()
    antenna_positions.append((float(cols[1]),float(cols[2]),1000.0)) # Assuming 1km above sea level
    antenna_names.append(cols[0])
    if cols[0][0] == 'M':
        antenna_diameters.append(float(13.5))
    elif cols[0][0] == 'S':
        antenna_diameters.append(float(15.0))
    line = f.readline()
f.close()


#--------------------------------------------------------------
# Convert lon,lat to XYZ

antennas_xyz = []
ant_x = []
ant_y = []
ant_z = []
for i, (lon, lat, elev) in enumerate(antenna_positions):
    X, Y, Z = wgs84_to_ecef(lat, lon, elev)
    ant_x.append(X)
    ant_y.append(Y)
    ant_z.append(Z)


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
# Use MeerKAT as the reference location since CASA knows about it
sm.setconfig(telescopename = 'MeerKAT',
                 x = ant_x,
                 y = ant_y,
                 z = ant_z,                 
                 dishdiameter = antenna_diameters,
                 mount = ['alt-az'],
                 antname = antenna_names,
                 coordsystem = 'global',
                 referencelocation = me.observatory('MeerKAT'))

# Polarisation mode, assuming perfect linear feeds
sm.setfeed(mode = 'perfect X Y', pol = [''])

# Spectral window and associated pol config
# Can be called repeatedly for multiple SPWs
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

# Ditch the autocorrelations
sm.setauto(autocorrwt = 0.0);

# Set the reference time and integration time
sm.settimes(integrationtime = t_int,
            usehourangle = True,
            referencetime = me.epoch('UTC',obs_time))

# Generate u,v,w tracks
sm.observe(sourcename = source_name,
           spwname = 'SPW0',
           starttime = start_ha,
           stoptime = end_ha)

sm.close()

