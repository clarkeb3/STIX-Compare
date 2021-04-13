from datetime import datetime
from stixcore.ephemeris.manager import Position
import stixcore.data.test
from pathlib import Path
import sunpy
import sys
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.map import Map
from sunpy.coordinates import get_body_heliographic_stonyhurst
from sunpy.coordinates import frames
from astropy.wcs import WCS
from pathlib import Path
from reproject import reproject_interp
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a


# Getting AIA Data

aia = (a.Instrument.aia &
       a.Sample(24 * u.hour) &
       a.Time('2020-10-01', '2020-10-02'))
wave = a.Wavelength(19.5 * u.nm, 19.5 * u.nm)
res = Fido.search(wave, aia)
files = Fido.fetch(res)

# Converting to sunpy map

aia171  = sunpy.map.Map(files)

# Using STIXCORE POISITION to find coords of SOLAR ORBTER on 2020-10-01

from pathlib import Path
mkp = Path(stixcore.ephemeris.manager.__file__).parent.parent / 'data' / 'test' / 'ephemeris' / 'test_position_20201001_V01.mk'
with Position(meta_kernel_path=mkp) as pos:
    p = pos.get_position(date=datetime(2020, 10, 1), frame='SOLO_HEE')
print(p)

# Converting return of STIXCORE POISITION to HeliocentricEarthEcliptic frame

solo_hee = SkyCoord(*p, frame=frames.HeliocentricEarthEcliptic, representation_type='cartesian', obstime="2020-10-01")

# Converting HeliocentricEarthEcliptic coords of SOLAR ORBTER position to HeliographicStonyhurst frame

solo_hee = solo_hee.transform_to(frames.HeliographicStonyhurst)

# Creating reference frame (AIA as seen by SOLAR ORBITER)

solo_hee_ref_coord = SkyCoord(aia171.reference_coordinate.Tx,aia171.reference_coordinate.Ty,obstime=aia171.reference_coordinate.obstime,
	observer=solo_hee,frame="helioprojective")

# Creating wcs header for SOLAR ORBITER frame

out_shape = (4096, 4096)
solo_header = sunpy.map.make_fitswcs_header(
    out_shape,
    solo_hee_ref_coord,
    scale=u.Quantity(aia171.scale),
    rotation_matrix=aia171.rotation_matrix,
    instrument="AIA",
    wavelength=aia171.wavelength
)

solo_wcs = WCS(solo_header)

# Performing reprojection to generate map of AIA as seen by Solar Orbiter

output, footprint = reproject_interp(aia171, solo_wcs, out_shape)
outmap = sunpy.map.Map((output, solo_header))
outmap.plot_settings = aia171.plot_settings


# Plotting maps

fig1 = plt.figure(figsize=(8,8))


ax1 = fig1.add_subplot(1, 2, 1, projection=aia171)
aia171.plot(axes=ax1)
outmap.draw_grid(color='w')

ax2 = fig1.add_subplot(1, 2, 2, projection=outmap)
outmap.plot(axes=ax2, title='AIA observation as seen from Solar Orbiter')
outmap.draw_grid(color='w')

fig1.tight_layout()





# Plotting positions of the Sun, Solar Oribter, and SDO

date = '2020-10-01'
solo_coords = solo_hee
aia_coords = aia171.observer_coordinate
sun = get_body_heliographic_stonyhurst("sun",  date)

fig2 = plt.figure(figsize=(8,8))

ax1 = fig2.add_subplot(projection="polar")
ax1.plot(solo_coords.lon.to(u.rad),
         solo_coords.radius.to(u.AU),
        'o', ms=12,
        label="SOLO")
ax1.plot(aia_coords.lon.to(u.rad),
         aia_coords.radius.to(u.AU),
        'o', ms=12,
        label="AIA")
ax1.plot(sun.lon.to(u.rad),
         sun.radius.to(u.AU),
        'o', ms=12,
        label="Sun")
plt.legend()

fig2.tight_layout()


plt.show()
