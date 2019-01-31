from astropy.table import Table
import numpy as np
from scipy.interpolate import interp1d
from .utils import filter_with_region


class OrbitalFunctions():
    lat_fun = None
    lon_fun = None
    alt_fun = None
    lst_fun = None


def get_orbital_functions(orbfile):
    from astropy.time import Time
    import astropy.units as u
    orbtable = Table.read(orbfile)
    mjdref = orbtable.meta['MJDREFF'] + orbtable.meta['MJDREFI']

    times = Time(np.array(orbtable['TIME'] / 86400 + mjdref), format='mjd')
    if 'GEODETIC' in orbtable.colnames:
        geod = np.array(orbtable['GEODETIC'])
        lat, lon, alt = geod[:, 0] * u.deg, geod[:, 1] * u.deg, geod[:,
                                                                2] * u.m
    else:
        geod = np.array(orbtable['POLAR'])
        lat, lon, alt = (geod[:, 0] * u.rad).to(u.deg), (
                    geod[:, 1] * u.rad).to(u.deg), geod[:, 2] * 1000 * u.m

    lat_fun = interp1d(times.mjd, lat, bounds_error=False,
                       fill_value='extrapolate')
    lon_fun = interp1d(times.mjd, lon, bounds_error=False,
                       fill_value='extrapolate')
    alt_fun = interp1d(times.mjd, alt, bounds_error=False,
                       fill_value='extrapolate')
    gst = times.sidereal_time('apparent', 'greenwich')
    lst = lon.to(u.hourangle) + gst.to(u.hourangle)
    lst[lst.value > 24] -= 24 * u.hourangle
    lst[lst.value < 0] += 24 * u.hourangle
    lst_fun = interp1d(times.mjd, lst, bounds_error=False,
                       fill_value='extrapolate')

    orbfunc = OrbitalFunctions()
    orbfunc.lat_fun = lat_fun
    orbfunc.lon_fun = lon_fun
    orbfunc.alt_fun = alt_fun
    orbfunc.lst_fun = lst_fun

    return orbfunc


def barycorr(evfile, orbfile, parfile, outfile=None,
             overwrite=False):
    from astropy.io import fits
    from pint.observatory.nustar_obs import NuSTARObs
    from pint.event_toas import load_fits_TOAs
    from pint.models import get_model
    from pint.toa import get_TOAs_list
    from shutil import copyfile
    from .utils import splitext_improved, NUSTAR_MJDREF, sec_to_mjd
    from astropy.time import Time
    import os
    import warnings

    warnings.warn("At the moment, GTIs are not barycentered. "
                  "Also, TSTART, TSTOP etc. might be affected by leap seconds")
    if outfile is None:
        ext = splitext_improved(evfile)[1]
        outfile = evfile.replace(ext, '_bary' + ext)

    if os.path.exists(outfile) and not overwrite:
        raise RuntimeError('Output file exists')

    NuSTARObs(name='NuSTAR',FPorbname=orbfile, tt2tdb_mode='pint')
    tl = load_fits_TOAs(evfile, mission='nustar', timeref='LOCAL')

    # Read in model
    modelin = get_model(parfile)

    ts = get_TOAs_list(tl, include_bipm=False,
        include_gps=False, planets=False, tdb_method='default',
                       ephem='DE421')
    ts.filename = orbfile
    mjds = modelin.get_barycentric_toas(ts)
    copyfile(evfile, outfile)

    with fits.open(outfile, memmap=True) as hdul:
        mets = (mjds.value - NUSTAR_MJDREF) * 86400

        uncorr_mets = hdul[1].data['TIME']
        start_corr = mets[0] - uncorr_mets[0]
        stop_corr = mets[-1] - uncorr_mets[-1]

        start = hdul[1].header['TSTART'] + start_corr
        start = Time(sec_to_mjd(start), format='mjd')

        stop = hdul[1].header['TSTOP'] + stop_corr
        stop = Time(sec_to_mjd(stop), format='mjd')

        hdul[1].data['TIME'] = mets
        version = '0.0dev'
        for hdu in hdul:
            hdu.header['CREATOR'] = f'NuSTAR Clock Utils - v. {version}'
            hdu.header['DATE'] = Time.now().fits
            hdu.header['DATE-END'] = stop.fits
            hdu.header['DATE-OBS'] = start.fits
            hdu.header['PLEPHEM'] = 'JPL-DE421'
            hdu.header['RADECSYS'] = 'FK5'
            hdu.header['TIMEREF'] = 'SOLARSYSTEM'
            hdu.header['TIMESYS'] = 'TDB'
            hdu.header['TIMEZERO'] = 0.0
            hdu.header['TSTART'] = (start.mjd - NUSTAR_MJDREF) * 86400
            hdu.header['TSTOP'] = (stop.mjd - NUSTAR_MJDREF) * 86400
            hdu.header['TREFDIR'] = 'RA_OBJ,DEC_OBJ'
            hdu.header['TREFPOS'] = 'BARYCENTER'

        hdul.writeto(outfile, overwrite=True)

    return outfile


def main_barycorr(args=None):
    import argparse
    description = ('Apply the barycenter correction to NuSTAR'
                   'event files')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("orbitfile", help="Orbit file")
    parser.add_argument("parfile", help="Parameter file in TEMPO/TEMPO2 "
                                        "format (for precise coordinates)")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name (default <inputfname>_tc.evt)")
    parser.add_argument("--overwrite",
                        help="Overwrite existing data",
                        action='store_true', default=False)

    parser.add_argument("-r", "--region", default=None, type=str,
                        help="Filter with ds9-compatible region file. MUST be"
                             " a circular region in the FK5 frame")

    args = parser.parse_args(args)

    if args.region is not None:
        args.file = filter_with_region(args.file)

    outfile = \
        barycorr(args.file, args.orbitfile, args.parfile, outfile=args.outfile,
                 overwrite=args.overwrite)

    return outfile