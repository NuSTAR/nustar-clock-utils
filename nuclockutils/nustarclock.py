from pint.observatory.nustar_obs import NuSTARObs
from pint.scripts.photonphase import main as photonphase
import glob
import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from nustar_tempcorr import read_clockfile, get_clock_correction
from nustar_tempcorr.utils import sec_to_mjd
import copy
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pint.models import get_model
from pint.event_toas import load_fits_TOAs
from .utils import NUSTAR_MJDREF, splitext_improved
from astropy.io import fits


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')
ALL_BAD_POINTS = np.genfromtxt(os.path.join(datadir, 'BAD_POINTS_DB.dat'),
                               dtype=np.longdouble)


def _look_for_temptable():
    """
    Look for the default temperature table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_temptable()
    >>> tempt.endswith('tp_eps_ceu_txco_tmp.csv')
    True
    """
    name = 'tp_eps_ceu_txco_tmp.csv'
    fullpath = os.path.join(datadir, name)
    assert os.path.exists(fullpath), \
        ("Temperature table not found. Have you run get_data.sh in "
         "the data directory?")

    return fullpath


def _look_for_clock_offset_file():
    """
    Look for the default clock offset table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_clock_offset_file()
    >>> os.path.basename(tempt).startswith('nustar_clock_offsets')
    True
    """
    name = 'nustar_clock_offsets*.dat'
    clockoff_files = sorted(glob.glob(os.path.join(datadir, name)))

    assert len(clockoff_files) > 0, \
        ("Clock offset file not found. Have you run get_data.sh in "
         "the data directory?")

    return clockoff_files[-1]


def _look_for_freq_change_file():
    """
    Look for the default frequency change table

    Examples
    --------
    >>> import os
    >>> tempt = _look_for_clock_offset_file()
    >>> os.path.basename(tempt).startswith('nustar_clock_offsets')
    True
    """
    name = 'nustar_freq_changes*.dat'
    fchange_files = sorted(glob.glob(os.path.join(datadir, name)))

    assert len(fchange_files) > 0, \
        ("Frequency change file not found. Have you run get_data.sh in "
         "the data directory?")

    return fchange_files[-1]


# def bary_eventlist(eventfile, orbfile, parfile, outfile):
#     if os.path.exists(outfile):
#         print(f'{outfile} exists, skipping')
#         return outfile
#     plotfile = outfile.replace('.evt', '') + '.png'
#     photonphase(
#         f'{eventfile} {parfile} --orbfile {orbfile} --absphase --barytime --outfile {outfile} --plot --plotfile {plotfile}'.split(
#             ' '))
#
#     return outfile


def read_clock_offset_table(clockoffset_file=None):
    """
    Parameters
    ----------
    clockoffset_file : str
        e.g. 'nustar_clock_offsets-2018-10-30.dat'

    Returns
    -------
    clock_offset_table : `astropy.table.Table` object
    """
    if clockoffset_file is None:
        clockoffset_file = _look_for_clock_offset_file()
    clock_offset_table = Table.read(clockoffset_file,
                                    format='csv', delimiter=' ',
                                    names=['uxt', 'met', 'offset', 'divisor',
                                           'station'])
    clock_offset_table['mjd'] = sec_to_mjd(clock_offset_table['met'])
    clock_offset_table.remove_row(len(clock_offset_table) - 1)
    clock_offset_table['flag'] = np.zeros(len(clock_offset_table), dtype=bool)

    for b in ALL_BAD_POINTS:
        nearest = np.argmin(np.abs(clock_offset_table['met'] - b))
        if np.abs(clock_offset_table['met'][nearest] - b) < 1:
            #             print(f"Removing clock offset at time {b}")
            clock_offset_table['flag'][nearest] = True

    return clock_offset_table


def read_freq_changes_table(freqchange_file=None, filter_bad=True):
    """
    Parameters
    ----------
    freqchange_file : str
        e.g. 'nustar_freq_changes-2018-10-30.dat'

    Returns
    -------
    freq_changes_table : `astropy.table.Table` object
    """
    if freqchange_file is None:
        freqchange_file = _look_for_freq_change_file()
    freq_changes_table = Table.read(freqchange_file,
                                format='csv', delimiter=' ',
                                names=['uxt', 'met', 'divisor'])
    freq_changes_table['mjd'] = sec_to_mjd(freq_changes_table['met'])
    freq_changes_table.remove_row(len(freq_changes_table) - 1)
    if filter_bad:
        freq_changes_table = \
            freq_changes_table[np.abs(freq_changes_table['divisor'] - 2.400034e7) < 20]
    return freq_changes_table


class OrbitalFunctions():
    lat_fun = None
    lon_fun = None
    alt_fun = None
    lst_fun = None


def get_orbital_functions(orbfile):
    from astropy.time import Time
    import astropy.units as u
    orbtable = Table.read(orbfile)
    # t = Time(100.0, format='mjd')
    mjdref = orbtable.meta['MJDREFF'] + orbtable.meta['MJDREFI']
    # print(mjdref)
    # print(orbtable['TIME'] / 86400 + mjdref)
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


def _filter_table(tablefile, start_date=None, end_date=None, tmpfile='tmp.csv'):
    try:
        from datetime import timezone
    except ImportError:
        # Python 2
        import pytz as timezone
    if start_date is None:
        start_date = 0
    if end_date is None:
        end_date = 99999

    start_date = Time(start_date, format='mjd', scale='utc')
    start_str = start_date.to_datetime(timezone=timezone.utc).strftime('%Y:%j')

    start_yr, start_day = [float(n) for n in start_str.split(':')]

    end_date = Time(end_date, format='mjd', scale='utc')
    stop_str = end_date.to_datetime(timezone=timezone.utc).strftime('%Y:%j')
    new_str = ""
    with open(tablefile) as fobj:
        before = True
        for i, l in enumerate(fobj.readlines()):
            if i == 0:
                new_str += l
                continue
            l = l.strip()
            # Now, let's check if the start date is before the start of the
            # clock file. It's sufficient to do it for the first 2-3 line(s)
            # (3 if there are units)
            if i <=2:
                try:
                    yr, day = [float(n) for n in l.split(':')[:2]]
                except ValueError:
                    continue
                if start_yr <= yr and start_day <= day:
                    before = False
            if l.startswith(start_str) and before is True:
                before = False
            if before is False:
                new_str += l + "\n"
            if l.startswith(stop_str):
                break

    with open(tmpfile, "w") as fobj:
        print(new_str, file=fobj)

    return tmpfile


def read_csv_temptable(mjdstart=None, mjdstop=None, temperature_file=None):
    if temperature_file is None:
        temperature_file = _look_for_temptable()

    if mjdstart is not None or mjdstop is not None:
        print("Filtering table...")
        tmpfile = _filter_table(temperature_file,
                                start_date=mjdstart - 10,
                                end_date=mjdstop + 10, tmpfile='tmp.csv')
        print("Done")
    else:
        tmpfile = temperature_file

    temptable = Table.read(tmpfile)
    temptable.remove_row(0)
    times_mjd = Time(temptable["Time"], scale='utc', format="yday",
                     in_subfmt="date_hms").mjd
    temptable["mjd"] = np.array(times_mjd)
    temptable['met'] = (temptable["mjd"] - NUSTAR_MJDREF) * 86400
    temptable.remove_column('Time')
    temptable.rename_column('tp_eps_ceu_txco_tmp', 'temperature')
    temptable['temperature_smooth'] = \
        savgol_filter(temptable['temperature'], 11, 3)
    return temptable


def read_temptable(temperature_file=None, mjdstart=None, mjdstop=None):
    if temperature_file is None or temperature_file.endswith('.csv'):
        return read_csv_temptable(mjdstart, mjdstop,
                                  temperature_file)


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      ppm0=13.965, slope=0.0733, t0=13):
    table_times = temptable['met'],
    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]
    temperature = temptable['temperature_smooth']

    temp_fun = interp1d(table_times, temperature,
                        fill_value='extrapolate', bounds_error=False)

    times_fine = np.arange(met_start, met_stop, 0.2)

    ppm_mod = ppm0 - slope * (temp_fun(times_fine) - t0)
    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1

    delay = np.cumsum(np.diff(times_fine) * clock_rate_corr[:-1])

    return interp1d(times_fine[:-1], delay, fill_value='extrapolate',
                    bounds_error=False)


def calculate_temperature_correction(met_start, met_stop,
                                     temperature_file=None,
                                     adjust=None):
    mjdstart, mjdstop = sec_to_mjd(met_start), sec_to_mjd(met_stop)
    temptable = read_temptable(mjdstart=mjdstart - 5,
                               mjdstop=mjdstop + 5,
                               temperature_file=temperature_file)
    freq_changes_table = read_freq_changes_table()
    met_intervals = list(
        zip(freq_changes_table['met'][:-1], freq_changes_table['met'][1:]))
    met_intervals = np.concatenate([met_intervals, met_intervals[-1] + 86400])
    divisors = freq_changes_table['divisor']
    last_corr = 0
    last_time = met_intervals[0][0]

    data = pd.DataFrame()

    for i, met_intv in enumerate(met_intervals):
        if met_intv[1] < met_start:
            continue
        if met_intv[0] > met_stop:
            break
        start, stop = met_intv
        good_temps = (temptable['met'] >= start - 20) & (
                      temptable['met'] <= stop + 20)

        temptable_filt = temptable[good_temps]

        if len(temptable_filt) < 10:
            print(
                f"Too few temperature points in interval "
                f"{start} to {stop} (MET)")
            continue

        delay_function = \
            temperature_delay(temptable_filt, divisors[i])

        times_fine = np.arange(start, stop, 0.5)

        temp_corr = \
            delay_function(times_fine) + last_corr - delay_function(last_time)

        new_table = pd.DataFrame(dict(met=times_fine,
                                      temp_corr=temp_corr))
        data = data.append(new_table, ignore_index=True)
        last_corr = temp_corr[-1]
        last_time = times_fine[-1]

    if adjust:
        data = adjust_temperature_correction(data)
    return data


def adjust_temperature_correction(data):
    from scipy.optimize import curve_fit
    times = np.array(data['met'])
    start = times[0]
    stop = times[-1]

    clock_offset_table = read_clock_offset_table()
    # Find relevant clock offsets
    good_times = \
        (clock_offset_table['met'] >= start) & (
                    clock_offset_table['met'] <= stop)
    clock_offset_table = clock_offset_table[good_times]
    no_flag = ~clock_offset_table['flag']
    clock_offset_table = clock_offset_table[no_flag]

    cltimes = clock_offset_table['met']
    offsets = clock_offset_table['offset']

    first_clock_idx = np.argmin(np.abs(times - cltimes[0]))

    data['temp_corr_rough'] = np.array(data['temp_corr'])

    data['temp_corr'] = \
        data['temp_corr_rough'] - data['temp_corr_rough'][first_clock_idx] + \
        offsets[0]

    fun = interp1d(data['met'], data['temp_corr'])

    def fit_function(times, m, q):
        return fun(times) - ((times - cltimes[0]) * m + q)

    popt, pcov = curve_fit(fit_function, cltimes,
                           clock_offset_table['offset'])

    m, q = popt

    data['temp_corr'] = data['temp_corr'] - (m * (times - cltimes[0]) + q)
    return data


def temperature_correction_fun(met_start, met_stop):
    data = calculate_temperature_correction(met_start, met_stop, adjust=True)

    return interp1d(data['met'], data['temp_corr'],
                    fill_value="extrapolate", bounds_error=False)


def create_clockfile(met_start, met_stop):
    data = calculate_temperature_correction(met_start, met_stop, adjust=True)
    pass


def apply_clock_correction(events_file, outfile=None):
    ext = splitext_improved(events_file)[1]
    if outfile is None:
        outfile = events_file.replace(ext, "_tc" + ext)
    if outfile == events_file:
        raise ValueError("outfile == events_file")

    with fits.open(events_file, memmap=False) as hdul:
        event_times = hdul[1].data['TIME']
        start, stop = event_times[0], event_times[-1]
        corr_fun = temperature_correction_fun(start - 2*86400,
                                              stop + 2*86400)
        hdul[1].data['TIME'] = event_times - corr_fun(event_times)
        hdul.writeto(outfile, overwrite=True)
    return outfile


def main_tempcorr(args=None):
    import argparse
    description = ('Apply experimental temperature correction to NuSTAR'
                   'event files')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name (default <inputfname>_tc.evt)")

    args = parser.parse_args(args)

    apply_clock_correction(args.file, outfile=args.outfile)

