from pint.observatory.nustar_obs import NuSTARObs
from pint.scripts.photonphase import main as photonphase
import glob
import os
import numpy as np
from astropy.table import Table
import pandas as pd
from astropy.time import Time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .utils import NUSTAR_MJDREF, splitext_improved, sec_to_mjd
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


def clock_ppm_model(time, temperature, T0=13.5, ppm_vs_T_pars=None,
                    ppm_vs_time_pars=None, old_version=False):
    """Improved clock model

    Original IDL function by Craig Markwardt:

    temperature0 = 13.5 ;; [degC]
    x = temperature at epoch ;; [degC]
    year = epoch time expressed in calendar years (2016.0 = Jan 1.0, 2016)
    p = [1.3930859254716697D+01,-7.3929902867262087D-02,1.4498257975709195D-03,-3.7891186658656098D-03,2.5104748381788913D-02,  1.9180710395375647D-01]
    function clock_ppm_model, x, p, year=year, temperature0=temp0
      temp = (x-temp0)
      ftemp = P[0] + P[1]*temp + P[2]*temp^2 ;; Temperature dependence
      t = (year-2012.5d)
      flongterm = P[3]*t - P[4]*exp(-t/P[5])
      return, ftemp + flongterm
    end

    Parameters
    ----------
    time : np.array of floats
        Times in MET
    temperature : np.array of floats
        TCXO Temperatures in C
    T0 : float
        Reference temperature
    ppm_vs_T_pars : list
        parameters of the ppm-temperature relation
    ppm_vs_T_pars : list
        parameters of the ppm-time relation (long-term clock decay)

    """
    if ppm_vs_T_pars is None:
        ppm_vs_T_pars = \
            [1.3930859254716697e+1, -7.3929902867262087e-2,
             1.4498257975709195e-3]
        if old_version:
            ppm_vs_T_pars = [13.965, -0.0733, 0]
            T0 = 13

    if ppm_vs_time_pars is None:
        ppm_vs_time_pars = \
            [-3.7891186658656098e-3,
             2.5104748381788913e-2, 1.9180710395375647e-1]
        # I want time in days
        ppm_vs_time_pars[0] /= 365.25
        ppm_vs_time_pars[2] *= 365.25
        if old_version:
            ppm_vs_time_pars = [0, 0, 1]

    temp = (temperature - T0)
    ftemp = ppm_vs_T_pars[0] + ppm_vs_T_pars[1] * temp + \
                ppm_vs_T_pars[2]*temp**2 # Temperature dependence

    mjd = sec_to_mjd(time)
    # year 2012.5 = MJD 56109.999994212965
    t = (mjd - 56109.999994212965)

    flongterm = \
        ppm_vs_time_pars[0]*t - ppm_vs_time_pars[1] * np.exp(-t / ppm_vs_time_pars[2])

    return ftemp + flongterm


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      ):
    table_times = temptable['met']
    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]
    temperature = temptable['temperature_smooth']

    temp_fun = interp1d(table_times, temperature,
                        fill_value='extrapolate', bounds_error=False)

    times_fine = np.arange(met_start, met_stop, 0.2)

    ppm_mod = clock_ppm_model(times_fine, temp_fun(times_fine))
    ppm_mod_old = clock_ppm_model(times_fine, temp_fun(times_fine),
                                  old_version=True)

    # ppm_mod = ppm0 - slope * (temp_fun(times_fine) - t0)
    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1
    clock_rate_corr_old = (1 + ppm_mod_old / 1000000) * 24000000 / divisor - 1

    delay = np.cumsum(np.diff(times_fine) * clock_rate_corr[:-1])
    delay_old = np.cumsum(np.diff(times_fine) * clock_rate_corr_old[:-1])

    import matplotlib.pyplot as plt
    plt.plot(times_fine[1:], delay, label="New")
    plt.plot(times_fine[1:], delay_old, label="Old")
    plt.legend()
    plt.show()
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
    allfreqtimes = np.array(freq_changes_table['met'])
    allfreqtimes = np.concatenate([allfreqtimes, [allfreqtimes[-1] + 86400]])

    met_intervals = list(
        zip(allfreqtimes[:-1], allfreqtimes[1:]))

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
    ext = splitext_improved(os.path.basename(events_file))[1]
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

