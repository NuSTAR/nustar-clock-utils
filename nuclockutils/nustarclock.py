import glob
import os
import shutil

import numpy as np
from astropy.table import Table
import pandas as pd
from astropy.time import Time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .utils import NUSTAR_MJDREF, splitext_improved, sec_to_mjd
from astropy.io import fits
import tqdm
from astropy import log


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
    >>> tempt = _look_for_temptable() # doctest: +ELLIPSIS
    ...
    >>> tempt.endswith('tp_eps_ceu_txco_tmp.csv')
    True
    """
    name = 'tp_eps_ceu_txco_tmp.csv'
    fullpath = os.path.join(datadir, name)

    if not os.path.exists(fullpath):
        import shutil
        import subprocess as sp
        sp.check_call('wget --no-check-certificate https://www.dropbox.com/s/spkn4v018m5fvkf/tp_eps_ceu_txco_tmp.csv?dl=0 -O bu.csv'.split(" "))

        shutil.copyfile('bu.csv', fullpath)
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
    log.info(f"Reading clock offsets from {clockoffset_file}")
    clock_offset_table = Table.read(clockoffset_file,
                                    format='csv', delimiter=' ',
                                    names=['uxt', 'met', 'offset', 'divisor',
                                           'station'])
    clock_offset_table['mjd'] = sec_to_mjd(clock_offset_table['met'])
    clock_offset_table.remove_row(len(clock_offset_table) - 1)
    clock_offset_table['flag'] = np.zeros(len(clock_offset_table), dtype=bool)

    log.info("Flagging bad points...")
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
    log.info(f"Reading frequency changes from {freqchange_file}")
    freq_changes_table = Table.read(freqchange_file,
                                format='csv', delimiter=' ',
                                names=['uxt', 'met', 'divisor'])
    freq_changes_table['mjd'] = sec_to_mjd(freq_changes_table['met'])
    freq_changes_table.remove_row(len(freq_changes_table) - 1)
    freq_changes_table['flag'] = \
        np.abs(freq_changes_table['divisor'] - 2.400034e7) > 20
    if filter_bad:
        freq_changes_table = freq_changes_table[~freq_changes_table['flag']]
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

    if new_str == "":
        raise ValueError(f"No temperature information is available for the "
                         "wanted time range in {temperature_file}")
    with open(tmpfile, "w") as fobj:
        print(new_str, file=fobj)

    return tmpfile


def read_csv_temptable(mjdstart=None, mjdstop=None, temperature_file=None):
    if mjdstart is not None or mjdstop is not None:
        log.info("Filtering table...")
        tmpfile = _filter_table(temperature_file,
                                start_date=mjdstart - 10,
                                end_date=mjdstop + 10, tmpfile='tmp.csv')
        log.info("Done")
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
    return temptable


def read_saved_temptable(mjdstart=None, mjdstop=None,
                         temperature_file='temptable.hdf5'):
    table = Table.read(temperature_file)
    if mjdstart is None and mjdstop is None:
        return table

    if 'mjd' not in table.colnames:
        table["mjd"] = sec_to_mjd(table['met'])

    if mjdstart is None:
        mjdstart = table['mjd'][0]
    if mjdstop is None:
        mjdstop = table['mjd'][-1]

    good = (table['mjd'] >= mjdstart - 10)&(table['mjd'] <= mjdstop + 10)
    if not np.any(good):
        raise ValueError(f"No temperature information is available for the "
                         "wanted time range in {temperature_file}")
    return table[good]


def read_fits_temptable(temperature_file):
    with fits.open(temperature_file) as hdul:
        temptable = Table.read(hdul['ENG_0x133'])
        temptable.rename_column('TIME', 'met')
        temptable.rename_column('sc_clock_ext_tmp', 'temperature')
        for col in temptable.colnames:
            if 'chu' in col:
                temptable.remove_column(col)
        temptable["mjd"] = sec_to_mjd(temptable['met'])
    return temptable


def interpolate_temptable(temptable, dt=10):
    time = temptable['met']
    temperature = temptable['temperature']
    new_times = np.arange(time[0], time[-1], dt)
    idxs = np.searchsorted(time, new_times)
    return Table({'met': new_times, 'temperature': temperature[idxs]})


def read_temptable(temperature_file=None, mjdstart=None, mjdstop=None,
                   dt=None):
    if temperature_file is None:
        temperature_file = _look_for_temptable()
    log.info(f"Reading temperature_information from {temperature_file}")
    ext = splitext_improved(temperature_file)[1]
    if ext in ['.csv']:
        temptable = read_csv_temptable(mjdstart, mjdstop, temperature_file)
    elif ext in ['.hk', '.hk.gz']:
        temptable = read_fits_temptable(temperature_file)
    elif ext in ['.hdf5', '.h5']:
        return read_saved_temptable(mjdstart, mjdstop,
                                    temperature_file)
    else:
        raise ValueError('Unknown format for temperature file')

    if dt is not None:
        temptable = interpolate_temptable(temptable, dt)
    else:
        good = np.diff(temptable['met']) > 0
        good = np.concatenate((good, [True]))
        temptable = temptable[good]

    temptable['temperature_smooth'] = \
        savgol_filter(temptable['temperature'], 11, 3)

    return temptable


def abs_des_fun(x, b0, b1, b2, b3, t0=77509250):
    """An absorption-desorption function"""
    x = (x - t0) / 86400. / 365.25
    return b0 * np.log(b1 * x + 1) + b2 * np.log(b3 * x + 1)


def clock_ppm_model(nustar_met, temperature):
    """Improved clock model

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
    T0 = 13.5
    offset = 13.9158325193 - 0.027918
    ppm_vs_T_pars = [-0.073795, 0.0015002]
    ppm_vs_time_pars = [0.008276, 256., -220.043,
                        3.408586903702425e-05]

    temp = (temperature - T0)
    ftemp = offset + ppm_vs_T_pars[0] * temp + \
            ppm_vs_T_pars[1] * temp ** 2  # Temperature dependence

    flongterm = abs_des_fun(nustar_met, *ppm_vs_time_pars)

    return ftemp + flongterm


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      debug=False):
    # temptable = self.temptable
    table_times = temptable['met']

    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]

    temperature = temptable['temperature_smooth']

    temp_fun = interp1d(table_times, temperature,
                        fill_value='extrapolate', bounds_error=False,
                        assume_sorted=True)

    times_fine = np.arange(met_start, met_stop, 0.2)

    ppm_mod = clock_ppm_model(times_fine, temp_fun(times_fine))

    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1

    delay = np.cumsum(np.diff(times_fine) * clock_rate_corr[:-1])

    return interp1d(times_fine[:-1], delay, fill_value='extrapolate',
                    bounds_error=False)


class ClockCorrection():
    def __init__(self, temperature_file, mjdstart=None, mjdstop=None,
                 temperature_dt=10, adjust_absolute_timing=False,
                 force_divisor=None, label="", additional_days=2,
                 clock_offset_table=None,
                 hdf_dump_file='dumped_data.hdf5'):
        self.temperature_file = temperature_file
        self.mjdstart = mjdstart - additional_days / 2
        self.mjdstop = mjdstop + additional_days / 2

        if mjdstart is not None:
            self.met_start = (mjdstart - NUSTAR_MJDREF) * 86400
            self.met_stop = (mjdstop - NUSTAR_MJDREF) * 86400
        self.temperature_dt = temperature_dt
        self.temptable = None

        self.read_temptable()
        self.force_divisor = force_divisor
        self.adjust_absolute_timing = adjust_absolute_timing

        self.hdf_dump_file = hdf_dump_file
        self.plot_file = label + "_clock_adjustment.png"
        self.clock_offset_table = clock_offset_table
        self.correct_met = \
            self.temperature_correction_fun(adjust=self.adjust_absolute_timing)
        if label is None:
            label = f"{self.met_start}-{self.met_stop}"

    def read_temptable(self):
        self.temptable = read_temptable(temperature_file=self.temperature_file,
                                        mjdstart=self.mjdstart,
                                        mjdstop=self.mjdstop,
                                        dt=self.temperature_dt)

    def temperature_correction_fun(self, adjust=False):
        data = \
            temperature_correction_table(
                self.met_start, self.met_stop,
                force_divisor=self.force_divisor,
                temptable = self.temptable,
                hdf_dump_file=self.hdf_dump_file)
        if adjust:
            log.info("Adjusting temperature correction")
            data = self.adjust_temperature_correction(data)
        return interp1d(np.array(data['met']), np.array(data['temp_corr']),
                        fill_value="extrapolate", bounds_error=False)

    def adjust_temperature_correction(self, data):
        import matplotlib.pyplot as plt
        times = np.array(data['met'])
        start = times[0]
        stop = times[-1]

        clock_offset_table_all = \
            read_clock_offset_table(self.clock_offset_table)
        # Find relevant clock offsets
        good_times = \
            (clock_offset_table_all['met'] >= start) & (
                    clock_offset_table_all['met'] <= stop)
        clock_offset_table_all = clock_offset_table_all[good_times]

        no_flag = ~clock_offset_table_all['flag']
        good_station = clock_offset_table_all['station'] == 'MLD'
        N = len(clock_offset_table_all)
        clock_offset_table = clock_offset_table_all[no_flag & good_station]
        n = len(clock_offset_table)
        log.info(f"Found {n}/{N} valid clock offset measurements "
                 f"between MET {int(start)}--{int(stop)}")

        cltimes = clock_offset_table['met']
        offsets = clock_offset_table['offset']

        first_clock_idx = np.argmin(np.abs(times - cltimes[0]))

        data['temp_corr_rough'] = np.array(data['temp_corr'])

        data['temp_corr'] = \
            data['temp_corr_rough'] - data['temp_corr_rough'][first_clock_idx] + \
            offsets[0]

        fun = interp1d(data['met'], data['temp_corr'], bounds_error=False,
                       fill_value='extrapolate')

        fit_result = robust_linear_fit(cltimes, fun(cltimes) - offsets)
        m, q = fit_result.estimator_.coef_, fit_result.estimator_.intercept_

        inlier_mask = fit_result.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        def fit_function(times, m, q):
            return fun(times) - (times * m + q)

        if self.plot_file is not None:
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 1, hspace=0, height_ratios=(3, 2))
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)
            fine_times = np.arange(start, stop, 100)
            ax0.scatter(clock_offset_table_all['met'],
                        clock_offset_table_all['offset'] * 1000, alpha=0.5,
                        color='k',
                        label="Discarded offsets (bad)")
            ax0.scatter(cltimes, offsets * 1000, label="Offsets used for fit")
            if np.any(outlier_mask):
                ax0.scatter(cltimes[outlier_mask],
                            offsets[outlier_mask] * 1000,
                            color='r', marker='x', s=30, zorder=10,
                            label="Fit outliers")
            ax0.plot(fine_times, fun(fine_times) * 1000, alpha=0.5)
            new_tcorr = fit_function(fine_times, m, q)
            ax0.plot(fine_times, new_tcorr * 1000)
            ax0.set_ylabel("Offset (ms)")
            ax0.grid()

            ax1.scatter(clock_offset_table_all['met'],
                        1e6 * (clock_offset_table_all['offset'] - fit_function(
                            clock_offset_table_all['met'], m, q)),
                        color='k', alpha=0.5)
            ax1.scatter(cltimes, 1e6 * (offsets - fit_function(cltimes, m, q)))
            if np.any(outlier_mask):
                ax1.scatter(cltimes[outlier_mask],
                            1e6 * (offsets - fit_function(cltimes, m, q))[
                                outlier_mask],
                            color='r', marker='x', s=30, zorder=10,
                            label="Fit outliers")

            ax1.axhline(0)
            ax1.set_xlabel("MET (s)")
            ax1.set_ylabel("Residual (us)")
            ax1.set_ylim((-500, 500))
            ax1.grid()

            plt.savefig(self.plot_file)
            plt.close(fig)

        log.info(f"Correcting for a drift of {m} s/s")

        data['temp_corr'] = data['temp_corr'] - (m * times + q)
        return data


def temperature_correction_table(met_start, met_stop,
                                 temptable=None,
                                 freqchange_file=None,
                                 hdf_dump_file='dump.hdf5',
                                 force_divisor=None,
                                 time_resolution=0.5):
    import six
    if hdf_dump_file is not None and os.path.exists(hdf_dump_file):
        log.info(f"Reading cached data from file {hdf_dump_file}")
        result_table = pd.read_hdf(hdf_dump_file, key='tempdata')
        mets = np.array(result_table['met'])
        if (met_start > mets[10] or met_stop < mets[-20]) and (
                met_stop - met_start < 3 * 365 * 86400):
            log.warning(
                "Interval not fully included in cached data. Recalculating.")
        else:
            good = (mets >= met_start - 86400) & (mets < met_stop + 86400)
            return result_table[good]

    if temptable is None or isinstance(temptable, six.string_types):
        mjdstart, mjdstop = sec_to_mjd(met_start), sec_to_mjd(met_stop)
        temptable = read_temptable(mjdstart=mjdstart,
                                   mjdstop=mjdstop,
                                   temperature_file=temptable)
    if force_divisor is None:
        freq_changes_table = \
            read_freq_changes_table(freqchange_file=freqchange_file)
        allfreqtimes = np.array(freq_changes_table['met'])
        allfreqtimes = \
            np.concatenate([allfreqtimes, [allfreqtimes[-1] + 86400]])
        met_intervals = list(
            zip(allfreqtimes[:-1], allfreqtimes[1:]))
        divisors = freq_changes_table['divisor']
    else:
        met_intervals = [[met_start - 10, met_stop + 10]]
        divisors = [force_divisor]

    last_corr = 0
    last_time = met_intervals[0][0]
    N = int((met_stop - met_start) / time_resolution + 50000000)

    log.info(f"Allocating temperature table with {N} entries...")
    table = Table(names=['met', 'temp_corr', 'divisor'],
                  data=np.zeros((N, 3))
                  )
    log.info("Done")

    firstidx = 0
    for i, met_intv in tqdm.tqdm(enumerate(met_intervals),
                                 total=len(met_intervals)):
        if met_intv[1] < met_start:
            continue
        if met_intv[0] > met_stop:
            break

        start, stop = met_intv
        log.info(f"Calculating temperature correction between "
                 f"MET {start:.1f}--{stop:.1f}")

        tempidx = np.searchsorted(temptable['met'], [start - 20, stop + 20])

        temptable_filt = temptable[tempidx[0]:tempidx[1] + 1]

        if len(temptable_filt) < 10:
            log.warning(
                f"Too few temperature points in interval "
                f"{start} to {stop} (MET)")
            continue

        delay_function = \
            temperature_delay(temptable_filt, divisors[i])

        times_fine = np.arange(start, stop, time_resolution)

        temp_corr = \
            delay_function(times_fine) + last_corr - delay_function(last_time)

        new_data = Table(dict(met=times_fine,
                              temp_corr=temp_corr,
                              divisor=np.zeros_like(times_fine) + divisors[i]))

        if np.any(temp_corr != temp_corr):
            log.error("Invalid data in temperature table")
            break

        N = len(times_fine)
        table[firstidx:firstidx + N] = new_data
        firstidx = firstidx + N

        last_corr = temp_corr[-1]
        last_time = times_fine[-1]

    log.info("Interpolation done.")
    table = table[:firstidx]

    data = table.to_pandas()

    if hdf_dump_file is not None:
        log.info(f"Saving intermediate data to {hdf_dump_file}...")
        data.to_hdf(hdf_dump_file, key='tempdata')
        log.info(f"Done.")
    return data




def create_clockfile(met_start, met_stop):
    data = temperature_correction_table(met_start, met_stop, adjust=True)
    pass


def robust_linear_fit(x, y):
    from sklearn import linear_model
    X = x.reshape(-1, 1)
    # # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor(residual_threshold=0.0002)
    ransac.fit(X, y)
    return ransac


class NuSTARCorr():
    def __init__(self, events_file, outfile=None,
                 adjust=True, force_divisor=None,
                 temperature_file=None, hdf_dump_file=None):
        self.events_file = events_file
        root, ext = splitext_improved(os.path.basename(events_file))
        if outfile is None:
            outfile = events_file.replace(ext, "_tc" + ext)
        if outfile == events_file:
            raise ValueError("outfile == events_file")
        self.outfile = outfile
        self.adjust = adjust
        self.force_divisor = force_divisor
        self.temperature_file = temperature_file
        self.tstart = None
        self.tstop = None
        self.correction_start = None
        self.correction_stop = None
        self.read_observation_info()
        mjdstart, mjdstop = \
            sec_to_mjd([self.tstart, self.tstop])
        self.clock_correction = ClockCorrection(temperature_file,
                                                mjdstart=mjdstart,
                                                mjdstop=mjdstop,
                                                temperature_dt=10,
                                                adjust_absolute_timing=adjust,
                                                force_divisor=force_divisor,
                                                label=root, additional_days=2,
                                                clock_offset_table=None,
                                                hdf_dump_file=hdf_dump_file)
        self.temperature_correction_fun = \
            self.clock_correction.correct_met

    def read_observation_info(self):
        with fits.open(self.events_file) as hdul:
            hdr = hdul[1].header
            self.tstart, self.tstop = hdr['TSTART'], hdr['TSTOP']

    def apply_clock_correction(self):
        events_file = self.events_file
        outfile = self.outfile

        log.info(f"Opening {events_file}")

        shutil.copyfile(events_file, outfile)

        with fits.open(outfile) as hdul:
            event_times = hdul[1].data['TIME']
            log.info(f"Calculating temperature correction")
            corr_fun = \
                self.temperature_correction_fun
            hdul[1].data['TIME'] = event_times - corr_fun(event_times)
            hdul.writeto(outfile, overwrite=True)

        return outfile


def main_tempcorr(args=None):
    import argparse
    description = ('Apply experimental temperature correction to NuSTAR'
                   'event files. For very recent ToO observations not covered'
                   'by the temperature history file or the frequency change '
                   'file, use the -t, --no-adjust and -D options.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name (default <inputfname>_tc.evt)")
    parser.add_argument("-t", "--tempfile", default=None,
                        help="Temperature file (e.g. the nu<OBSID>_eng.hk.gz "
                             "file in the auxil/directory "
                             "or the tp_tcxo*.csv file)")
    parser.add_argument("--cache", default=None,
                        help="HDF5 dump file used as cache (ext. hdf5)")
    parser.add_argument("--no-adjust",
                        help="Do not adjust using tabulated clock offsets",
                        action='store_true', default=False)
    parser.add_argument("-D", "--force-divisor", default=None, type=float,
                        help="Force frequency divisor to this value. Typical "
                             "values are around 24000330")
    args = parser.parse_args(args)

    observation = NuSTARCorr(args.file, outfile=args.outfile,
                             adjust=not args.no_adjust,
                             force_divisor=args.force_divisor,
                             temperature_file=args.tempfile,
                             hdf_dump_file=args.cache)
    outfile = observation.apply_clock_correction()
    return outfile

