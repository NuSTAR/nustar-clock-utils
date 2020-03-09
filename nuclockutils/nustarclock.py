import glob
import os
import shutil

import numpy as np
from astropy.table import Table, vstack
import pandas as pd
from astropy.time import Time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from .utils import NUSTAR_MJDREF, splitext_improved, sec_to_mjd
from .utils import filter_with_region
from astropy.io import fits
import tqdm
from astropy import log
from statsmodels.robust import mad
import copy
import holoviews as hv
from holoviews.operation.datashader import datashade
from holoviews import opts

hv.extension('bokeh')

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')
ALL_BAD_POINTS = np.genfromtxt(os.path.join(datadir, 'BAD_POINTS_DB.dat'),
                               dtype=np.longdouble)


def get_rolling_std(clock_residuals_detrend,
                    clock_offset_table, window=20 * 86400):
    malindi_stn = clock_offset_table['station'] == 'MLD'
    detrended_not_nan = clock_residuals_detrend == clock_residuals_detrend
    use_for_interpol = \
        malindi_stn & ~clock_offset_table['flag'] & detrended_not_nan
    overlap = 0.1

    clstart = clock_offset_table['met'][0]
    clstop = clock_offset_table['met'][-1]

    control_points = np.arange(clstart, clstop + window, window * overlap)
    rolling_std = np.zeros_like(control_points)

    times_to_search = clock_offset_table['met'][use_for_interpol]
    res_to_search = clock_residuals_detrend[use_for_interpol]

    for i, t in enumerate(control_points):
        s, e = t - window / 2, t + window / 2
        s_idx, e_idx = np.searchsorted(times_to_search, s), \
                       np.searchsorted(times_to_search, e)
        rolling_std[i] = mad(res_to_search[s_idx:e_idx])
    return control_points, rolling_std



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


FREQ_CHANGE_DB= {"delete": [77509247, 78720802],
                 "add": [(1023848462, 77506869, 24000340),
                         (1025060017, 78709124, 24000337),
                         (0, 102576709, 24000339),
                         (1051488464, 105149249, 24000336),
                         (0, 182421890, 24000334),
                         (1157021125, 210681910, 24000337),
                         ( 0, 215657278, 24000333),
                         (0, 215794126, 24000328),
                         (1174597307, 228258092, 24000334),
                         (1174759273, 228420058, 24000334)]}


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
                                comment="\s*#",
                                names=['uxt', 'met', 'divisor'])
    log.info("Correcting known bad frequency points")
    for time in FREQ_CHANGE_DB['delete']:
        bad_time_idx = freq_changes_table['met'] == time
        freq_changes_table[bad_time_idx] = [0, 0, 0]
    for line in FREQ_CHANGE_DB['add']:
        freq_changes_table.add_row(line)
    freq_changes_table = freq_changes_table[freq_changes_table['met'] > 0]
    freq_changes_table.sort('met')

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
        mjdstart_use = mjdstart
        mjdstop_use = mjdstop
        if mjdstart is not None:
            mjdstart_use -= 10
        if mjdstop is not None:
            mjdstop_use += 10
        log.info("Filtering table...")
        tmpfile = _filter_table(temperature_file,
                                start_date=mjdstart_use,
                                end_date=mjdstop_use, tmpfile='tmp.csv')
        log.info("Done")
    else:
        tmpfile = temperature_file

    temptable = Table.read(tmpfile)
    temptable.remove_row(0)
    log.info("Converting times (it'll take a while)...")
    times_mjd = Time(temptable["Time"], scale='utc', format="yday",
                     in_subfmt="date_hms").mjd
    log.info("Done.")
    temptable["mjd"] = np.array(times_mjd)
    temptable['met'] = (temptable["mjd"] - NUSTAR_MJDREF) * 86400
    temptable.remove_column('Time')
    temptable.rename_column('tp_eps_ceu_txco_tmp', 'temperature')
    temptable["temperature"] = np.array(temptable["temperature"], dtype=float)
    if os.path.exists('tmp.csv'):
        os.unlink('tmp.csv')

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

    window = np.median(310 / np.diff(temptable['met']))
    window = int(window // 2 * 2 + 1)
    temptable['temperature_smooth'] = \
        savgol_filter(temptable['temperature'], window, 2)

    return temptable


class ClockCorrection():
    def __init__(self, temperature_file, mjdstart=None, mjdstop=None,
                 temperature_dt=10, adjust_absolute_timing=False,
                 force_divisor=None, label="", additional_days=2,
                 clock_offset_file=None,
                 hdf_dump_file='dumped_data.hdf5',
                 freqchange_file=None,
                 spline_through_residuals=False):
        # hdf_dump_file_adj = hdf_dump_file.replace('.hdf5', '') + '_adj.hdf5'
        self.temperature_dt = temperature_dt
        self.temperature_file = temperature_file
        self.freqchange_file = freqchange_file

        # Initial value. it will be changed in the next steps
        self.mjdstart = mjdstart
        self.mjdstop = mjdstop

        self.read_temptable()

        if mjdstart is None:
            mjdstart = sec_to_mjd(self.temptable['met'].min())
        else:
            mjdstart = mjdstart - additional_days / 2

        if mjdstop is None:
            mjdstop = sec_to_mjd(self.temptable['met'].max())

            mjdstop = mjdstop + additional_days / 2

        self.mjdstart = mjdstart
        self.mjdstop = mjdstop

        self.met_start = (self.mjdstart - NUSTAR_MJDREF) * 86400
        self.met_stop = (self.mjdstop - NUSTAR_MJDREF) * 86400

        if label is None or label == "":
            label = f"{self.met_start}-{self.met_stop}"

        self.force_divisor = force_divisor
        self.adjust_absolute_timing = adjust_absolute_timing

        self.hdf_dump_file = hdf_dump_file
        self.plot_file = label + "_clock_adjustment.png"
        self.clock_offset_file = clock_offset_file
        self.clock_offset_table = \
            read_clock_offset_table(self.clock_offset_file)

        self.temperature_correction_data = \
            temperature_correction_table(
                self.met_start, self.met_stop,
                force_divisor=self.force_divisor,
                time_resolution=10,
                temptable = self.temptable,
                hdf_dump_file=self.hdf_dump_file,
                freqchange_file=self.freqchange_file)

        if adjust_absolute_timing:
            log.info("Adjusting temperature correction")
            try:
                self.adjust_temperature_correction()
            except Exception:
                import traceback
                logfile = 'adjust_temperature_error.log'
                log.warn("Temperature adjustment failed. "
                         "Full error stack in {logfile}")
                with open(logfile, 'w') as fobj:
                    traceback.print_last(file=logfile)

    def read_temptable(self, cache_temptable_name=None):
        if cache_temptable_name is not None and \
                os.path.exists(cache_temptable_name):
            self.temptable = Table.read(cache_temptable_name, path='temptable')
        else:
            self.temptable = \
                    read_temptable(temperature_file=self.temperature_file,
                                   mjdstart=self.mjdstart,
                                   mjdstop=self.mjdstop,
                                   dt=self.temperature_dt)
            if cache_temptable_name is not None:
                self.temptable.write(cache_temptable_name, path='temptable')

    def temperature_correction_fun(self, adjust=False):
        data = self.temperature_correction_data

        return interp1d(np.array(data['met']), np.array(data['temp_corr']),
                        fill_value="extrapolate", bounds_error=False)

    def adjust_temperature_correction(self):
        import copy
        table = self.temperature_correction_data
        table_new = copy.deepcopy(table)
        mets = np.array(table_new['met'])

        start = mets[0]
        stop = mets[-1]
        clock_offset_table = read_clock_offset_table(self.clock_offset_file)
        clock_mets = clock_offset_table['met']

        jump_times = np.array([78708320, 79657575, 81043985, 82055671])
        jump_times = jump_times[(jump_times >= start) & (jump_times < stop)]
        jumpidx = np.searchsorted(table_new['met'], jump_times)
        cljumpidx = np.searchsorted(clock_offset_table['met'], jump_times)

        for clidx, idx in zip(cljumpidx, jumpidx):
            jump = clock_offset_table['offset'][clidx + 1] - \
                   clock_offset_table['offset'][clidx]
            table_new['temp_corr'][:idx] -= jump  # - corrjump

        good = table_new['temp_corr'] == table_new['temp_corr']
        table_new = table_new[good]

        tempcorr_idx = np.searchsorted(table_new['met'],
                                       clock_offset_table['met'])

        clock_residuals = clock_offset_table['offset'] - \
                          table_new['temp_corr'][tempcorr_idx]

        good = clock_residuals == clock_residuals
        if len(clock_offset_table['met']) > 100:
            z = np.polyfit(clock_offset_table['met'][good],
                           clock_residuals[good], 1)
            p = np.poly1d(z)
        else:
            fit_result = robust_linear_fit(clock_offset_table['met'][good],
                                           clock_residuals[good])
            m, q = fit_result.estimator_.coef_, \
                   fit_result.estimator_.intercept_

            def p(times):
                return times * m + q

        print(f'df/f = {(p(stop) - p(start)) / (stop - start)}')

        table_new['temp_corr'] += p(table_new['met'])

        clock_residuals = clock_offset_table['offset'] - \
                          table_new['temp_corr'][tempcorr_idx]

        smooth = median_filter(clock_residuals, 21)
        diffs = clock_residuals - smooth
        surelybad = np.logical_or(diffs < -50, diffs > 80)
        no_malindi_intvs = [[93681591, 98051312]]
        bad_malindi_time = np.zeros(len(clock_mets), dtype=bool)
        for nmi in no_malindi_intvs:
            bad_malindi_time = bad_malindi_time | (clock_mets >= nmi[0]) & (
                        clock_mets < nmi[1])

        # For bad Malindi Times, we use clock offsets minus half a millisecond
        clock_offset_table[bad_malindi_time]['offset'] -= 0.0005

        malindi_stn = clock_offset_table['station'] == 'MLD'
        use_for_interpol = \
            (malindi_stn | bad_malindi_time) & ~clock_offset_table[
                'flag'] & ~surelybad

        good_clock_mets = clock_mets[use_for_interpol]
        good_diffs = clock_residuals[use_for_interpol]

        # running_median = median_filter(good_diffs, 11)
        # better_points = np.abs(good_diffs - running_median) < 0.001
        mscuts = [-0.002, -0.001, -0.0006, -0.0004]

        better_points = np.array(good_diffs == good_diffs, dtype=bool)
        good_diffs[~better_points] = 0

        for i, cut in enumerate(mscuts):
            mm = median_filter(good_diffs, 15)
            wh = ((good_diffs[better_points] - mm[better_points]) < mscuts[
                i]) | ((good_diffs[better_points] - mm[better_points]) <
                       mscuts[0])
            better_points[better_points] = ~wh

        diff_trend = interp1d(good_clock_mets[better_points],
                              savgol_filter(good_diffs[better_points], 11, 2),
                              fill_value='extrapolate',
                              bounds_error=False, assume_sorted=True)

        table_new['temp_corr_detrend'] = \
            table_new['temp_corr'] + diff_trend(table_new['met'])

        table_new['temp_corr'] = table_new['temp_corr_detrend']
        table_new.pop('temp_corr_detrend')

        self.temperature_correction_data = table_new

    def write_clock_file(self, filename=None):
        from astropy.io.fits import Header, HDUList, BinTableHDU, PrimaryHDU
        from astropy.time import Time

        if filename is None:
            filename = 'new_clock_file.fits'
        clock_offset_table = self.clock_offset_table
        table_new = self.temperature_correction_data

        tempcorr_idx = np.searchsorted(table_new['met'],
                                       clock_offset_table['met'])

        clock_residuals_detrend = clock_offset_table['offset'] - \
                                  table_new['temp_corr'][tempcorr_idx]

        control_points, rolling_std = get_rolling_std(clock_residuals_detrend,
                                                      clock_offset_table)

        clock_err_fun = interp1d(control_points, rolling_std, assume_sorted=True,
                                 bounds_error=False, fill_value='extrapolate')
        new_clock_table = Table({'TIME': table_new['met'],
                                 'CLOCK_OFF_CORR': -table_new['temp_corr'],
                                 'CLOCK_FREQ_CORR': np.gradient(
                                     -table_new['temp_corr'],
                                     table_new['met'], edge_order=2),
                                 'CLOCK_ERR_CORR': clock_err_fun(
                                     table_new['met'])})

        new_clock_table_subsample = new_clock_table[::100]
        t = Time.now()
        t.format = 'iso'
        t.out_subfmt = 'date'

        date = t.value
        clock_mets = clock_offset_table['met']
        clstart = clock_mets.min()
        clstop = clock_mets.max()
        tstart = clstart
        tstop = clstop

        header = Header()
        header["XTENSION"] = ('BINTABLE', "Written by IDL")
        header["BITPIX"] = (8, "NAXIS   =                    2 /Binary table")
        header["TFIELDS"] = (4, "Number of columns")
        header["EXTNAME"] = ('NU_FINE_CLOCK', "NuSTAR fine clock correction")
        header["DATE"] = (date, "Creation date")
        header["ORIGIN"] = ('Caltech ', "Source of FITS file")
        header["TELESCOP"] = ('NuSTAR  ', "Telescope (mission) name")
        header["INSTRUME"] = (
        'FPM     ', "Instrument name (FPM=mission-level)")
        header["LONGSTRN"] = (
        'OGIP 1.0', "The OGIP long string convention may be used.")
        header["TFORM1"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE1"] = ('TIME    ', "Epoch of spline control point")
        header["TUNIT1"] = ('s       ', "Units of column 1")
        header["TFORM2"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE2"] = ('CLOCK_OFF_CORR', "Clock correction at epoch")
        header["TUNIT2"] = ('s       ', "Units of column 2")
        header["TFORM3"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE3"] = (
        'CLOCK_FREQ_CORR', "Clock freq correction [fractional]")
        header["TFORM4"] = ('1D      ', "Real*8 (double precision)")
        header["TTYPE4"] = ('CLOCK_ERR_CORR', "Approx. 1-sigma error of model")
        header["TUNIT4"] = ('s       ', "Units of column 4")
        header[
            'COMMENT'] = "This FITS file may contain long string keyword values that are"
        header[
            'COMMENT'] = "continued over multiple keywords.  This convention uses the  '&'"
        header[
            'COMMENT'] = "character at the end of a string which is then continued"
        header['COMMENT'] = "on subsequent keywords whose name = 'CONTINUE'."
        header["RADECSYS"] = ('FK5     ', "celestial coord system")
        header["TIMEUNIT"] = ('s       ', "Units of header time keywords")
        header["TASSIGN"] = ('SATELLITE', "Time assigned by onboard clock")
        header["TIMESYS"] = ('TT      ', "time measured from")
        header["MJDREFI"] = (55197, "MJD reference day")
        header["MJDREFF"] = (
        0.000766018520000, "MJD reference (fraction of day)")
        header["TIMEREF"] = ('LOCAL   ', "reference time")
        header["VERSION"] = (87, "Extension version number")
        header["CONTENT"] = ('NuSTAR Fine Clock Correction', "File content")
        header["CDES0001"] = (
        'NUSTAR FINE CLOCK CORRECTION FILE', "Description")
        header["CCLS0001"] = (
        'BCF     ', "Dataset is a Basic Calibration File")
        header["CDTP0001"] = ('DATA    ', "Calibration file contains data")
        header["CCNM0001"] = ('FINECLOCK', "Type of calibration data")

        header["CVSD0001"] = ('2010-01-01', "UCT date when file should first be used")
        header["CVST0001"] = ('00:00:00', "UCT time when file should first be used")
        
        header["CVTD0001"] = (date, "Last UTC date when the file should be used (same as DATE)")
        header["CVTT0001"] = (date, "Last UTC time when the file should be used")
        
        header["TSTART"] = (tstart, "[s] Start time [MET] of data in the offset file")
        header["TSTOP"] = (tstop, "[s] Stop time [MET] of data in the offset file")

        comment = """NuSTAR fine clock correction file

        This extension contains NuSTAR 'fine' clock correction, which improves
        the NuSTAR clock performance into the 10s of microsecond range.

        During a transition period both kinds of extensions, the standard and
        the fine clock correction, may appear in the same FITS file.
        Use the standard correction, or the fine correction, but not both.
        The task 'barycorr' version 2.2 and higher will automatically choose
        this extension in preference over the standard correction, if it is
        present.

        The extension contains the parameters of a cubic spline model.
        The spline model represents short term clock changes driven by
        the thermal environment of the NuSTAR observatory's clock, as
        well as longer term stochastic drifts.

        The spline is sampled irregularly depending on changes in the
        clock performance and thermal environment.

        Each TIME point in the table is considered a cublic spline
        control point.  At the control point, the value of the spline
        and its derivative are specified.  The correspond to the
        clock *offset* (CLOCK_OFFSET, in seconds) and *frequency*
        (CLOCK_FRACT_FREQ, expressed as a fractional quantity, or
        seconds per second).

        For a given epoch, t, locate the bracketing TIME samples
        in the table.  Use cubic interpolation with the given
        tabulated values to find the clock offset (and optionally
        frequency) at epoch.

        The resulting quantity is the clock correction value.  Thus,
        this value should be *added* from observatory timestamps
        recorded at that epoch.
        """

        for comm in comment.split("\n"):
            header['COMMENT'] = comm

        prihdu = PrimaryHDU()
        prihdu.header['DATE'] = date
        prihdu.header['TELESCOP'] = ('NuSTAR', 'Telescope (mission) name')
        prihdu.header['INSTRUME'] = \
            ('FPM', 'Instrument name (FPM=mission-level)')
        prihdu.header['ORIGIN'] = ('Caltech', 'Source of FITS file')
        prihdu.header['COMMENT'] = \
            "This FITS file may contain long string keyword values that are"
        prihdu.header['COMMENT'] = \
            "continued over multiple keywords.  This convention uses the  '&'"
        prihdu.header['COMMENT'] = \
            "character at the end of a string which is then continued"
        prihdu.header['COMMENT'] = \
            "on subsequent keywords whose name = 'CONTINUE'."

        hdu = BinTableHDU(new_clock_table_subsample, header=header)

        HDUList([prihdu, hdu]).writeto(filename, overwrite=True)

    def adjust_temperature_correction_old(self):
        import matplotlib.pyplot as plt
        data = self.temperature_correction_data
        times = np.array(data['met'])
        start = times[0]
        stop = times[-1]

        clock_offset_table_all, mask = self.load_offset_table(start, stop)
        clock_offset_table = clock_offset_table_all[mask]

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
        self.temperature_correction_data = data
        return data


def calculate_clock_function(new_clock_table, clock_offset_table):
    tab_times = new_clock_table['TIME']
    clock_mets = clock_offset_table['met']
    good_mets = (clock_mets > tab_times.min()) & (clock_mets < tab_times.max())
    clock_offset_table = copy.deepcopy(clock_offset_table[good_mets])
    clock_mets = clock_offset_table['met']
    tab_idxs = np.searchsorted(tab_times, clock_mets, side='right') - 1

    x = clock_mets[:-1]
    xtab = tab_times[tab_idxs]
    ytab = new_clock_table['CLOCK_OFF_CORR'][tab_idxs]
    yptab = new_clock_table['CLOCK_FREQ_CORR'][tab_idxs]

    xtab = [xtab[:-1], xtab[1:]]
    ytab = [ytab[:-1], ytab[1:]]
    yptab = [yptab[:-1], yptab[1:]]

    dx = x - xtab[0];
    #     /* Distance between adjoining tabulated abcissae and ordinates */
    xs = xtab[1] - xtab[0];
    ys = ytab[1] - ytab[0];

    #     /* Rescale or pull out quantities of interest */
    dx = dx / xs  # ;             /* Rescale DX */
    y0 = ytab[0]  # ;           /* No rescaling of Y - start of interval */
    yp0 = yptab[
        0]  # ;      /* Rescale tabulated derivatives - start of interval */
    yp1 = yptab[
              1] * xs  # ;      /* Rescale tabulated derivatives - end of interval */

    #     /* Compute polynomial coefficients */
    a = y0;
    b = yp0;
    c = 3 * ys - 2 * yp0 - yp1;
    d = yp0 + yp1 - 2 * ys;

    #     /* Perform cubic interpolation */
    yint = -a + dx * (b + dx * (c + dx * d))
    return yint, good_mets


def plot_scatter(new_clock_table, clock_offset_table):
    from bokeh.models import HoverTool
    yint, good_mets = calculate_clock_function(new_clock_table,
                                               clock_offset_table)
    clock_offset_table = clock_offset_table[good_mets]
    clock_mets = clock_offset_table['met']
    clock_mjds = clock_offset_table['mjd']
    clock_residuals_detrend = clock_offset_table['offset'][:-1] - yint

    control_points, rolling_std = \
        get_rolling_std(clock_residuals_detrend, clock_offset_table[:-1],
                        window=5 * 86400)

    dates = Time(clock_mjds[:-1], format='mjd')

    all_data = pd.DataFrame({'met': clock_mets[:-1],
                             'mjd': np.array(clock_mjds[:-1], dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'offset': clock_offset_table['offset'][:-1],
                             'station': clock_offset_table['station'][:-1]})
    all_data = hv.Dataset(all_data, [('met', 'Mission Epoch Time'),
                                     ('station', 'Ground Station')],
                                    [('offset', 'Clock Offset (s)'),
                                     ('mjd', 'MJD'),
                                     ('doy', 'DOY'),
                                     ('utc', 'UT')])

    tooltips = [
        ('MET', '@met'),
        ('MJD', '@mjd'),
        ('DOY', '@doy'),
        ('UT', '@utc'),
    ]
    hover = HoverTool(tooltips=tooltips)

    plot_0 = all_data.to.scatter('met', ['offset', 'mjd', 'doy', 'utc'],
                                 groupby='station').options(
        color_index='station', alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station')
    plot_0a = hv.Curve(dict(x=clock_mets[:-1], y=yint))
    plot_0_all = plot_0.opts(opts.Scatter(width=900, height=350, tools=[hover])).opts(
                             ylim=(-0.1, 0.8)) * plot_0a

    all_data_res = pd.DataFrame({'met': clock_mets[:-1],
                             'mjd': np.array(clock_mjds[:-1], dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'residual': clock_residuals_detrend * 1e6,
                             'station': clock_offset_table['station'][:-1]})
                             
                             
    # Jump in and save this to disk here:
    all_data_res.to_pickle('all_data_res.pkl')
    
    all_data_res = hv.Dataset(all_data_res, [('met', 'Mission Epoch Time'),
                                     ('station', 'Ground Station')],
                                    [('residual', 'Residuals (us)'),
                                     ('mjd', 'MJD'),
                                     ('doy', 'DOY'),
                                     ('utc', 'UT')])
    plot_1 = all_data_res.to.scatter('met', ['residual', 'mjd', 'doy', 'utc'],
                                 groupby='station').options(
        color_index='station', alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station')
    plot_1b = hv.Curve({'x': control_points, 'y': rolling_std * 1e6}).opts(
        opts.Curve(color='k'))
    plot_1a = hv.Curve({'x': control_points, 'y': -rolling_std * 1e6}).opts(
        opts.Curve(color='k'))

    plot_1_all = plot_1.opts(
        opts.Scatter(width=900, height=350, tools=[hover])).opts(
                             ylim=(-700, 700)) * plot_1b * plot_1a

    rolling_data = pd.DataFrame({'met':control_points, 'rolling_std':rolling_std*1e6})
    rolling_data.to_pickle('rolling_data.pkl')

    return hv.Layout(plot_0_all + plot_1_all).cols(1)


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
                                                clock_offset_file=None,
                                                hdf_dump_file=hdf_dump_file)
        self.temperature_correction_fun = \
            self.clock_correction.temperature_correction_fun(adjust=adjust)

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


def simpcumquad(x, y):
    n = len(x)

    if (n != len(y)):
        raise ValueError('ERROR: X and Y array dimensions must match')
    if n == 0:
        raise ValueError('ERROR: X not defined')
    if n == 1:
        return x * 0

    # Find the step size between samples - assume equally sampled
    h = x[1] - x[0]

    # Trapezoid rule for two points
    if n == 2:
        return h * (y[0] + y[1]) / 2

    ii = np.arange((n - 1) // 2, dtype=int) * 2 + 1

    # initialize the area array to zero
    area = np.zeros_like(y)

    area[ii] = 1.25 * y[ii - 1] + 2.0 * y[ii] - 0.25 * y[ii + 1]
    area[ii + 1] = -0.25 * y[ii - 1] + 2.0 * y[ii] + 1.25 * y[ii + 1]

    if n % 2 == 0:
        area[n - 1] = -0.25 * y[n - 3] + 2.0 * y[n - 2] + 1.25 * y[n - 1]

    area1 = (h / 3) * np.cumsum(area)

    return area1


def abs_des_fun(x, b0, b1, b2, b3, t0=77509250):
    """An absorption-desorption function"""
    x = (x - t0) / 86400. / 365.25
    return b0 * np.log(b1 * x + 1) + b2 * np.log(b3 * x + 1)


def clock_ppm_model(nustar_met, temperature, craig_fit=False):
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
    #     offset = 13.9158325193 - 0.027918 - 4.608765729063114e-4 -7.463444052344004e-9
    offset = 13.8874536353 - 4.095179312091239e-4
    ppm_vs_T_pars = [-0.073795, 0.0015002]
    ppm_vs_time_pars = [0.008276, 256., -220.043,
                        3.408586903702425e-05]
    if craig_fit:
        offset = 1.3847529679329989e+01
        ppm_vs_T_pars = [-7.3964896025586133e-02, 1.5055740907563737e-03]

    temp = (temperature - T0)
    ftemp = offset + ppm_vs_T_pars[0] * temp + \
            ppm_vs_T_pars[1] * temp ** 2  # Temperature dependence

    flongterm = abs_des_fun(nustar_met, *ppm_vs_time_pars)

    return ftemp + flongterm


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      debug=False, craig_fit=False,
                      time_resolution=10):
    from scipy.integrate import cumtrapz
    table_times = temptable['met']

    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]

    temperature = temptable['temperature_smooth']

    temp_fun = interp1d(table_times, temperature,
                        assume_sorted=True)

    dt = time_resolution
    times_fine = np.arange(met_start, met_stop, dt)

    try:
        ppm_mod = clock_ppm_model(times_fine, temp_fun(times_fine),
                                  craig_fit=craig_fit)
    except:
        print(times_fine.min(), times_fine.max())
        print(table_times.min(), table_times.max())
        raise

    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1

    # delay = cumtrapz(clock_rate_corr, times_fine, initial=0)
    delay_sim = simpcumquad(times_fine, clock_rate_corr)
    #     print(np.max(np.abs(delay - delay_sim)))
    #     delay = np.cumsum(dt * clock_rate_corr)
    return interp1d(times_fine, delay_sim, fill_value='extrapolate',
                    bounds_error=False)


def temperature_correction_table(met_start, met_stop,
                                 temptable=None,
                                 freqchange_file=None,
                                 hdf_dump_file='dump.hdf5',
                                 force_divisor=None,
                                 time_resolution=0.5,
                                 craig_fit=False):
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
    log.info(f"Calculating temperature correction between "
             f"MET {met_start:.1f}--{met_stop:.1f}")

    for i, met_intv in tqdm.tqdm(enumerate(met_intervals),
                                 total=len(met_intervals)):
        if met_intv[1] < met_start:
            continue
        if met_intv[0] > met_stop:
            break

        start, stop = met_intv

        tempidx = np.searchsorted(temptable['met'], [start - 20, stop + 20])

        temptable_filt = temptable[tempidx[0]:tempidx[1] + 1]

        times_fine = np.arange(start, stop, time_resolution)

        if len(temptable_filt) < 5:
            log.warning(
                f"Too few temperature points in interval "
                f"{start} to {stop} (MET)")
            continue

        delay_function = \
            temperature_delay(temptable_filt, divisors[i], craig_fit=craig_fit,
                              time_resolution=time_resolution)

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
    parser.add_argument("-r", "--region", default=None, type=str,
                        help="Filter with ds9-compatible region file. MUST be"
                             " a circular region in the FK5 frame")

    args = parser.parse_args(args)

    if args.region is not None:
        args.file = filter_with_region(args.file)

    observation = NuSTARCorr(args.file, outfile=args.outfile,
                             adjust=not args.no_adjust,
                             force_divisor=args.force_divisor,
                             temperature_file=args.tempfile,
                             hdf_dump_file=args.cache)

    outfile = observation.apply_clock_correction()
    return outfile


def main_create_clockfile(args=None):
    import argparse
    description = ('Calculate experimental clock file for NuSTAR, using a '
                   'temperature-driven correction for the onboard TCXO.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("tempfile", type=str,
                        help="Temperature file (e.g. tp_tcxo*.csv)")
    parser.add_argument("offsets", type=str,
                        help="Table of clock offsets")
    parser.add_argument("frequency_changes", type=str,
                        help="Table of divisor changes")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output file name")
    parser.add_argument("--cache", default=None,
                        help="HDF5 dump file used as cache (ext. hdf5)")

    args = parser.parse_args(args)

    clockcorr = ClockCorrection(temperature_file=args.tempfile,
                                adjust_absolute_timing=True,
                                clock_offset_file=args.offsets,
                                hdf_dump_file=args.cache,
                                freqchange_file=args.frequency_changes)
    clockcorr.write_clock_file(args.outfile)

    plot = plot_scatter(Table.read(args.outfile, hdu="NU_FINE_CLOCK"),
                        clockcorr.clock_offset_table)
    from bokeh.io import output_file, save, show
    outfig = args.outfile.replace(".gz", "").replace(".fits", "")
    renderer = hv.renderer('bokeh')
    renderer.save(plot, outfig)


def main_update_temptable(args=None):
    import argparse
    description = ('Calculate experimental clock file for NuSTAR, using a '
                   'temperature-driven correction for the onboard TCXO.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("tempfile", type=str,
                        help="Temperature file (e.g. tp_tcxo*.csv)")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Output HDF5 file name")

    args = parser.parse_args(args)

    last_measurement = None
    existing_table = None
    if os.path.exists(args.outfile):
        log.info("Reading existing temperature table")
        existing_table = Table.read(args.outfile)
        last_measurement = existing_table['mjd'][-1] + 0.001 / 86400

    log.info("Reading new temperature values")
    new_table = read_csv_temptable(temperature_file=args.tempfile,
                                   mjdstart=last_measurement)

    if last_measurement is not None:
        new_values = new_table['mjd'] > last_measurement
        if not np.any(new_values):
            log.info("Temperature table is up to date")
            return

        log.info("Updating temperature table")
        new_table = vstack((existing_table, new_table[new_values]))

    window = np.median(310 / np.diff(new_table['met']))
    window = int(window // 2 * 2 + 1)
    new_table['temperature_smooth'] = \
        savgol_filter(new_table['temperature'], window, 2)

    log.info(f"Saving to {args.outfile}")
    new_table.write(args.outfile, path="temptable", overwrite=True)


def main_plot_diagnostics(args=None):
    import argparse
    description = ('Plot diagnostic information about the newly produced '
                   'clock file.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("clockcorr", type=str,
                        help="Clock correction file")
    parser.add_argument("clockoff", default=None,
                        help="Clock offset table")

    args = parser.parse_args(args)

    clock_offset_table = read_clock_offset_table(args.clockoff)
    plot = plot_scatter(Table.read(args.clockcorr, hdu="NU_FINE_CLOCK"),
                        clock_offset_table)

    outfig = args.clockcorr.replace(".gz", "").replace(".fits", "")
    renderer = hv.renderer('bokeh')
    renderer.save(plot, outfig)
