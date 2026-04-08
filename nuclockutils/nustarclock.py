"""
NuSTAR Clock Correction Module
==============================

This module provides tools for correcting the NuSTAR spacecraft clock drift
using temperature-dependent models. The onboard TCXO (Temperature Compensated
Crystal Oscillator) exhibits predictable drift correlated with temperature,
which this module models and corrects.

Data Flow Overview
------------------
1. **Temperature data** is read from housekeeping files (CSV, HDF5, or FITS)
2. **Clock offset measurements** from ground station passes are loaded
3. **Frequency divisor changes** (commanded clock adjustments) are loaded
4. Temperature-dependent clock drift is modeled using `clock_ppm_model`
5. Residuals between model and measurements are detrended with splines
6. Final correction is written as a CALDB-compatible FITS clock file

Primary Table Schemas
---------------------
**clock_offset_table** (from ground station measurements):
    - ``uxt``: Unix timestamp of measurement
    - ``met``: Mission Elapsed Time (seconds since MJDREF)
    - ``offset``: Measured clock offset from ground truth (seconds)
    - ``divisor``: Clock divisor value at measurement time
    - ``station``: Ground station code (e.g. 'MLD'=Malindi, 'SNG'=Singapore, 'UHI'=Hawaii)
    - ``mjd``: Modified Julian Date (computed)
    - ``flag``: Boolean, True if measurement is flagged as bad

**temptable** (temperature measurements):
    - ``met``: Mission Elapsed Time (seconds)
    - ``mjd``: Modified Julian Date
    - ``temperature``: Raw TCXO temperature (Celsius)
    - ``temperature_smooth``: Savitzky-Golay smoothed temperature
    - ``temperature_smooth_gradient``: Time derivative of smoothed temperature

**temptable with corrections** (after processing):
    - All columns above, plus:
    - ``temp_corr``: Temperature-model clock correction (seconds)
    - ``temp_corr_raw``: Original temp_corr before trend removal
    - ``temp_corr_trend``: Spline trend fitted to residuals
    - ``temp_corr_detrend``: temp_corr + temp_corr_trend (final correction)
    - ``std``: Rolling standard deviation of residuals

**freq_changes_table** (commanded frequency divisor changes):
    - ``uxt``: Unix timestamp of command
    - ``met``: Mission Elapsed Time
    - ``divisor``: New clock divisor value (~24000336)
    - ``mjd``: Modified Julian Date (computed)
    - ``flag``: Boolean, True if divisor value is anomalous

Key Classes
-----------
- :class:`ClockCorrection`: Main class for computing clock corrections
- :class:`NuSTARCorr`: Applies corrections to event files

Key Functions
-------------
- :func:`temperature_correction_table`: Builds the temperature-based correction
- :func:`eliminate_trends_in_residuals`: Removes systematic trends from residuals
- :func:`clock_ppm_model`: The physical model for clock drift vs temperature

Notes
-----
- MET (Mission Elapsed Time) is in seconds since MJDREF = 55197.00076601852
- The "Malindi problem" (2012-12-20 to 2013-02-08) requires special handling
- Clock jumps occur when the frequency divisor is commanded to change
"""

import glob
import os
import shutil
from functools import lru_cache
import traceback
import warnings
import numpy as np
from astropy.table import Table, vstack
import pandas as pd
from astropy.time import Time

from scipy.interpolate import PchipInterpolator

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.stats import norm
from scipy.optimize import minimize

from .utils import NUSTAR_MJDREF, splitext_improved, sec_to_mjd
from .utils import filter_with_region, fix_byteorder, rolling_std
from .utils import measure_overall_trend, cross_two_gtis, get_rough_trend_fun
from .utils import spline_through_data, cubic_interpolation, robust_poly_fit
from .utils import get_temperature_parameters
from astropy.io import fits
import tqdm
from astropy import log
from statsmodels.robust import mad
import copy
import holoviews as hv
from holoviews.operation.datashader import datashade
from holoviews import opts

from . import SECONDS_PER_DAY, SECONDS_PER_MONTH, SECONDS_PER_YEAR, HALF_DAY_SECONDS
# =============================================================================
# Constants
# =============================================================================

# File paths
_BAD_POINTS_FILE = "BAD_POINTS_DB.dat"

# Station timing offset: non-Malindi stations have a systematic 0.5 ms offset
# due to different signal processing pipelines
NON_MALINDI_OFFSET_SECONDS = 0.0005

# Clock hardware constants
NOMINAL_CLOCK_DIVISOR = 24000336  # Typical commanded divisor value
CLOCK_DIVISOR_TOLERANCE = 20  # Divisor values outside ±20 of 2.400034e7 are flagged

# Known problematic time intervals (MET values)
# Malindi ground station outage: 2012-12-20 to 2013-02-08
MALINDI_OUTAGE_INTERVALS = [
    (93681591, 98051312),   # 2012/12/20 - 2013/02/08 Malindi outage
    (357295300, 357972500),  # 2021/04/28 - 2021/05/06 Malindi clock issues
]

# Known clock jump times (MET) - when frequency divisor was commanded to change
KNOWN_CLOCK_JUMP_TIMES = np.array([
    78708320, 79657575, 81043985, 82055671, 293346772,
    392200784, 394825882, 395304135, 407914525, 408299422
])

# Reference epoch for absorption-desorption model
ABS_DES_REFERENCE_MET = 77509250

FIXED_CONTROL_POINTS = np.arange(291e6, 295e6, SECONDS_PER_DAY)

hv.extension('bokeh')

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def filter_and_log_table(
        table,
        mask,
        intro_text="Filtering table",
        comment_to_point=None,
        point_log_func=log.info
        ):
    n_before = mask.size
    n_after = np.count_nonzero(mask)
    comment_to_point = f" ({comment_to_point})" if comment_to_point else ""

    if n_after < n_before:
        log.info(f"{intro_text} ({n_after} -> {n_before})")
        for eliminated in table[~mask]['met']:
            point_log_func(f"  MET {eliminated}{comment_to_point}")
        return table[mask]
    return table

def get_bad_points_db(db_file=None):
    """Load the database of known bad clock offset measurement times.

    Bad points are MET values where clock offset measurements are known
    to be unreliable (e.g., due to ground station issues, spacecraft
    anomalies, etc.).

    Parameters
    ----------
    db_file : str, optional
        Path to bad points database file. If None, uses the default
        BAD_POINTS_DB.dat in the package data directory.

    Returns
    -------
    bad_points : np.ndarray
        Array of MET values (longdouble) marking bad measurement times.
    """
    if db_file is None:
        db_file = _BAD_POINTS_FILE

    if not os.path.exists(db_file):
        log.warning(f"No local bad points database found. Using the default one.")
        db_file = os.path.join(datadir, _BAD_POINTS_FILE)

    log.info(f"Reading bad points from {db_file}")
    return np.genfromtxt(db_file, dtype=np.longdouble)


def flag_bad_points(all_data, db_file=None):
    """Flag known bad measurements in a clock offset table.

    Adds or updates a 'flag' column in the table, setting True for
    measurements at times listed in the bad points database.

    Parameters
    ----------
    all_data : astropy.table.Table
        Table with clock offset data. Must contain a 'met' column.
        May already have a 'flag' column (which will be updated).
    db_file : str, optional
        Path to bad points database. If None, uses default.

    Returns
    -------
    all_data : astropy.table.Table
        Input table with 'flag' column added/updated.

    Examples
    --------
    >>> db_file = 'dummy_bad_points.dat'
    >>> np.savetxt(db_file, np.array([-1, 3, 10]))
    >>> all_data = Table({'met': [0, 1, 2, 3, 4]})
    >>> all_data = flag_bad_points(all_data, db_file='dummy_bad_points.dat')
    INFO: ...
    >>> bool(np.all(all_data['flag'] == [False, False, False, True, False]))
    True
    """

    intv = [all_data['met'][0] - 0.5, all_data['met'][-1] + 0.5]
    ALL_BAD_POINTS = get_bad_points_db(db_file)
    log.info("Sorting and filtering bad points...")
    ALL_BAD_POINTS.sort()
    ALL_BAD_POINTS = np.unique(ALL_BAD_POINTS)
    ALL_BAD_POINTS = ALL_BAD_POINTS[
        (ALL_BAD_POINTS > intv[0]) & (ALL_BAD_POINTS < intv[1])]

    idxs = all_data['met'].searchsorted(ALL_BAD_POINTS)

    log.info("Flagging bad points...")
    if 'flag' in all_data.colnames:
        mask = np.array(all_data['flag'], dtype=bool)
    else:
        mask = np.zeros(len(all_data), dtype=bool)

    for idx in idxs:
        if idx >= mask.size:
            continue
        mask[idx] = True
    all_data['flag'] = mask
    log.info("Finished flagging bad points.")
    return all_data


def find_good_time_intervals(temperature_table, clock_jump_times=None):
    """Identify Good Time Intervals (GTIs) for clock correction processing.

    GTIs are contiguous time ranges where:
    1. Temperature data is available (no gaps > 600s)
    2. No clock frequency jumps occur within the interval
    3. Duration is at least half a day (43200 seconds)

    Parameters
    ----------
    temperature_table : astropy.table.Table
        Temperature table with 'met' column. May have 'gti' in metadata.
    clock_jump_times : array-like, optional
        MET values where clock frequency divisor changed. Each jump
        creates a GTI boundary.

    Returns
    -------
    gtis : np.ndarray
        Array of shape (N, 2) with [start, stop] MET for each GTI.
    """
    start_time = temperature_table['met'][0]
    stop_time = temperature_table['met'][-1]

    clock_gtis = no_jump_gtis(
        start_time, stop_time, clock_jump_times)

    if not 'gti' in temperature_table.meta:
        temp_gtis = temperature_gtis(temperature_table)
    else:
        temp_gtis = temperature_table.meta['gti']

    gtis = cross_two_gtis(temp_gtis, clock_gtis)
    lengths = gtis[:, 1] - gtis[:, 0]
    # ensure at least half a day duration for GTIs
    good = lengths > HALF_DAY_SECONDS
    if not np.all(good):
        log.info(f"Some GTIs are too short. cleaning up: {gtis[~good]}")
    for g, is_good in zip(gtis, good):
        log.info(f"GTI: {g[0]}-{g[1]} ({'OK' if is_good else 'too short'})")

    return gtis[good]


def calculate_stats(all_data):
    """Calculate and print statistics on clock correction residuals.

    Computes the Median Absolute Deviation (MAD) and rolling standard
    deviation of the detrended residuals to assess correction quality.

    Parameters
    ----------
    all_data : astropy.table.Table
        Table with 'residual_detrend' column containing the residuals
        between measured clock offsets and the temperature model.
    """
    log.info("Calculating statistics")
    r_std = residual_roll_std(all_data['residual_detrend'])

    scatter = mad(all_data['residual_detrend'])
    print()
    print("----------------------------- Stats -----------------------------------")
    print()
    print(f"Overall MAD: {scatter * 1e6:.0f} us")
    print(f"Minimum scatter: ±{np.nanmin(r_std) * 1e6:.0f} us")
    print()
    print("-----------------------------------------------------------------------")


def load_and_flag_clock_table(clockfile="latest_clock.dat", shift_non_malindi=False, db_file=None):
    """Load clock offset table and flag known bad measurements.

    Convenience function combining load_clock_offset_table and flag_bad_points.

    Parameters
    ----------
    clockfile : str
        Path to clock offset data file.
    shift_non_malindi : bool
        If True, subtract NON_MALINDI_OFFSET_SECONDS from non-Malindi
        station measurements to align them with Malindi.
    db_file : str, optional
        Path to bad points database.

    Returns
    -------
    clock_offset_table : astropy.table.Table
        Clock offset table with 'flag' column populated.
    """
    clock_offset_table = load_clock_offset_table(clockfile,
                                                 shift_non_malindi=shift_non_malindi)

    # Column added/updated: 'flag' (bool) - True for known bad measurements based on db_file
    clock_offset_table = flag_bad_points(
        clock_offset_table, db_file=db_file)
    return clock_offset_table


def spline_detrending(clock_offset_table, temptable, outlier_cuts=None,
                      fixed_control_points=None):
    """Fit a spline to clock residuals and add detrended correction to temptable.

    This function:
    1. Computes residuals between measured clock offsets and temperature model
    2. Optionally removes outliers based on deviation from median-filtered signal
    3. Fits a spline through the residuals to capture systematic trends
    4. Adds the trend to the temperature correction to improve accuracy

    Parameters
    ----------
    clock_offset_table : astropy.table.Table
        Clock offset measurements with columns: 'met', 'offset', 'flag'.
        Filtered to only include times within temptable range.
    temptable : astropy.table.Table
        Temperature table with 'met' and 'temp_corr' columns.
        Modified in-place to add: 'std', 'temp_corr_trend', 'temp_corr_detrend'.
    outlier_cuts : list of float, optional
        Threshold values (in seconds) for iterative outlier removal.
        Measurements deviating more than these thresholds from the median
        are flagged, except for data in the most recent month.
    fixed_control_points : array-like, optional
        MET values where spline control points should be placed.

    Returns
    -------
    temptable : astropy.table.Table
        Input table with added columns:
        - 'std': Rolling standard deviation of residuals
        - 'temp_corr_trend': Spline fit to residuals
        - 'temp_corr_detrend': temp_corr + temp_corr_trend (improved correction)
    """
    tempcorr_idx = np.searchsorted(temptable['met'], clock_offset_table['met'])
    temperature_is_present = tempcorr_idx < temptable['met'].size
    tempcorr_idx = tempcorr_idx[temperature_is_present]

    clock_offset_table = filter_and_log_table(
        clock_offset_table,
        temperature_is_present,
        intro_text="spline_detrending: Filtering clock_offset_table to times with temperature data",
        comment_to_point="beyond temperature table range"
        )

    clock_residuals = \
        np.array(clock_offset_table['offset'] -
                 temptable['temp_corr'][tempcorr_idx])

    clock_mets = clock_offset_table['met']

    if outlier_cuts is not None:
        log.info("Cutting outliers...")
        better_points = np.array(clock_residuals == clock_residuals,
                                 dtype=bool)

        for i, cut in enumerate(outlier_cuts):
            mm = median_filter(clock_residuals, 11)
            wh = ((clock_residuals[better_points] - mm[better_points]) < outlier_cuts[
                i]) | ((clock_residuals[better_points] - mm[better_points]) <
                       outlier_cuts[0])
            better_points[better_points] = ~wh
        # Eliminate too recent flags, in the last month of solution.
        do_not_flag = clock_mets > clock_mets.max() - SECONDS_PER_MONTH
        better_points[do_not_flag] = True

        clock_offset_table = filter_and_log_table(
            clock_offset_table,
            better_points,
            "spline_detrending: Filtering clock_offset_table outliers"
            )
        clock_residuals = clock_residuals[better_points]

    detrend_fun = spline_through_data(
        clock_offset_table['met'], clock_residuals, downsample=10,
        fixed_control_points=fixed_control_points)

    r_std = residual_roll_std(
        clock_residuals - detrend_fun(clock_offset_table['met']))

    clidx = np.searchsorted(clock_offset_table['met'], temptable['met'])
    clidx[clidx == clock_offset_table['met'].size] = \
        clock_offset_table['met'].size - 1

    temptable['std'] = r_std[clidx]
    temptable['temp_corr_trend'] = detrend_fun(temptable['met'])
    temptable['temp_corr_detrend'] = \
        temptable['temp_corr'] + temptable['temp_corr_trend']

    return temptable


def eliminate_trends_in_residuals(temptable, clock_offset_table,
                                  gtis, debug=False,
                                  fixed_control_points=None):
    """Remove systematic trends from temperature-model residuals within each GTI.

    For each Good Time Interval (GTI), this function:
    1. Extracts clock offset measurements and computes residuals vs temp model
    2. Fits a low-order robust polynomial to the residuals
    3. Adds the polynomial correction to the temperature model
    4. For Bad Time Intervals (gaps between GTIs), interpolates the correction
    5. Finally applies spline_detrending for fine-scale adjustments

    The Malindi ground station is preferred for interpolation due to its
    more reliable timing. During known Malindi outages, other stations
    are used with a 0.5 ms offset correction.

    Parameters
    ----------
    temptable : astropy.table.Table
        Temperature correction table with 'met' and 'temp_corr' columns.
        Modified in-place. Will have 'temp_corr_raw' added to preserve original.
    clock_offset_table : astropy.table.Table
        Clock offset measurements with 'met', 'offset', 'flag', 'station' columns.
    gtis : np.ndarray
        Array of shape (N, 2) with [start, stop] MET for each GTI.
    debug : bool, optional
        If True, save diagnostic plots for each GTI.
    fixed_control_points : array-like, optional
        MET values for spline control points in final detrending step.

    Returns
    -------
    temptable : astropy.table.Table
        Modified temperature table with improved 'temp_corr' and additional
        diagnostic columns from spline_detrending.

    Notes
    -----
    The function distinguishes between:
    - GTIs: Intervals with good temperature data, processed with polynomial fits
    - BTIs: Gaps between GTIs, handled by interpolation from nearby good data
    """
    # good = clock_offset_table['met'] < np.max(temptable['met'])
    # clock_offset_table = clock_offset_table[good]
    temptable['temp_corr_raw'] = temptable['temp_corr']

    tempcorr_idx = np.searchsorted(temptable['met'],
                                   clock_offset_table['met'])
    temperature_is_present = tempcorr_idx < temptable['met'].size
    tempcorr_idx = tempcorr_idx[temperature_is_present]

    clock_offset_table = filter_and_log_table(
        clock_offset_table,
        temperature_is_present,
        intro_text="eliminate_trends_in_residuals: Filtering clock_offset_table to times with temperature data",
        comment_to_point="beyond temperature table range"
        )

    clock_residuals = \
        clock_offset_table['offset'] - temptable['temp_corr'][tempcorr_idx]

    # Only use for interpolation Malindi points; however, during the Malindi
    # problem in 2013, use the other data for interpolation but subtracting
    # half a millisecond

    use_for_interpol, bad_malindi_time = \
        get_malindi_data_except_when_out(clock_offset_table)

    clock_residuals[bad_malindi_time] -= 0.0005

    good = (clock_residuals == clock_residuals) & ~clock_offset_table['flag'] & use_for_interpol

    clock_offset_table = filter_and_log_table(
        clock_offset_table,
        good,
        intro_text="eliminate_trends_in_residuals: Filtering clock_offset_table to "
                 "trustworthy points (Malindi or during Malindi outage)",
        comment_to_point="bad point or not from Malindi",
        point_log_func=log.debug
    )
    clock_residuals = clock_residuals[good]

    for g in gtis:
        log.info(f"Treating data from METs {g[0]}--{g[1]}")
        start, stop = g

        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)

        if cl_idx_end - cl_idx_start < 3:
            log.info("Too few clock measurements in this interval")
            continue

        temp_idx_start, temp_idx_end = \
            np.searchsorted(temptable['met'], g)

        table_new = temptable[temp_idx_start:temp_idx_end]
        cltable_new = clock_offset_table[cl_idx_start:cl_idx_end]
        met = cltable_new['met']

        residuals = clock_residuals[cl_idx_start:cl_idx_end]
        met0 = met[0]
        met_rescale = (met - met0)/(met[-1] - met0)
        _, m, q = measure_overall_trend(met_rescale, residuals)
        # p_new = get_rough_trend_fun(met, residuals)
        #
        # if p_new is not None:
        #     p = p_new
        poly_order = min(met.size // 300 + 1, 2)
        p0 = np.zeros(poly_order + 1)
        p0[0] = q
        if p0.size > 1:
            p0[1] = m
        log.info(f"Fitting a polinomial of order {poly_order}")
        p = robust_poly_fit(met_rescale, residuals, order=poly_order,
                            p0=p0)

        table_mets_rescale = (table_new['met'] - met0) / (met[-1] - met0)
        corr = p(table_mets_rescale)

        sub_residuals = residuals - p(met_rescale)
        m = (sub_residuals[-1] - sub_residuals[0]) / (met_rescale[-1] - met_rescale[0])
        q = sub_residuals[0]

        corr = corr + m * (table_mets_rescale - met_rescale[0]) + q
        table_new['temp_corr'] += corr

        if debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(table_new['met'], table_new['temp_corr'], alpha=0.5)
            plt.scatter(cltable_new['met'], cltable_new['offset'])
            plt.plot(table_new['met'], table_new['temp_corr'])
            plt.savefig(f'{int(start)}--{int(stop)}_detr.png')
            plt.close(fig)
        # plt.show()

        # print(f'df/f = {(p(stop) - p(start)) / (stop - start)}')

    bti_list = [[0, 77674700]]
    bti_list += [[g0, g1] for g0, g1 in zip(gtis[:-1, 1], gtis[1:, 0])]
    bti_list += [[gtis[-1, 1], max(clock_offset_table['met'][-1], temptable['met'][-1]) + 10000]]
    btis = np.array(bti_list)
    # Interpolate the solution along bad time intervals
    for g in btis:
        start, stop = g
        log.info(f"Treating bad data from METs {start}--{stop}")

        temp_idx_start, temp_idx_end = \
            np.searchsorted(temptable['met'], g)
        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)
        if temp_idx_end - temp_idx_start == 0 and \
                temp_idx_end < len(temptable) and temp_idx_start > 0:
            log.info("No temperature measurements in this interval")
            continue
        else:
            table_new = temptable[temp_idx_start:temp_idx_end]

            last_good_tempcorr = temptable['temp_corr'][temp_idx_start - 1]
            last_good_time = temptable['met'][temp_idx_start - 1]

            local_clockoff = clock_offset_table[max(cl_idx_start - 1, 0):cl_idx_end + 1]
            clock_off = local_clockoff['offset']
            clock_tim = local_clockoff['met']

            if temp_idx_end < temptable['temp_corr'].size:
                next_good_tempcorr = temptable['temp_corr'][temp_idx_end + 1]
                next_good_time = temptable['met'][temp_idx_end + 1]
                clock_off = np.concatenate(
                    ([last_good_tempcorr], clock_off, [next_good_tempcorr]))
                clock_tim = np.concatenate(
                    ([last_good_time], clock_tim, [next_good_time]))
            else:
                clock_off = np.concatenate(
                    ([last_good_tempcorr], clock_off))
                clock_tim = np.concatenate(
                    ([last_good_time], clock_tim))

                next_good_tempcorr = clock_off[-1]
                next_good_time = clock_tim[-1]

            if cl_idx_end - cl_idx_start < 2:
                log.info("Not enough good clock measurements. Interpolating")
                m = (next_good_tempcorr - last_good_tempcorr) / \
                (next_good_time - last_good_time)
                q = last_good_tempcorr
                table_new['temp_corr'][:] = \
                    q + (table_new['met'] - last_good_time) * m
                continue

            order = np.argsort(clock_tim)

            clock_off_fun = PchipInterpolator(
                clock_tim[order], clock_off[order],
                extrapolate=True)
            table_new['temp_corr'][:] = clock_off_fun(table_new['met'])

    log.info("Final detrending...")

    # Columns added to temptable: 'std', 'temp_corr_trend', 'temp_corr_detrend'
    table_new = spline_detrending(
        clock_offset_table, temptable,
        outlier_cuts=[-0.002, -0.001],
        fixed_control_points=fixed_control_points)
    return table_new


def residual_roll_std(residuals, window=30):
    """Calculate the rolling standard deviation of clock residuals.

    Examples
    --------
    >>> residuals = np.zeros(5000)
    >>> residuals[:4000] = np.random.normal(0, 1, 4000)
    >>> roll_std = residual_roll_std(residuals, window=500)
    >>> bool(np.allclose(roll_std[:3500], 1., rtol=0.2))
    True
    >>> bool(np.all(roll_std[4500:] == 0.))
    True
    """
    r_std = rolling_std(residuals, window)
    # r_std = rolling_std(np.diff(residuals), window) / np.sqrt(2)
    # return np.concatenate(([r_std[:1], r_std]))
    return r_std


def get_malindi_data_except_when_out(clock_offset_table):
    """Select offset measurements from Malindi, unless Malindi is out.

    In the time interval between METs 93681591 and 98051312, Malindi was out
    of work. For that time interval, we use all clock offset measurements
    available. In all other cases, we just use Malindi

    Parameters
    ----------
    clock_offset_table : :class:`Table` object
        Table containing the clock offset measurements. At least, it has to
        contain a 'met' and a 'station' columns.

    Returns
    -------
    use_for_interpol : array of ``bool``
        Mask of "trustworthy" time measurements
    bad_malindi_time : array of ``bool``
        Mask of time measurements during Malindi outage

    Example
    -------
    >>> clocktable = Table({'met': [93681592, 1e8, 1.5e8],
    ...                     'station': ['SNG', 'MLD', 'UHI']})
    >>> ufp, bmt = get_malindi_data_except_when_out(clocktable)
    >>> assert np.all(ufp == [True, True, False])
    >>> assert np.all(bmt == [True, False, False])
    """
    # Covers 2012/12/20 - 2013/02/08 Malindi outage
    # Also covers 2021/04/28 - 2021/05/06 issues with Malindi clock

    no_malindi_intvs = [[93681591, 98051312],[357295300, 357972500]]
    clock_mets = clock_offset_table['met']

    bad_malindi_time = np.zeros(len(clock_mets), dtype=bool)
    for nmi in no_malindi_intvs:
        bad_malindi_time = bad_malindi_time | (clock_mets >= nmi[0]) & (
                    clock_mets < nmi[1])

    malindi_stn = clock_offset_table['station'] == 'MLD'
    use_for_interpol = \
        (malindi_stn | bad_malindi_time)

    return use_for_interpol, bad_malindi_time


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


def read_clock_offset_table(clockoffset_file=None, shift_non_malindi=False):
    """Read the clock offset table from a file and prepare it for processing.

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
    if shift_non_malindi:
        log.info("Shifting non-Malindi clock offsets down by 0.5 ms")
        all_but_malindi = clock_offset_table['station'] != 'MLD'
        clock_offset_table['offset'][all_but_malindi] -= 0.0005
    clock_offset_table['mjd'] = sec_to_mjd(clock_offset_table['met'])
    # clock_offset_table.remove_row(len(clock_offset_table) - 1)
    clock_offset_table['flag'] = np.zeros(len(clock_offset_table), dtype=bool)

    log.info("Flagging bad points in clock offset table...")
    ALL_BAD_POINTS = get_bad_points_db()

    for b in ALL_BAD_POINTS:
        nearest = np.argmin(np.abs(clock_offset_table['met'] - b))
        if np.abs(clock_offset_table['met'][nearest] - b) < 1:
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


def no_jump_gtis(start_time, stop_time, clock_jump_times=None):
    """Create GTIs that avoid clock frequency jump times.

    Splits a time range into segments at each clock jump, so that
    no segment contains a frequency divisor change.

    Parameters
    ----------
    start_time : float
        Start of time range (MET).
    stop_time : float
        End of time range (MET).
    clock_jump_times : array-like, optional
        MET values where clock jumps occur.

    Returns
    -------
    gtis : np.ndarray
        Array of [start, stop] pairs, one per segment.

    Examples
    --------
    >>> gtis = no_jump_gtis(0, 3, [1, 1.1])
    >>> bool(np.allclose(gtis, [[0, 1], [1, 1.1], [1.1, 3]]))
    True
    >>> gtis = no_jump_gtis(0, 3)
    >>> bool(np.allclose(gtis, [[0, 3]]))
    True
    """
    if clock_jump_times is None:
        return [[start_time, stop_time]]

    clock_gtis = []
    current_start = start_time
    for jump in clock_jump_times:
        clock_gtis.append([current_start, jump])
        current_start = jump
    clock_gtis.append([current_start, stop_time])
    clock_gtis = np.array(clock_gtis)
    return clock_gtis


def temperature_gtis(temperature_table, max_distance=600):
    """Identify GTIs based on temperature data continuity.

    Creates GTIs where consecutive temperature measurements are within
    max_distance seconds of each other. Gaps larger than this indicate
    missing data and create GTI boundaries.

    Parameters
    ----------
    temperature_table : astropy.table.Table
        Table with 'met' column of measurement times.
    max_distance : float, optional
        Maximum allowed gap (seconds) between measurements. Default 600s.

    Returns
    -------
    gtis : np.ndarray
        Array of shape (N, 2) with [start, stop] MET for each GTI.

    Examples
    --------
    >>> temperature_table = Table({'met': [0, 1, 2, 10, 11, 12]})
    >>> gti = temperature_gtis(temperature_table, 5)
    >>> bool(np.allclose(gti, [[0, 2], [10, 12]]))
    True
    >>> temperature_table = Table({'met': [-10, 0, 1, 2, 10, 11, 12, 20]})
    >>> gti = temperature_gtis(temperature_table, 5)
    >>> bool(np.allclose(gti, [[0, 2], [10, 12]]))
    True
    """
    temp_condition = np.concatenate(
        ([False], np.diff(temperature_table['met']) > max_distance, [False]))

    bad_times = [[482.2896e6, 482.2958e6]]
    for b in bad_times:
        bad = np.where((temperature_table['met'] > b[0]) & (temperature_table['met'] < b[1]))
        temp_condition[bad[0]] = True

    temp_edges_l = np.concatenate((
        [temperature_table['met'][0]],
        temperature_table['met'][temp_condition[:-1]]))

    temp_edges_h = np.concatenate((
        [temperature_table['met'][temp_condition[1:]],
         [temperature_table['met'][-1]]]))

    temp_gtis = np.array(list(zip(
        temp_edges_l, temp_edges_h)))

    length = temp_gtis[:, 1] - temp_gtis[:, 0]

    return temp_gtis[length > 0]


def read_freq_changes_table(freqchange_file=None, filter_bad=True):
    """Read the table of commanded clock frequency divisor changes.

    The clock divisor determines the tick rate of the spacecraft clock.
    This table records when the divisor was commanded to change, which
    causes discontinuities in the clock drift that must be handled
    separately.

    Parameters
    ----------
    freqchange_file : str, optional
        Path to frequency changes file (e.g., 'nustar_freq_changes-2018-10-30.dat').
        If None, uses the latest file in the data directory.
    filter_bad : bool, optional
        If True (default), remove entries with anomalous divisor values
        (more than ±20 from 2.400034e7).

    Returns
    -------
    freq_changes_table : astropy.table.Table
        Table with columns: 'uxt', 'met', 'divisor', 'mjd', 'flag'.
        Sorted by MET. Known bad entries are corrected per FREQ_CHANGE_DB.
    """
    if freqchange_file is None:
        freqchange_file = _look_for_freq_change_file()
    log.info(f"Reading frequency changes from {freqchange_file}")
    freq_changes_table = Table.read(freqchange_file,
                                format='csv', delimiter=' ',
                                comment=r"\s*#",
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
    # freq_changes_table.remove_row(len(freq_changes_table) - 1)
    freq_changes_table['flag'] = \
        np.abs(freq_changes_table['divisor'] - 2.400034e7) > 20

    if filter_bad:
        freq_changes_table = filter_and_log_table(
            freq_changes_table,
            ~freq_changes_table['flag'],
            intro_text="read_freq_changes_table: Filtering freq_changes_table to remove bad points",
            comment_to_point="bad frequency change point",
            point_log_func=log.debug
            )

    return freq_changes_table


def _filter_csv_temperature_table(tablefile, start_date=None, end_date=None, tmpfile='tmp.csv'):
    """Filter the CSV temperature table to only include rows within a specified date range.

    Parameters
    ----------
    tablefile : str
        Path to the input CSV temperature table file.
    start_date : float, optional
        Start date for filtering (MJD).
    end_date : float, optional
        End date for filtering (MJD).
    tmpfile : str, optional
        Path to the temporary filtered file.

    Returns
    -------
    tmpfile : str
        Path to the filtered CSV temperature table file.
    """
    from datetime import timezone

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
    """Read the temperature table from a CSV file, optionally filtering by MJD range.

    Parameters
    ----------
    mjdstart : float, optional
        Start MJD for filtering the temperature table. If None, no lower bound is applied.
    mjdstop : float, optional
        Stop MJD for filtering the temperature table. If None, no upper bound is applied.
    temperature_file : str, optional
        Path to the CSV temperature table file. If None, the default file is used.

    Returns
    -------
    temptable : Table
        The filtered temperature table.
    """
    if mjdstart is not None or mjdstop is not None:
        mjdstart_use = mjdstart
        mjdstop_use = mjdstop
        if mjdstart is not None:
            mjdstart_use -= 10
        if mjdstop is not None:
            mjdstop_use += 10
        log.info("Filtering table...")
        tmpfile = _filter_csv_temperature_table(temperature_file,
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
    temptable['met'] = (temptable["mjd"] - NUSTAR_MJDREF) * SECONDS_PER_DAY
    temptable.remove_column('Time')
    temptable.sort("met")
    temptable.rename_column('tp_eps_ceu_txco_tmp', 'temperature')
    temptable["temperature"] = np.array(temptable["temperature"], dtype=float)
    if os.path.exists('tmp.csv'):
        os.unlink('tmp.csv')

    return temptable


def read_saved_temptable(mjdstart=None, mjdstop=None,
                         temperature_file='temptable.hdf5'):
    """Read a previously saved temperature table from an HDF5 file.

    Parameters
    ----------
    mjdstart : float, optional
        Start MJD for filtering the temperature table. If None, no lower bound is applied.
    mjdstop : float, optional
        Stop MJD for filtering the temperature table. If None, no upper bound is applied.
    temperature_file : str, optional
        Path to the HDF5 temperature table file. If None, the default file is used.

    Returns
    -------
    temptable : Table
        The filtered temperature table.
    """
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
    """Read the temperature table from a FITS file, e.g. those from the HK data.

    Parameters
    ----------
    temperature_file : str
        Path to the FITS temperature table file.

    Returns
    -------
    temptable : Table
        The temperature table.
    """
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
    """Interpolate the temperature table to a regular time grid with spacing dt."""
    time = temptable['met']
    temperature = temptable['temperature']
    new_times = np.arange(time[0], time[-1], dt)
    idxs = np.searchsorted(time, new_times)
    return Table({'met': new_times, 'temperature': temperature[idxs]})


def read_temptable(temperature_file=None, mjdstart=None, mjdstop=None,
                   dt=None, gti_tolerance=600):
    """Read the temperature table, handling different formats.

    Parameters
    ----------
    temperature_file : str, optional
        Path to the temperature table file. If None, the default file is used.

    Other parameters
    ----------------
    mjdstart : float, optional
        Start MJD for filtering the temperature table. If None, no lower bound is applied.
    mjdstop : float, optional
        Stop MJD for filtering the temperature table. If None, no upper bound is applied.
    dt : float, optional
        Time resolution for interpolation. If None, no interpolation is done. Default is None.
    gti_tolerance : float, optional
        Maximum gap (seconds) between temperature measurements to be considered a GTI. Default is
        600 seconds.
    Returns
    -------
    temptable : Table
        The temperature table with columns 'met', 'temperature', 'mjd', and optionally 'temperature_smooth' and 'temperature_smooth_gradient'.
    """
    if temperature_file is None:
        temperature_file = _look_for_temptable()
    log.info(f"Reading temperature_information from {temperature_file}")
    log.info(f"dt={str(dt)}")
    log.info(f"mjdstart={mjdstart}")
    log.info(f"mjdstop={mjdstop}")
    ext = splitext_improved(temperature_file)[1]
    if ext in ['.csv']:
        temptable = read_csv_temptable(mjdstart, mjdstop, temperature_file)
    elif ext in ['.hk', '.hk.gz']:
        temptable = read_fits_temptable(temperature_file)
    elif ext in ['.hdf5', '.h5']:
        temptable = read_saved_temptable(mjdstart, mjdstop,
                                         temperature_file)
        temptable = fix_byteorder(temptable)
    else:
        raise ValueError('Unknown format for temperature file')

    temp_gtis = temperature_gtis(temptable, gti_tolerance)
    if dt is not None:
        temptable = interpolate_temptable(temptable, dt)
    else:
        good = np.diff(temptable['met']) > 0
        good = np.concatenate((good, [True]))

        temptable = filter_and_log_table(
            temptable,
            good,
            intro_text="read_temptable: Filtering temptable for non-increasing time points",
            comment_to_point="non-increasing time point",
            point_log_func=log.debug
        )
    temptable.meta['gti'] = temp_gtis

    window = np.median(1000 / np.diff(temptable['met']))
    window = int(window // 2 * 2 + 1)
    log.info(f"Smoothing temperature with a window of {window} points")
    temptable['temperature_smooth'] = \
        savgol_filter(temptable['temperature'], window, 3)
    temptable['temperature_smooth_gradient'] = \
        np.gradient(temptable['temperature_smooth'], temptable['met'],
                    edge_order=2)

    return temptable


@lru_cache(maxsize=64)
def load_temptable(temptable_name):
    log.info(f"Reading data from {temptable_name}")
    IS_CSV = temptable_name.endswith('.csv')
    hdf5_name = temptable_name.replace('.csv', '.hdf5')

    h5_exists = os.path.exists(hdf5_name)
    h5_newer = h5_exists and os.path.getmtime(hdf5_name) > os.path.getmtime(temptable_name)
    if IS_CSV and h5_exists and not h5_newer:
        log.info(f"HDF5 file {hdf5_name} is older than CSV file {temptable_name}. "
                 "Re-reading from CSV and overwriting HDF5.")
    if IS_CSV and h5_newer:
        IS_CSV = False
        # Returns table with: 'met', 'temperature', 'mjd', 'temperature_smooth'
        temptable_raw = read_temptable(hdf5_name, dt=10)
    else:
        # Returns table with: 'met', 'temperature', 'mjd', 'temperature_smooth'
        temptable_raw = read_temptable(temptable_name, dt=10)

    if IS_CSV:
        log.info(f"Saving temperature data to {hdf5_name}")
        temptable_raw.write(hdf5_name, overwrite=True)
    return temptable_raw


@lru_cache(maxsize=64)
def load_freq_changes(freq_change_file):
    # Returns table with: 'uxt', 'met', 'divisor', 'mjd', 'flag'
    log.info(f"Reading data from {freq_change_file}")
    return read_freq_changes_table(freq_change_file)


@lru_cache(maxsize=64)
def load_clock_offset_table(clock_offset_file, shift_non_malindi=False):
    # Returns table with: 'met', 'offset', 'station', 'mjd', 'flag'
    return read_clock_offset_table(clock_offset_file,
                                   shift_non_malindi=shift_non_malindi)


class ClockCorrection():
    """Main class for computing NuSTAR temperature-based clock corrections.

    This class orchestrates the full clock correction pipeline:
    1. Loads temperature data and clock offset measurements
    2. Computes temperature-dependent clock drift model
    3. Optionally adjusts the model using measured clock offsets
    4. Can write CALDB-compatible clock correction FITS files

    Parameters
    ----------
    temperature_file : str
        Path to temperature data file (CSV or HDF5).
    mjdstart, mjdstop : float, optional
        MJD range for the correction. If None, derived from data.
    temperature_dt : float, optional
        Time resolution for temperature interpolation. Default 10s.
    adjust_absolute_timing : bool, optional
        If True, adjust the temperature model using measured clock offsets
        via eliminate_trends_in_residuals. Default False.
    force_divisor : int, optional
        Force a specific clock divisor value instead of reading from file.
    label : str, optional
        Label for output files.
    additional_days : float, optional
        Extra days of data to load beyond requested range. Default 2.
    clock_offset_file : str, optional
        Path to clock offset measurements file.
    hdf_dump_file : str, optional
        Path for caching intermediate results.
    freqchange_file : str, optional
        Path to frequency changes file.
    spline_through_residuals : bool, optional
        Reserved for future use.

    Attributes
    ----------
    temptable : astropy.table.Table
        Temperature data table.
    clock_offset_table : astropy.table.Table
        Clock offset measurements.
    temperature_correction_data : astropy.table.Table
        Computed correction table with 'met', 'temp_corr', etc.
    gtis : np.ndarray
        Good Time Intervals for processing.
    """
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

        # Sets self.temptable with: 'met', 'temperature', 'mjd', 'temperature_smooth'
        self.read_temptable()

        if mjdstart is None:
            mjdstart = sec_to_mjd(self.temptable['met'].min())
        else:
            mjdstart = mjdstart - additional_days / 2

        self.clock_offset_file = clock_offset_file
        # Returns table with: 'met', 'offset', 'station', 'mjd', 'flag'
        self.clock_offset_table = \
            read_clock_offset_table(self.clock_offset_file,
                                    shift_non_malindi=True)
        if mjdstop is None:
            last_met = max(self.temptable['met'].max(),
                           self.clock_offset_table['met'].max())
            mjdstop = sec_to_mjd(last_met)

            mjdstop = mjdstop + additional_days / 2

        self.mjdstart = mjdstart
        self.mjdstop = mjdstop

        self.met_start = (self.mjdstart - NUSTAR_MJDREF) * SECONDS_PER_DAY
        self.met_stop = (self.mjdstop - NUSTAR_MJDREF) * SECONDS_PER_DAY

        if label is None or label == "":
            label = f"{self.met_start}-{self.met_stop}"

        self.force_divisor = force_divisor
        self.adjust_absolute_timing = adjust_absolute_timing

        self.hdf_dump_file = hdf_dump_file
        self.plot_file = label + "_clock_adjustment.png"

        self.clock_jump_times = KNOWN_CLOCK_JUMP_TIMES

        self.fixed_control_points = FIXED_CONTROL_POINTS
        #  Sum 30 seconds to avoid to exclude these points
        #  from previous interval
        self.gtis = find_good_time_intervals(
            self.temptable, self.clock_jump_times + 30)

        self.temperature_correction_data = \
            temperature_correction_table(
                self.met_start, self.met_stop,
                # force_divisor=self.force_divisor,
                time_resolution=10,
                temptable = self.temptable,
                # hdf_dump_file=self.hdf_dump_file,
                freqchange_file=self.freqchange_file)

        if adjust_absolute_timing:
            log.info("Adjusting temperature correction")
            try:
                # Adds 'temp_corr_nodetrend'; replaces 'temp_corr' with detrended version
                self.adjust_temperature_correction()
            except Exception:
                logfile = 'adjust_temperature_error.log'
                log.warning(f"Temperature adjustment failed. "
                            f"Full error stack in {logfile}")
                with open(logfile, 'w') as fobj:
                    traceback.print_exc(file=fobj)

    def read_temptable(self, cache_temptable_name=None):
        """Read the temperature table, using a cached version if available."""
        if cache_temptable_name is not None and \
                os.path.exists(cache_temptable_name):
            self.temptable = Table.read(cache_temptable_name, path='temptable')
        else:
            # Returns table with: 'met', 'temperature', 'mjd', 'temperature_smooth'
            self.temptable = \
                    read_temptable(temperature_file=self.temperature_file,
                                   mjdstart=self.mjdstart,
                                   mjdstop=self.mjdstop,
                                   dt=self.temperature_dt)
            if cache_temptable_name is not None:
                self.temptable.write(cache_temptable_name, path='temptable')

    def temperature_correction_fun(self, adjust=False):
        """Create an interpolating function for the temperature correction.

        Uses PCHIP interpolation to create a smooth function that can be evaluated
        at any time. If adjust is True, uses the adjusted temperature correction data.
        """
        data = self.temperature_correction_data

        if not adjust:
            warnings.warn("Since the use of PCHIP interpolation, the adjust option is "
                          "ignored. The function will always use the adjusted temperature correction data.")

        return PchipInterpolator(np.array(data['met']), np.array(data['temp_corr']),
                        extrapolate=True)

    def adjust_temperature_correction(self):
        """Adjust the temperature correction using measured clock offsets."""
        # Adds 'temp_corr_detrend' column to returned table
        table_new = eliminate_trends_in_residuals(
            self.temperature_correction_data,
            load_clock_offset_table(
                self.clock_offset_file, shift_non_malindi=True), self.gtis,
            debug=False,
            fixed_control_points=self.fixed_control_points)
        table_new['temp_corr_nodetrend'] = table_new['temp_corr']
        table_new['temp_corr'] = table_new['temp_corr_detrend']
        table_new.remove_column('temp_corr_detrend')

        # 'temp_corr_detrend']
        self.temperature_correction_data = table_new

    def write_clock_file(self, filename=None, save_nodetrend=False,
                         shift_times=0., highres=False):
        """Write a CALDB-compatible clock correction FITS file with the temperature correction.


        Parameters
        ----------
        filename : str, optional
            Path to the output FITS file. If None, defaults to 'new_clock_file.fits'.
        save_nodetrend : bool, optional
            If True, also save the non-detrended temperature correction in the FITS file.
        shift_times : float, optional
            Time shift (seconds) to apply to the clock offset correction. Default is 0. This can be used to align the correction with the measured clock offsets if needed.
        highres : bool, optional
            If True, save the clock correction at the full time resolution of the temperature table. If False (default), save a subsampled version with points every 100 seconds, and more points around known clock jump times.

        Returns
        -------
        None
            Writes the clock correction data to the specified FITS file.
        """
        from astropy.io.fits import Header, HDUList, BinTableHDU, PrimaryHDU
        from astropy.time import Time

        if filename is None:
            filename = 'new_clock_file.fits'
        table_new = self.temperature_correction_data
        clock_offset_table = self.clock_offset_table

        good_clock = clock_offset_table['met'] < table_new['met'][-1]

        clock_offset_table = filter_and_log_table(
            clock_offset_table,
            good_clock,
            intro_text="write_clock_file: Filtering clock_offset_table to remove points"
            " beyond the temperature table range",
            comment_to_point="beyound temperature range",
        )

        tempcorr_idx = np.searchsorted(table_new['met'],
                                       clock_offset_table['met'])
        temperature_is_present = tempcorr_idx < table_new['met'].size
        tempcorr_idx = tempcorr_idx[temperature_is_present]

        clock_offset_table = filter_and_log_table(
            clock_offset_table,
            temperature_is_present,
            intro_text="write_clock_file: Filtering clock_offset_table to remove points"
            " without temperature information",
            comment_to_point="no temperature data present",
        )

        clock_residuals_detrend = clock_offset_table['offset'] - \
                                  table_new['temp_corr'][tempcorr_idx]

        good, _ = get_malindi_data_except_when_out(clock_offset_table) & ~clock_offset_table['flag']

        roll_std = residual_roll_std(clock_residuals_detrend[good])
        control_points = clock_offset_table['met'][good]
        clock_err_fun = PchipInterpolator(
            control_points,
            roll_std,
            extrapolate=True)

        clockerr = clock_err_fun(table_new['met'])

        bti_list = list(zip(self.gtis[:-1, 1], self.gtis[1:, 0]))
        btis = np.array(bti_list)

        for g in btis:
            start, stop = g
            log.info(f"Treating bad data from METs {start}--{stop}")

            temp_idx_start, temp_idx_end = \
                np.searchsorted(table_new['met'], g)
            if temp_idx_end - temp_idx_start == 0:
                continue
            clockerr[temp_idx_start:temp_idx_end] = 0.001
        clockerr[table_new['met'] > self.gtis[-1, 1]] = 0.001
        new_clock_table = Table(
            {'TIME': table_new['met'],
             'CLOCK_OFF_CORR': -table_new['temp_corr'] + shift_times,
             'CLOCK_FREQ_CORR': np.gradient(
                 -table_new['temp_corr'],
                 table_new['met'], edge_order=2),
             'CLOCK_ERR_CORR': clockerr})

        allmets = new_clock_table['TIME']

        if not highres:
            good_for_clockfile = np.zeros(allmets.size, dtype=bool)
            good_for_clockfile[::100] = True
            twodays = SECONDS_PER_DAY * 2
            for jumptime in self.clock_jump_times:
                idx0, idx1 = np.searchsorted(
                    allmets, [jumptime - twodays, jumptime + twodays])
                # print(idx0, idx1)
                good_for_clockfile[idx0:idx1:10] = True
            new_clock_table_subsample = new_clock_table[good_for_clockfile]
        else:
            new_clock_table_subsample = new_clock_table
        del new_clock_table

        if save_nodetrend:
            new_clock_table_nodetrend = Table(
                {'TIME': table_new['met'],
                 'CLOCK_OFF_CORR': -table_new['temp_corr_nodetrend'] + shift_times,
                 'CLOCK_FREQ_CORR': np.gradient(
                     -table_new['temp_corr'],
                     table_new['met'], edge_order=2),
                 'CLOCK_ERR_CORR': clock_err_fun(
                     table_new['met'])})
            if not highres:
                new_clock_table_subsample_nodetrend = \
                    new_clock_table_nodetrend[good_for_clockfile]
            else:
                new_clock_table_subsample_nodetrend = \
                    new_clock_table_nodetrend

            del new_clock_table_nodetrend

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
        header["XTENSION"] = ('BINTABLE', "Written by Python")
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
        header['COMMENT'] = ""
        if shift_times != 0.:
            header['COMMENT'] = f"A systematic shift of {shift_times} s was applied to the correction"
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
        header["CCNM0001"] = ('CLOCK', "Type of calibration data")

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
        hdulist = [prihdu, hdu]
        if save_nodetrend:
            hdu_nodetrend = \
                BinTableHDU(new_clock_table_subsample_nodetrend,
                            header=header, name="NU_FINE_CLOCK_NODETREND")
            hdulist.append(hdu_nodetrend)

        HDUList(hdulist).writeto(filename, overwrite=True)


def interpolate_clock_function(new_clock_table, mets):
    """Interpolate the clock correction function at the given METs using the new clock table.

    Parameters
    ----------
    new_clock_table : astropy.table.Table
        The new clock correction table with columns 'TIME', 'CLOCK_OFF_CORR', and 'CLOCK_FREQ_CORR'
    mets : array-like
        The METs at which to interpolate the clock correction

    Returns
    -------
    tuple
        A tuple containing the interpolated clock corrections and a boolean mask indicating which METs were successfully interpolated
    """
    tab_times = new_clock_table['TIME']
    good_mets = (mets > tab_times.min()) & (mets < tab_times.max())
    mets = mets[good_mets]
    tab_idxs = np.searchsorted(tab_times, mets, side='right') - 1

    clock_off_corr = new_clock_table['CLOCK_OFF_CORR']
    clock_freq_corr = new_clock_table['CLOCK_FREQ_CORR']

    x = np.array(mets)
    xtab = [tab_times[tab_idxs], tab_times[tab_idxs + 1]]
    ytab = [clock_off_corr[tab_idxs], clock_off_corr[tab_idxs + 1]]
    yptab = [clock_freq_corr[tab_idxs], clock_freq_corr[tab_idxs + 1]]

    return cubic_interpolation(x, xtab, ytab, yptab), good_mets


def _analyze_residuals(filtered_data, nbins=101, fit_range=[-np.inf, np.inf]):
    """Analyze the residuals by fitting a Gaussian to the histogram of the filtered data.

    Parameters
    ----------
    filtered_data : array-like
        The data to analyze, typically the residuals after detrending.
    nbins : int, optional
        The number of bins to use for the histogram. Default is 101.
    fit_range : list, optional
        The range of values to consider for fitting the Gaussian. Default is [-inf, inf].

    Returns
    -------
    edges: array-like
        edges of the histogram bins,
    frequencies: array-like
        the normalized frequencies
    info_dict: dict
        a dictionary of statistics including the median, MAD, and fitted Gaussian parameters.
    """
    frequencies, edges = np.histogram(
        filtered_data,
        bins=np.linspace(-1000, 1500, nbins)
    )
    errors = frequencies**0.5
    errors[errors < 1] = 1
    renorm = np.sum(frequencies*np.diff(edges))
    frequencies = frequencies / renorm
    errors /= renorm
    edge_centers = (edges[:-1] + edges[1:]) / 2
    good = np.abs(filtered_data) < 2000
    stats = {"median": np.median(filtered_data[good]), "mad": mad(filtered_data[good])}

    inner = (edge_centers > fit_range[0]) & (edge_centers < fit_range[1])
    def fitfunc(p, x):
        if p[2] < 0:
            return np.inf
        return p[2] * norm(loc=p[0], scale=p[1]).pdf(x)

    def errfunc(p, x, y, e):
        res = np.sum(((y - fitfunc(p, x)) / e)**2)
        return res

    out = minimize(errfunc, [stats["median"], stats["mad"], 1], args=(edge_centers[inner], frequencies[inner], errors[inner]))
    stats["fit_mean"] = out.x[0]
    stats["fit_std"] = out.x[1]
    stats["fit_norm"] = out.x[2]
    return edges, frequencies, stats

def plot_scatter(new_clock_table, clock_offset_table, shift_times=0,
                 debug=False):
    from bokeh.models import HoverTool
    yint, good_mets = interpolate_clock_function(new_clock_table,
                                                 clock_offset_table['met'])
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(clock_offset_table[good_mets]['met'],
                    clock_offset_table[good_mets]['offset'] - shift_times)
        plt.scatter(
            clock_offset_table[good_mets]['met'],
            clock_offset_table[good_mets]['offset'] - shift_times + yint)
        plt.plot(new_clock_table['TIME'], -new_clock_table['CLOCK_OFF_CORR'])
        plt.show()

    yint = - yint

    clock_offset_table = filter_and_log_table(
        clock_offset_table,
        good_mets,
        intro_text="plot_scatter: Filtering clock_offset_table to remove points beyond interpolation range",
        comment_to_point="beyond interpolation range",
    )

    clock_offset_table['offset'] -= shift_times
    clock_mets = clock_offset_table['met']
    clock_mjds = clock_offset_table['mjd']
    clock_residuals_detrend = clock_offset_table['offset'] - yint

    good, _ = \
        get_malindi_data_except_when_out(
            clock_offset_table) & ~clock_offset_table['flag']

    roll_std = residual_roll_std(clock_residuals_detrend[good])
    control_points = clock_offset_table['met'][good]

    dates = Time(clock_mjds, format='mjd')

    all_data = pd.DataFrame({'met': clock_mets,
                             'mjd': np.array(clock_mjds, dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'offset': clock_offset_table['offset'],
                             'station': clock_offset_table['station']})
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
        alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station')
    plot_0a = hv.Curve(dict(x=clock_mets, y=yint),
                       group='station', label='Clock corr').opts(color="k")

    plot_0_all = plot_0.opts(opts.Scatter(width=800, height=350, tools=[hover])).opts(
                             ylim=(-0.1, 0.8), xlim=(77571700, None)) * plot_0a

    all_data_res = pd.DataFrame({'met': clock_mets,
                             'mjd': np.array(clock_mjds, dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'residual': clock_residuals_detrend * 1e6,
                             'station': clock_offset_table['station']})

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
        alpha=0.5, muted_line_alpha=0.1,
        muted_fill_alpha=0.03).overlay('station').hist(["residual"])

    plot_1b = hv.Curve({'x': new_clock_table['TIME'],
                        'y': new_clock_table['CLOCK_ERR_CORR'] * 1e6},
                       group='station', label='scatter up').opts(
        opts.Curve(color='k'))
    plot_1a = hv.Curve({'x': new_clock_table['TIME'],
                        'y': -new_clock_table['CLOCK_ERR_CORR'] * 1e6},
                       group='station', label='scatter down').opts(
        opts.Curve(color='k'))

    plot_1_all = plot_1.opts(
        opts.Scatter(width=800, height=350, tools=[hover])).opts(
                             ylim=(-700, 700), xlim=(77571700, None)) * plot_1b * plot_1a

    plots = []
    list_of_stations = sorted(list(set(all_data_res['station'])))
    stats = {}
    for station in list_of_stations:
        print(station)
        nbins = 101
        fit_range = [-500, 1500]
        if station == 'MLD':
            nbins = 501
            fit_range = [-200, 200]
        # from astropy.stats import histogram
        filtered_data = all_data_res[all_data_res['station']==station]['residual']
        edges, frequencies, stats[station] = _analyze_residuals(filtered_data, nbins=nbins, fit_range=fit_range)

        plots.append(hv.Histogram((edges, frequencies), label=station).opts(opts.Histogram(alpha=0.3, xlabel="residual (us)", ylabel="Density")))
    plot_2 = hv.Overlay(plots)

    fine_x = np.linspace(-1000, 1500, nbins * 10 + 1)
    rv = norm(loc=stats["MLD"]["median"], scale=stats["MLD"]["mad"])
    plot_2b = hv.Curve({'x': fine_x,
                        'y': rv.pdf(fine_x)},
                       group='station', label='Raw estimate').opts(
        opts.Curve(color='grey'))

    rv = norm(loc=stats["MLD"]["fit_mean"], scale=stats["MLD"]["fit_std"])
    plot_2c = hv.Curve({'x': fine_x,
                        'y': rv.pdf(fine_x) * stats["MLD"]["fit_norm"]},
                       group='station', label='Fit').opts(
        opts.Curve(color='k'))

    plot_2 = (plot_2 * plot_2b * plot_2c).opts(opts.Overlay(width=800, height=350))
    rolling_data = pd.DataFrame({'met':control_points,
                                 'rolling_std':roll_std*1e6})
    rolling_data.to_pickle('rolling_data.pkl')

    text_top = hv.Div("""
        <p>
        Clock offsets measured at ground station passes throughout
        the mission (scatter points), compared to the thermal model for clock
        delays (line). Different colors indicate different ground stations
        (MLD: Malindi; SNG: Singapore; UHI: US Hawaii).
        <p>Use the tools on the top right to zoom in the plot.</p>
        """).opts(width=500)

    text_bottom = hv.Div("""
        <p>
        Residuals of the clock offsets with respect to the thermal model.
        The black lines indicate the local scatter, calculated over a time
        span of approximately 5 days. The largest spikes indicate pathological
        intervals.</p>
        <p>
        For periods when there are no temperature measurements the code interpolates
        linearly between the raw clock offsets. We arbitrarily assign a 1-ms error to
        these periods, which should account for the typical deviation of the real clock
        delay with respect to the linear interpolation. This results in 1-ms "spikes" in
        the residuals and the rolling averages.
        </p>
        <p>Use the tools on the top right to zoom in the plot.</p>
        """).opts(width=500)

    stat_str = """<p>
        Some statistical information about the residuals. This plot shows the histograms of
        the residuals and a few statistical indicators that might be used to evaluate the
        quality of the fit. The median and the median absolute deviations are used as
        robust proxies for the mean and the standard deviation. This is needed because of the
        many large outliers that alter the naive estimates of the moments of the distributions.
        Then, we fit the core of the distributions (the part cleaner from outliers, which is
        [-150, 200] us for Malindi and [-500, 1500] for the others) with Gaussian curves, using
        the Poisson error of the histograms as weights. The results are listed in the table and
        show in the plot.
        </p>
        <div style="overflow-x:auto;text-align:center;">"""
    stat_str += f"<p>Clock residuals stats (us):</p>\n"
    stat_str += "<table style='width:100%;th.border:1px solid;'>\n"
    stat_str += f"<tr><th>Station</th><th>Median</th><th>MAD</th><th>Fit mean</th><th>Fit STD</th></tr>\n"

    for station, stat in stats.items():
        stat_str += f"<tr><th>{station}</th><td>{stat['median']:.0f}</td><td>{stat['mad']:.0f}</td>"
        stat_str += f"<td>{stat['fit_mean']:.0f}</td><td>{stat['fit_std']:.0f}</td></tr>"
    stat_str += "</table></div>"

    text_stats = hv.Div(f"{stat_str}").opts(width=500)

    return hv.Layout((plot_0_all + text_top) +
                     (plot_1_all + text_bottom) +
                     plot_2 + text_stats).cols(2)


class NuSTARCorr():
    """Apply clock correction to a NuSTAR event file.

    Convenience class that creates a ClockCorrection for a specific
    observation and applies it to the event times.

    Parameters
    ----------
    events_file : str
        Path to input NuSTAR event file (FITS).
    outfile : str, optional
        Path for output corrected file. Default: input with '_tc' suffix.
    adjust : bool, optional
        If True, adjust temperature model using clock offsets. Default True.
    force_divisor : int, optional
        Force a specific clock divisor value.
    temperature_file : str, optional
        Path to temperature data. If None, uses default.
    hdf_dump_file : str, optional
        Path for caching intermediate results.

    Attributes
    ----------
    clock_correction : ClockCorrection
        The underlying correction calculator.
    temperature_correction_fun : callable
        Interpolation function for the correction.
    tstart, tstop : float
        Observation time range (MET).

    Examples
    --------
    >>> corr = NuSTARCorr('nu12345_01_cl.evt')  # doctest: +SKIP
    >>> corr.apply_clock_correction()  # doctest: +SKIP
    'nu12345_01_cl_tc.evt'
    """
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
        """Read TSTART and TSTOP from the event file header."""
        with fits.open(self.events_file) as hdul:
            hdr = hdul[1].header
            self.tstart, self.tstop = hdr['TSTART'], hdr['TSTOP']

    def apply_clock_correction(self):
        """Apply the clock correction to the event times and write a new FITS file.

        Returns
        -------
        str
            Path to the output corrected FITS file.
        """
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
    """Cumulative Simpson's rule integration for equally-spaced data.

    Computes the cumulative integral of y(x) using Simpson's rule,
    returning the integrated value at each sample point.

    Parameters
    ----------
    x : array-like
        Independent variable values (must be equally spaced).
    y : array-like
        Dependent variable values to integrate.

    Returns
    -------
    integral : np.ndarray
        Cumulative integral values at each x point.

    Raises
    ------
    ValueError
        If x and y have different lengths or x is empty.
    """
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
    """Absorption-desorption function for long-term clock drift.

    Models the long-term systematic drift of the TCXO due to material
    outgassing and other aging effects. The drift follows a sum of
    logarithmic terms representing different decay processes.

    Parameters
    ----------
    x : array-like
        Time values in MET (Mission Elapsed Time in seconds).
    b0, b1, b2, b3 : float
        Model coefficients for the two logarithmic components.
    t0 : float, optional
        Reference epoch in MET. Default is ABS_DES_REFERENCE_MET (77509250),
        corresponding to early mission.

    Returns
    -------
    drift : np.ndarray
        Long-term drift contribution in ppm.

    Notes
    -----
    The function computes: b0*ln(b1*t + 1) + b2*ln(b3*t + 1)
    where t is time in years since t0.
    """
    x = (x - t0) / SECONDS_PER_YEAR
    return b0 * np.log(b1 * x + 1) + b2 * np.log(b3 * x + 1)


def clock_ppm_model(nustar_met, temperature, craig_fit=False, version=None, pars=None):
    """Compute the clock frequency deviation model in parts-per-million.

    The NuSTAR TCXO frequency varies with both temperature and time.
    This function combines:
    1. A quadratic temperature dependence around a reference temperature
    2. A long-term drift modeled by absorption-desorption functions

    Parameters
    ----------
    nustar_met : array-like
        Mission Elapsed Time values in seconds.
    temperature : array-like
        TCXO temperature values in Celsius.
    craig_fit : bool, optional
        If True, use the original Craig Markwardt fit parameters.
    version : str, optional
        Model version identifier. If None, uses the latest version.
    pars : dict, optional
        Override model parameters. Expected keys:
        - 'T0': Reference temperature
        - 'offset': Constant ppm offset
        - 'ppm_vs_T_pars': [linear_coeff, quadratic_coeff]
        - 'ppm_vs_time_pars': [b0, b1, b2, b3] for abs_des_fun

    Returns
    -------
    ppm : np.ndarray
        Clock frequency deviation in parts-per-million. Positive values
        mean the clock runs fast; negative means it runs slow.

    See Also
    --------
    abs_des_fun : Long-term drift component
    temperature_delay : Integrates ppm to get time delay
    """
    if craig_fit:
        version = "craig"

    if pars is None:
        pars = get_temperature_parameters(version)

    T0 = pars['T0']
    offset = pars['offset']
    ppm_vs_T_pars = pars['ppm_vs_T_pars']
    ppm_vs_time_pars = pars['ppm_vs_time_pars']

    temp = (temperature - T0)
    ftemp = offset + ppm_vs_T_pars[0] * temp + \
            ppm_vs_T_pars[1] * temp ** 2  # Temperature dependence

    flongterm = abs_des_fun(nustar_met, *ppm_vs_time_pars)

    return ftemp + flongterm


def temperature_delay(temptable, divisor,
                      met_start=None, met_stop=None,
                      debug=False, craig_fit=False,
                      time_resolution=10, version=None, pars=None):
    """Compute cumulative clock delay from temperature-dependent drift.

    Integrates the clock rate correction (from clock_ppm_model) over time
    to get the cumulative time delay. Returns an interpolation function
    that can evaluate the delay at any MET within the range.

    Parameters
    ----------
    temptable : astropy.table.Table
        Temperature table with 'met' and 'temperature_smooth' columns.
    divisor : int
        Clock frequency divisor value (nominally ~24000336).
    met_start, met_stop : float, optional
        Time range for computation. Defaults to temptable range.
    debug : bool, optional
        Reserved for debugging output.
    craig_fit : bool, optional
        Use original Craig Markwardt model parameters.
    time_resolution : float, optional
        Time step in seconds for integration. Default 10s.
    version : str, optional
        Model version identifier.
    pars : dict, optional
        Override model parameters.

    Returns
    -------
    delay_function : scipy.interpolate.PchipInterpolator
        Function that returns cumulative clock delay (seconds) for any MET.
        Can extrapolate beyond the input range.
    """
    table_times = temptable['met']

    if met_start is None:
        met_start = table_times[0]
    if met_stop is None:
        met_stop = table_times[-1]

    temperature = temptable['temperature_smooth']

    temp_fun = PchipInterpolator(
        table_times,
        temperature,
        extrapolate=False)

    dt = time_resolution
    times_fine = np.arange(met_start, met_stop, dt)

    try:
        ppm_mod = clock_ppm_model(times_fine, temp_fun(times_fine),
                                  craig_fit=craig_fit, version=version, pars=pars)
    except:
        error_msg = f"""
        Error in clock_ppm_model:
        Times: {times_fine.min()} - {times_fine.max()}
        Table times: {table_times.min()} - {table_times.max()}
        """
        log.error(error_msg)
        raise

    clock_rate_corr = (1 + ppm_mod / 1000000) * 24000000 / divisor - 1

    delay_sim = simpcumquad(times_fine, clock_rate_corr)
    return PchipInterpolator(times_fine, delay_sim, extrapolate=True)


def temperature_correction_table(met_start, met_stop,
                                 temptable=None,
                                 freqchange_file=None,
                                 hdf_dump_file='dump.hdf5',
                                 force_divisor=None,
                                 time_resolution=0.5,
                                 craig_fit=False,
                                 version=None,
                                 pars=None):
    """Build a table of temperature-based clock corrections.

    This is the main function for computing the temperature-dependent
    clock correction. It:
    1. Loads or reads cached correction data if available
    2. Processes each time interval between frequency divisor changes
    3. Integrates the clock drift to get cumulative delay
    4. Handles gaps in temperature data gracefully

    Parameters
    ----------
    met_start, met_stop : float
        Time range (MET) for the correction table.
    temptable : astropy.table.Table or str, optional
        Temperature table or path to temperature file. If None, uses default.

    Other parameters
    ----------------
    freqchange_file : str, optional
        Path to frequency changes file. If None, uses default.
    hdf_dump_file : str, optional
        Path for caching computed corrections. Set to None to disable caching.
    force_divisor : int, optional
        If set, use this divisor value for entire range instead of
        reading from frequency changes file.
    time_resolution : float, optional
        Output table time step in seconds. Default 0.5s.
    craig_fit : bool, optional
        Use original Craig Markwardt model.
    version : str, optional
        Model version identifier.
    pars : dict, optional
        Override model parameters.

    Returns
    -------
    table : astropy.table.Table
        Correction table with columns:
        - 'met': Mission Elapsed Time
        - 'temp_corr': Clock correction in seconds (add to event times)
        - 'divisor': Clock divisor value at each time
    """
    import six
    if hdf_dump_file is not None and os.path.exists(hdf_dump_file):
        log.info(f"Reading cached data from file {hdf_dump_file}")

        result_table = fix_byteorder(Table.read(hdf_dump_file))
        mets = np.array(result_table['met'])
        if (met_start > mets[10] or met_stop < mets[-20]) and (
                met_stop - met_start < 3 * SECONDS_PER_YEAR):
            log.warning(
                "Interval not fully included in cached data. Recalculating.")
        else:
            good = (mets >= met_start - SECONDS_PER_DAY * 2) & (mets < met_stop + SECONDS_PER_DAY * 2)

            filtered_table = filter_and_log_table(
                result_table,
                good,
                intro_text="temperature_correction_table: Filtering cached result_table to requested MET range",
                comment_to_point="beyond requested range",
                point_log_func=log.debug
            )
            return filtered_table

    if temptable is None or isinstance(temptable, six.string_types):
        mjdstart, mjdstop = sec_to_mjd(met_start), sec_to_mjd(met_stop)
        # Returns table with: 'met', 'temperature', 'mjd', 'temperature_smooth'
        temptable = read_temptable(mjdstart=mjdstart,
                                   mjdstop=mjdstop,
                                   temperature_file=temptable, dt=10)
    if force_divisor is None:
        # Returns table with: 'uxt', 'met', 'divisor', 'mjd', 'flag'
        freq_changes_table = \
            read_freq_changes_table(freqchange_file=freqchange_file)
        allfreqtimes = np.array(freq_changes_table['met'])
        allfreqtimes = \
            np.concatenate([allfreqtimes, [temptable['met'][-1]]])
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

    mean_history = np.mean(temptable["temperature_smooth"])

    for i, met_intv in tqdm.tqdm(enumerate(met_intervals),
                                 total=len(met_intervals)):
        if met_intv[1] < met_start:
            continue
        if met_intv[0] > met_stop:
            break
        if met_intv[0] > met_intv[1]:
            log.warning(f"Invalid interval: {met_intv}")
            continue

        start, stop = met_intv

        tempidx = np.searchsorted(temptable['met'], [start - 20, stop + 20])

        temptable_filt = temptable[tempidx[0]:tempidx[1] + 1]

        times_fine = np.arange(start, stop, time_resolution)

        if len(temptable_filt) < 5:
            log.warning(
                f"Too few temperature points in interval "
                f"{start} to {stop} (MET)")
            # Get an estimate of the mean temperature closest to those dates
            if tempidx[0] == 0:
                ref_t = np.mean(temptable['temperature_smooth'][:20])
                print("Using average temperature at the start of series")
            elif tempidx[1] == len(temptable):
                ref_t = np.mean(temptable['temperature_smooth'][-20:])
                print("Using average temperature at the end of series")
            else:
                print("Using average temperature")
                ref_t = mean_history

            raw_mets = np.arange(start - 20, stop + 20)

            temptable_filt = Table(dict(met=raw_mets, temperature_smooth=ref_t + np.zeros_like(raw_mets)))

        delay_function = \
            temperature_delay(temptable_filt, divisors[i], craig_fit=craig_fit,
                                time_resolution=time_resolution,version=version, pars=pars)

        temp_corr = \
            delay_function(times_fine) + last_corr - delay_function(last_time)
        temp_corr[temp_corr != temp_corr] = 0

        new_data = Table(dict(met=times_fine,
                              temp_corr=temp_corr,
                              divisor=np.zeros_like(times_fine) + divisors[i]))

        N = len(times_fine)
        table[firstidx:firstidx + N] = new_data
        firstidx = firstidx + N

        last_corr = temp_corr[-1]
        last_time = times_fine[-1]

    log.info("Interpolation done.")
    table = table[:firstidx]

    if hdf_dump_file is not None:
        log.info(f"Saving intermediate data to {hdf_dump_file}...")
        table.write(hdf_dump_file, overwrite=True)
        log.info(f"Done.")
    return table


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
    parser.add_argument("--shift-times", default=4.9e-3, type=float,
                        help="Shift times by this amount")
    parser.add_argument("--save-nodetrend",
                        help="Save un-detrended correction in separate FITS "
                             "extension",
                        action='store_true', default=False)
    parser.add_argument("--high-resolution",
                        help="Create a high-resolution file "
                             "(100 times larger)",
                        action='store_true', default=False)

    args = parser.parse_args(args)

    clockcorr = ClockCorrection(temperature_file=args.tempfile,
                                adjust_absolute_timing=True,
                                clock_offset_file=args.offsets,
                                hdf_dump_file=args.cache,
                                freqchange_file=args.frequency_changes)
    clockcorr.write_clock_file(args.outfile,
                               save_nodetrend=args.save_nodetrend,
                               shift_times=args.shift_times,
                               highres=args.high_resolution)

    clock_offset_table = read_clock_offset_table(args.offsets)
    plot = plot_scatter(Table.read(args.outfile, hdu="NU_FINE_CLOCK"),
                        clock_offset_table,
                        shift_times=args.shift_times)
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
    if args.outfile is not None and os.path.exists(args.outfile):
        log.info("Reading existing temperature table")
        existing_table = Table.read(args.outfile)
        last_measurement = existing_table['mjd'][-1] + 0.001 / SECONDS_PER_DAY

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

    outfile = args.outfile
    if args.outfile is None:
        outfile = os.path.splitext(args.tempfile)[0] + '.hdf5'

    log.info(f"Saving to {outfile}")

    new_table.write(outfile, path="temptable", overwrite=True)


def main_plot_diagnostics(args=None):
    import argparse
    import re
    description = ('Plot diagnostic information about the newly produced '
                   'clock file.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("clockcorr", type=str,
                        help="Clock correction file")
    parser.add_argument("clockoff", default=None,
                        help="Clock offset table")

    args = parser.parse_args(args)

    clock_offset_table = read_clock_offset_table(args.clockoff)
    with fits.open(args.clockcorr) as hdul:
        data = hdul["NU_FINE_CLOCK"].data
        header = hdul["NU_FINE_CLOCK"].header
        shift_times = 0.
        for comment in header["COMMENT"]:
            if "A systematic shift" in comment:
                shift_time_re = re.compile(r".*A systematic shift of ([^ ]+) s was applied.*")
                shift_times = float(shift_time_re.match(comment).group(1))
                break
        data = Table(data)
    plot = plot_scatter(data, clock_offset_table, shift_times=shift_times)

    outfig = args.clockcorr.replace(".gz", "").replace(".fits", "")
    renderer = hv.renderer('bokeh')
    renderer.save(plot, outfig)
