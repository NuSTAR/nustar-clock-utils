import os
import sys
import pickle

import tqdm
from hendrics.io import load_events_and_gtis
from hendrics.efsearch import _fast_phase_fddot
from hendrics.base import histogram

import numpy as np

from astropy import log
from astropy.table import Table
from astropy.io import fits

import matplotlib.pyplot as plt
from pint import models
from stingray.pulse.pulsar import _load_and_prepare_TOAs, get_model
from scipy.interpolate import interp1d



def get_ephemeris_from_parfile(parfile):
    model = models.get_model(parfile)
    return model


def calculate_profile(events_time, ephem, nbin=256, expo=None):
    phase = _fast_phase_fddot(events_time,
        ephem.F0.value.astype(np.double),
        ephem.F1.value.astype(np.double),
        ephem.F2.value.astype(np.double))
    prof = histogram(phase, bins=nbin, ranges=[0, 1])
    prof_corr = prof

    if expo is not None:
        prof_corr = prof / expo

    t = Table({'phase': np.linspace(0, 1, nbin + 1)[:-1], 'profile': prof_corr, 'profile_raw': prof})
    if expo is not None:
        t['expo'] = expo

    return t


def calculate_profile_from_phase(phase, nbin=256, expo=None):
    phase = phase - np.floor(phase)
    prof = histogram(phase.astype(np.double), bins=nbin, ranges=[0, 1])
    # np.linspace(0, 1, nbin + 1)
    prof_corr = prof

    if expo is not None:
        prof_corr = prof / expo

    t = Table({'phase': np.linspace(0, 1, nbin + 1)[:-1] + 0.5 / nbin,
               'profile': prof_corr, 'profile_raw': prof})
    if expo is not None:
        t['expo'] = expo

    return t


def get_events_from_fits(evfile):
    log.info(f"Opening file {evfile}")
    res = load_events_and_gtis(evfile)
    events = res.ev_list
    if events.mission == 'nustar':
        events.energy = events.pi * 0.04 + 1.6
    elif events.mission == 'nicer':
        events.energy = events.pi * 0.01
    elif events.mission == 'xmm':
        events.energy = events.cal_pi * 0.001

    return events


def prepare_TOAs(mjds, ephem):
    toalist = _load_and_prepare_TOAs(mjds, ephem=ephem)

    toalist.clock_corr_info['include_bipm'] = False
    toalist.clock_corr_info['include_gps'] = False
    return toalist


# def get_expo_kristin(evfile, nbin, n_phot_max=1000000):
#     with fits.open(evfile) as hdul:
#         times = np.array(hdul[1].data['TIME'][:n_phot_max])
#         priors = np.array(hdul[1].data['PRIOR'][:n_phot_max])
#         mjdref = hdul[1].header['MJDREFF'] + hdul[1].header['MJDREFI']
#
#         for i in range(times.size):
#             istart = np.floor((lcDT[i] - evt[i].prior) / (nbin * p0))
#             iend = np.floor(lcDT[i] / (nbin * p0))
#             if (istart lt 0) then begin
#             livetime[0:iend] += 1
#             livetime[200 + istart: *] += 1
#
#
#     endif
#     if (istart ge 0) then begin
#     if (iend eq 200) then begin
#     iend = 199
#     livetime[0] += 1
#     endif
#     livetime[istart:iend] += 1
#     endif
#     endfor
def get_expo_corr(evfile, parfile, nbin=128, n_phot_max=None):
    cache_file = os.path.basename(
            os.path.splitext(evfile)[0]) + f'_expo_{nbin}.pickle'
    if cache_file is not None and os.path.exists(cache_file):
        log.info("Using cached exposure information")
        return pickle.load(open(cache_file, 'rb'))
    if n_phot_max is None:
        n_phot_max = 500 * nbin

    # ephem = get_model(parfile)

    with fits.open(evfile) as hdul:
        times = np.array(hdul[1].data['TIME'][:n_phot_max])
        priors = np.array(hdul[1].data['PRIOR'][:n_phot_max])
        mjdref = hdul[1].header['MJDREFF'] + hdul[1].header['MJDREFI']

    hdul.close()

    expo = get_exposure_per_bin(times, priors, parfile,
                         nbin=nbin, cache_file=cache_file, mjdref=mjdref)
    return expo


def get_exposure_per_bin(event_times, event_priors, parfile,
                         nbin=16, cache_file=None, mjdref=0):
    """
    Examples
    --------
    # >>> event_times = np.arange(1, 10, 0.01)
    # >>> event_priors = np.zeros_like(event_times)
    # >>> event_priors[0] = 1
    # >>> prof = get_exposure_per_bin(event_times, event_priors, 1/10, nbin=10)
    # >>> prof[0]
    # 0
    # >>> np.allclose(prof[1:], 1)
    # True
    """
    log.info("Calculating exposure")

    if cache_file is not None and os.path.exists(cache_file):
        log.info("Using cached exposure information")
        return pickle.load(open(cache_file, 'rb'))

    start_live_time = event_times - event_priors

    m = get_model(parfile)

    sampling_time = 1 / m.F0.value / nbin / 7.1234514351515132414
    chunk_size = np.min((sampling_time * 50000000, 1000))
    mjdstart = (event_times[0] - 10) / 86400 + mjdref
    mjdstop = (event_times[-1] + 10) / 86400 + mjdref

    phase_correction_fun = get_phase_from_ephemeris_file(mjdstart, mjdstop,
                                                         parfile)

    prof_all = 0
    prof_dead = 0
    for start_time in tqdm.tqdm(np.arange(start_live_time[0], event_times[-1], chunk_size)):
        sample_times = np.arange(start_time, start_time + chunk_size, sampling_time)

        idxs_live = np.searchsorted(start_live_time, sample_times, side='right')
        idxs_dead = np.searchsorted(event_times, sample_times, side='right')

        dead = idxs_live == idxs_dead

        sample_times = sample_times / 86400 + mjdref
        # phases_all = _fast_phase_fddot(sample_times, freq, fdot, fddot)
        # phases_dead = _fast_phase_fddot(sample_times[dead], freq, fdot, fddot)
        phases_all = phase_correction_fun(sample_times)
        # phases_dead = phase_correction_fun(sample_times[dead])
        phases_all = phases_all - np.floor(phases_all)
        phases_dead = phases_all[dead]

        prof_all += histogram(phases_all.astype(np.double), bins=nbin, ranges=[0, 1])
        prof_dead += histogram(phases_dead.astype(np.double), bins=nbin, ranges=[0, 1])

    expo = (prof_all - prof_dead) / prof_all

    if cache_file is not None:
        pickle.dump(expo, open(cache_file, 'wb'))

    return expo


def get_phase_from_ephemeris_file(mjdstart, mjdstop, parfile,
                                  ntimes=1000, ephem="DE405",
                                  return_pint_model=False):
    """Get a correction for orbital motion from pulsar parameter file.

    Parameters
    ----------
    mjdstart, mjdstop : float
        Start and end of the time interval where we want the orbital solution
    parfile : str
        Any parameter file understood by PINT (Tempo or Tempo2 format)

    Other parameters
    ----------------
    ntimes : int
        Number of time intervals to use for interpolation. Default 1000

    Returns
    -------
    correction_mjd : function
        Function that accepts times in MJDs and returns the deorbited times.
    """
    from astropy import units

    mjds = np.linspace(mjdstart, mjdstop, ntimes)

    toalist = prepare_TOAs(mjds, ephem)
    m = get_model(parfile)
    phase_int, phase_frac = np.array(m.phase(toalist, abs_phase=True))
    phases = phase_int + phase_frac

    correction_mjd_rough = \
        interp1d(mjds, phases,
                  fill_value="extrapolate")
    return correction_mjd_rough


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("fname", help="Input file name")
    parser.add_argument("parfile",
        help=("Parameter file name for ephemeris. If Crab, it gets downloaded "
              "automatically from the Jodrell bank Crab monthly ephemeris"))

    parser.add_argument('--nbin', default=512, type=int,
                        help="Number of bins in the pulse profile")
    parser.add_argument('--emin', default=3, type=float,
                        help="Minimum energy of photons, in keV")
    parser.add_argument('--emax', default=10, type=float,
                        help="Maximum energy of photons, in keV")
    parser.add_argument('--n-phot-max', default=None, type=int,
                        help="Maximum number of photons")
    parser.add_argument("--expocorr", default=None, type=str,
                        help="Unfiltered file to calculate the exposure from "
                             "(NuSTAR only)")

    args = parser.parse_args(args)

    evfile = args.fname
    emin = args.emin
    emax = args.emax
    nbin = args.nbin
    n_phot_max = args.n_phot_max

    n_phot_per_bin = 1000
    if n_phot_max is None:
        n_phot_max = n_phot_per_bin * nbin

    ephem = get_ephemeris_from_parfile(args.parfile)
    expo = None
    if args.expocorr is not None:
        if args.expocorr.endswith('pickle'):
            expo = pickle.load(open(args.expocorr, 'rb'))
            if expo.size < nbin:
                log.warning("Interpolating exposure. Use at your own risk.")
        else:
            expo = get_expo_corr(
                args.expocorr, args.parfile,
                nbin=nbin * 4, n_phot_max=300 * 4 * nbin)
        if expo.size != nbin:
            expo_fun = interp1d(
                np.linspace(0, 1, expo.size + 1) + 0.5 / expo.size,
                np.concatenate((expo, [expo[0]])))
            expo = expo_fun(
                np.linspace(0, 1, nbin + 1)[:nbin] + 0.5 / nbin)
    log.info(f"Loading {evfile}")
    events = get_events_from_fits(evfile)

    MJDREF = events.mjdref
    MJD = events.time.mean() / 86400 + MJDREF

    good = np.ones(events.time.size, dtype=bool)
    if emin is not None:
        good = (events.energy >= emin)
    if emax is not None:
        good = good & (events.energy < emax)

    events_time = events.time[good] / 86400 + MJDREF
    mjdstart = events_time[0] - 0.01
    mjdstop = events_time[-1] + 0.01

    log.info("Calculating correction function...")
    correction_fun = \
        get_phase_from_ephemeris_file(mjdstart, mjdstop, args.parfile)

    del events

    for event_start in tqdm.tqdm(list(range(0, events_time.size, n_phot_max))):
        local_events = events_time[event_start : event_start + n_phot_max]
        if event_start != 0 and local_events.size < n_phot_max:
            break
        elif event_start == 0:
            n_phot_max = min(n_phot_max, local_events.size)

        phase = correction_fun(local_events)

        t = calculate_profile_from_phase(phase, nbin=nbin, expo=expo)

        label = f'{event_start}-{event_start + n_phot_max}'
        if emin is not None or emax is not None:
            label += f'_{emin}-{emax}keV'
        if args.expocorr:
            label += '_deadcorr'

        for attr in ['F0', 'F1', 'F2']:
            t.meta[attr] = getattr(ephem, attr).value
        t.meta['epoch'] = ephem.PEPOCH.value
        t.meta['mjd'] = (local_events[0] + local_events[-1]) / 2

        t.write(os.path.splitext(evfile)[0] + f'_{label}.ecsv', overwrite=True)

        fig = plt.figure()
        plt.title(f"MJD {t.meta['mjd']}")
        phase = np.concatenate((t['phase'], t['phase'] + 1))
        profile = np.concatenate((t['profile'], t['profile']))
        plt.plot(phase, profile, label='corrected')
        if args.expocorr:
            profile_raw = np.concatenate((t['profile_raw'], t['profile_raw']))
            expo = np.concatenate((t['expo'], t['expo']))

            plt.plot(phase, profile_raw, alpha=0.5, label='raw')
            plt.plot(phase, expo * np.max(t['profile']),
                     alpha=0.5, label='expo')
            plt.legend()
        plt.savefig(os.path.splitext(evfile)[0] + f'_{label}.png', overwrite=True)
        plt.close(fig)



if __name__ == '__main__':
    main(sys.argv[1:])
