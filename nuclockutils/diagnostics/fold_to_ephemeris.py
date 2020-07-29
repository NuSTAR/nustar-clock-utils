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

    t = Table({'phase': np.linspace(0, 1, nbin + 1)[:-1], 'profile': prof_corr, 'profile_raw': prof})
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
    correction_fun = get_phase_from_ephemeris_file(mjdstart, mjdstop, args.parfile)

    del events

    for event_start in tqdm.tqdm(list(range(0, events_time.size, n_phot_max))):
        local_events = events_time[event_start : event_start + n_phot_max]
        if event_start != 0 and local_events.size < n_phot_max:
            break
        elif event_start == 0:
            n_phot_max = min(n_phot_max, local_events.size)

        phase = correction_fun(local_events)

        t = calculate_profile_from_phase(phase, nbin=nbin)

        label = f'{event_start}-{event_start + n_phot_max}'
        if emin is not None or emax is not None:
            label += f'_{emin}-{emax}keV'

        for attr in ['F0', 'F1', 'F2']:
            t.meta[attr] = getattr(ephem, attr).value
        t.meta['epoch'] = ephem.PEPOCH.value
        t.meta['mjd'] = (local_events[0] + local_events[-1]) / 2

        t.write(os.path.splitext(evfile)[0] + f'_{label}.ecsv', overwrite=True)

        fig = plt.figure()
        plt.plot(t['phase'], t['profile'], label='corrected')

        plt.legend()
        plt.savefig(os.path.splitext(evfile)[0] + f'_{label}.png', overwrite=True)
        plt.close(fig)



if __name__ == '__main__':
    main(sys.argv[1:])
