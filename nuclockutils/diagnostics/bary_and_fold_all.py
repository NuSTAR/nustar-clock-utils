import os
import glob
import re
import traceback
import subprocess as sp

import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy import log
from astropy.io import fits
import pint

from .get_crab_ephemeris import get_crab_ephemeris
from .fold_to_ephemeris import get_events_from_fits, \
    calculate_profile_from_phase, get_phase_from_ephemeris_file, \
    get_ephemeris_from_parfile
from .compare_pulses import main as main_compare_pulses
from nuclockutils.barycorr import main_barycorr as nubarycorr
from nuclockutils.utils import high_precision_keyword_read


def mkdir(folder):
    import errno
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_observing_mjd(fname):
    try:
        hdul = fits.open(fname)
        tstart = hdul[1].header['TSTART']
        tstop = hdul[1].header['TSTOP']
        mjdref = high_precision_keyword_read(hdul[1].header, 'MJDREF')
        hdul.close()
    except astropy.io.fits.verify.VerifyError:
        log.error(f"{fname} might be corrupted")
        return None

    return np.array([tstart, tstop]) / 86400 + mjdref


def fold_file_to_ephemeris(fname, parfile, emin=None, emax=None,
                           nbin=128, outroot="out"):
    mjdstart, mjdstop = get_observing_mjd(fname)
    mjdmean = (mjdstart + mjdstop) / 2
    ephem = get_ephemeris_from_parfile(parfile)

    log.info(f"Calculating correction function between "
             f"{mjdstart} and {mjdstop}...")
    correction_fun = get_phase_from_ephemeris_file(mjdstart, mjdstop, parfile)

    events = get_events_from_fits(fname)
    good = np.ones(events.time.size, dtype=bool)
    if emin is not None:
        good = (events.energy >= emin)
    if emax is not None:
        good = good & (events.energy < emax)

    times = events.time[good]
    event_mjds = times / 86400 + events.mjdref
    phase = correction_fun(event_mjds)

    t = calculate_profile_from_phase(phase, nbin=nbin)

    for attr in ['PEPOCH', 'F0', 'F1', 'F2', 'F3', 'RAJ', 'DECJ']:
        if hasattr(ephem, attr):
            t.meta[attr] = getattr(ephem, attr).value
    t.meta['mjd'] = mjdmean

    t.write(f'{outroot}.ecsv', overwrite=True)
    fig = plt.figure()
    plt.plot(t['phase'], t['profile'], label='corrected')

    plt.legend()
    plt.savefig(f'{outroot}.jpg', dpi=300)
    plt.close(fig)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("fnames", help="Input file names", nargs='+')
    parser.add_argument('--nbin', default=128, type=int,
                        help="Number of bins in the pulse profile")
    parser.add_argument('--emin', default=None, type=float,
                        help="Minimum energy of photons, in keV")
    parser.add_argument('--emax', default=None, type=float,
                        help="Maximum energy of photons, in keV")
    parser.add_argument('-c', '--clockfile', required=True, type=str,
                        help="Clock correction file")
    parser.add_argument('--bary-suffix', default='_bary')
    parser.add_argument('--plot-phaseogram', default=False,
                        action='store_true',
                        help='Plot the phaseogram (requires PINT)')
    parser.add_argument("--expocorr", default=False,
                        action='store_true',
                        help="Calculate the exposure from (NuSTAR only, the"
                             "event file must be unfiltered)")
    parser.add_argument("--use-standard-barycorr", default=False,
                        action='store_true',
                        help="Use standard barycorr instead of nuclockutils")

    args = parser.parse_args(args)

    nu_root_file_re = re.compile(r'^(.*nu[0-9]+[AB])(.*evt.*)$')

    emin = args.emin
    emax = args.emax
    nbin = args.nbin

    outdir = \
        os.path.basename(args.clockfile
                         ).replace('.gz', '').replace('.fits', '')
    for fname in args.fnames:
        basename = os.path.basename(fname)
        try:
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} does not exist")
            matchobj = nu_root_file_re.match(basename)
            if not matchobj:
                raise ValueError(f"File name {fname} is incorrect")
            nu_root = matchobj.group(1)
            outroot = os.path.splitext(basename.replace('.gz', ''))[0]

            parfile = nu_root + '.par'
            mjdstart, mjdstop = get_observing_mjd(fname)
            mjdmean = (mjdstart + mjdstop) / 2
            if not os.path.exists(parfile):
                log.info("Downloading Crab ephemeris")
                get_crab_ephemeris(mjdmean, parfile)

            attorb_file = nu_root + '.attorb'
            attorb_file = glob.glob(attorb_file + '*')[0]

            outroot = os.path.join(outdir, outroot)

            bary_file = outroot + args.bary_suffix + '.evt.gz'
            mkdir(outdir)

            if not os.path.exists(bary_file):
                if args.use_standard_barycorr:
                    model = pint.models.get_model(parfile)
                    ra = model.RAJ.to("degree").value
                    dec = model.DECJ.to("degree").value
                    cmd = f'barycorr {fname} {bary_file} ' \
                          f'orbitfiles={attorb_file} ' \
                          f'clockfile={args.clockfile} ' \
                          f'refframe=ICRS ra={ra} dec={dec}'
                    sp.check_call(cmd.split())
                else:
                    cmd = f'{fname} {attorb_file} -p {parfile} ' \
                          f'-c {args.clockfile} -o {bary_file}'

                    nubarycorr(cmd.split())

            if args.plot_phaseogram:
                cmd = f'photonphase {bary_file} {parfile} ' \
                      f'--plotfile {outroot}_phaseogram.jpg ' \
                      f'--addphase --absphase'
                sp.check_call(cmd.split())

            if emin is not None or emax is not None:
                outroot += f'_{emin}-{emax}keV'

            fold_file_to_ephemeris(
                bary_file, parfile, emin=emin, emax=emax, nbin=nbin,
                outroot=outroot)

        except Exception as e:
            log.error(f"Error processing {fname}")
            tb = traceback.format_exc()
            for line in tb.split('\n'):
                log.error(line)
            continue

    cmd = glob.glob(os.path.join(outdir, '*.ecsv'))
    if len(cmd) > 0:
        cmd += ['--outfile', outdir + '.jpg',
                '--template', 'profile_template.ecsv']
        main_compare_pulses(cmd)


if __name__ == '__main__':
    main(None)
