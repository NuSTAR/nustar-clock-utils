import sys
from astropy.time import Time
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from nuclockutils.nustarclock import interpolate_clock_function


def get_todays_met():
    t = Time.now()
    met = (t.mjd - 55197.00076601852) * 86400
    return met

def get_launch_met():
    t = Time('2012-06-12')
    met = (t.mjd - 55197.00076601852) * 86400
    return met


def show_diff(files, MET1, MET2):

    mets = np.linspace(MET1, MET2, 30000)
    ref_corr = 0

    for i, fname in enumerate(files):
        table = Table.read(fname)
        good_corr, good_mets = interpolate_clock_function(table, mets)
        corr = np.zeros_like(mets)
        corr[good_mets] = good_corr
        if i == 0:
            ref_corr = corr
            good = good_mets
            plt.title(f"Reference: {fname}")
            continue
        good = good & good_mets

        plt.plot(mets, 1e3 * (corr - ref_corr), label=fname)
    plt.xlabel("MET")
    plt.ylabel("Difference (ms)")
    plt.legend()


def show_all(files, MET1, MET2):
    mets = np.linspace(MET1, MET2, 30000)

    good = np.ones(len(mets), dtype=bool)
    for i, fname in enumerate(files):
        table = Table.read(fname)
        good_corr, good_mets = interpolate_clock_function(table, mets)
        corr = np.zeros_like(mets)
        corr[good_mets] = good_corr

        good = good & good_mets
        plt.plot(mets, corr, label=fname)
    plt.xlabel("MET")
    plt.ylabel("Correction (ms)")
    plt.legend()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("fnames", help="Input file names", nargs='+')
    parser.add_argument('--metmin', default=None, type=float,
                        help="Minimum energy of photons, in keV")
    parser.add_argument('--metmax', default=None, type=float,
                        help="Maximum energy of photons, in keV")

    args = parser.parse_args(args)

    if args.metmin is None:
        args.metmin = get_launch_met()
    if args.metmax is None:
        args.metmax = get_todays_met()

    plt.figure("DIFF")
    show_diff(args.fnames, args.metmin, args.metmax)
    plt.figure("ALL")
    show_all(args.fnames, args.metmin, args.metmax)

    plt.show()
