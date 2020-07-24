import sys
from astropy.time import Time
from astropy.table import Table
import matplotlib.pyplot as plt
from uncertainties import ufloat
import numpy as np
from statsmodels.robust import mad

from .fftfit import fftfit


def format_for_plotting(phase, prof, period_ms):
    phase_time = phase * period_ms
    phase_time_plot = np.concatenate((phase_time, phase_time + period_ms))
    prof_plot = np.concatenate((prof, prof))
    prof_plot -= np.median(prof)
    return phase_time_plot, prof_plot


def format_profile_and_get_phase(file, template=None):
    print(file)
    table = Table.read(file)
    mjd = table.meta['mjd']
    if 'epoch' in table.meta:
        pepoch = table.meta['epoch']
    else:
        pepoch = table.meta['PEPOCH']

    freq = table.meta['F0']
    fdot = table.meta['F1']
    fddot = table.meta['F2']
    delta_t = (mjd - pepoch) * 86400
    f0 = freq + fdot * delta_t + 0.5 * fddot * delta_t ** 2

    period_ms = 1 / f0 * 1000
    phase = table['phase']

    prof = table['profile'] / np.max(table['profile'])
    local_max = phase[np.argmax(prof)]

    if template is not None:
        # plt.axvline(local_max, color='k', lw=0.5, alpha=alpha)
        mean_amp, std_amp, phase_res, phase_res_err = \
            fftfit(prof, template=template)
    else:
        phase_res = local_max
        phase_res_err = 1 / prof.size

    return mjd, phase, prof, phase_res, phase_res_err, period_ms



def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("fnames", help="Input file names", nargs='+')

    parser.add_argument("--outfile", default=None, type=str,
                        help="Output image file")
    parser.add_argument("--template", default=None, type=str,
                        help="Template pulse profile")

    args = parser.parse_args(args)

    files = args.fnames

    # tables = dict([(f, Table.read(f)) for f in files])

    maxs = []
    phases = []
    phase_errs = []
    ref_max = 0
    mjds = []
    plt.figure(figsize=(6, 9))
    ref_profile = None
    if args.template is not None:
        mjd, phase, prof, _, _, period_ms = \
            format_profile_and_get_phase(args.template, template=None)
        phase_time_plot, prof_plot = format_for_plotting(phase, prof, period_ms)
        local_max = phase[np.argmax(prof)] * period_ms

        plt.plot(phase_time_plot, prof_plot,
            drawstyle='steps-mid', label=args.template, color='k')
        ref_profile = prof
        ref_max = local_max

    if ref_profile is None:
        log.warning("No template provided; using maxima for phase calculation")
    for i, f in enumerate(files):
        mjd, phase, prof, phase_res, phase_res_err, period_ms = \
            format_profile_and_get_phase(f, template=ref_profile)

        phase_time_plot, prof_plot = format_for_plotting(phase, prof, period_ms)

        local_max = phase[np.argmax(prof)] * period_ms

        phases.append(phase_res * period_ms)
        phase_errs.append(phase_res_err * period_ms)
        mjds.append(mjd)
        maxs.append(local_max)

        plt.plot(phase_time_plot, (i + 1) * 0.2 + prof_plot, drawstyle='steps-mid', label=f, alpha=0.5, color='grey')
        for plot_shift in [0, period_ms, 2 * period_ms]:
            plt.scatter(plot_shift + phase_res * period_ms, (i + 1) * 0.2, s=10, color='b')

    mjds = np.array(mjds)
    maxs = np.array(maxs) - ref_max
    phases = np.array(phases)
    phase_errs = np.array(phase_errs)
    # print(mjds.size, phases.size, phase_errs.size)

    fit_max = np.mean(phases)

    if maxs.size > 30:
        phase_err_sqr = np.sqrt(np.sum(phase_errs ** 2)) / phase_errs.size
        tot_err = mad(phases)
    else:
        phase_err_sqr = np.sqrt(np.sum(phase_errs ** 2)) / phase_errs.size
        tot_err = np.sqrt(np.var(phases) + phase_err_sqr ** 2)

    plt.xlabel("Time (ms)")
    plt.ylabel("Flux (arbitrary units)")
    shift = ufloat(np.mean(maxs), np.std(maxs))
    fit_shift = ufloat(fit_max, tot_err)
    # print(f"Fitted Mean shift = {fit_shift}" + " ms")
    # print(f"Mean shift = {shift}" + " ms")
    plt.title(f"Mean shift = {fit_shift}" + " ms")
    plt.xlim([0, period_ms * 2])
    plt.ylim([-0.1, None])
    plt.axvline(ref_max, alpha=0.5, color='b')
    for t0 in [0, period_ms, 2 * period_ms]:
        plt.axvspan(t0 + fit_max - tot_err, t0 + fit_max + tot_err, alpha=0.5, color='red')

    if len(maxs) <= 5:
        plt.legend()
    plt.tight_layout()
    if args.outfile is not None:
        plt.savefig(args.outfile)

    result_table = Table({'mjd': mjds, 'shift': phases, 'err': phase_errs})
    result_table.write('fit_results.ecsv')

    plt.figure()
    plt.errorbar(mjds, phases, phase_errs, fmt='o')
    times = Time(mjds, format='mjd')
    plt.show()

if __name__ == "__main__":
    main()


