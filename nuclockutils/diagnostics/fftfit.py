"""Use FFT techniques to align a template with a pulse profile.

This should be accompanied by test_fftfit.py, which uses hypothesis to check
its reliability. If not, don't edit this, you don't have the git version.

Anne Archibald, 2020

"""
from __future__ import division

import numpy as np
import scipy.optimize
import scipy.stats


class FFTFITResult:
    pass


def wrap(a):
    """Wrap a floating-point number or array to the range -0.5 to 0.5."""
    return (a + 0.5) % 1 - 0.5


def zap_nyquist(profile):
    if len(profile) % 2:
        return profile
    else:
        c = np.fft.rfft(profile)
        c[-1] = 0
        return np.fft.irfft(c)


def vonmises_profile(kappa, n, phase=0):
    """Generate a profile based on a von Mises distribution.
    The von Mises distribution is a cyclic analogue of a Gaussian distribution. The width is
    specified by the parameter kappa, which for large kappa is approximately 1/(2*pi*sigma**2).
    """
    return np.diff(
        scipy.stats.vonmises(kappa).cdf(
            np.linspace(-2 * np.pi * phase, 2 * np.pi * (1 - phase), n + 1)
        )
    )


def upsample(profile, factor):
    """Produce an up-sampled version of a pulse profile.
    This uses a Fourier algorithm, with zero in the new Fourier coefficients.
    """
    output_len = len(profile) * factor
    if output_len % 2:
        raise ValueError("Cannot cope with odd output profile lengths")
    c = np.fft.rfft(profile)
    output_c = np.zeros(output_len // 2 + 1, dtype=complex)
    output_c[: len(c)] = c * factor
    output = np.fft.irfft(output_c)
    assert len(output) == output_len
    return output


def shift(profile, phase):
    """Shift a profile in phase.
    This is a shift towards later phases - if your profile has a 1 in bin zero
    and apply a phase shift of 1/4, the 1 will now be in bin n/4.  If the
    profile has even length, do not modify the Nyquist component.
    """
    c = np.fft.rfft(profile)
    if len(profile) % 2:
        c *= np.exp(-2.0j * np.pi * phase * np.arange(len(c)))
    else:
        c[:-1] *= np.exp(-2.0j * np.pi * phase * np.arange(len(c) - 1))
    return np.fft.irfft(c, len(profile))


def irfft_value(c, phase, n=None):
    """Evaluate the inverse real FFT at a particular position.
    If the phase is one of the usual grid points the result will agree with
    the results of `np.fft.irfft` there.
    No promises if n is small enough to imply truncation.
    """
    natural_n = (len(c) - 1) * 2
    if n is None:
        n = natural_n
    phase = np.asarray(phase)
    s = phase.shape
    phase = np.atleast_1d(phase)
    c = np.array(c)
    c[0] /= 2
    if n == natural_n:
        c[-1] /= 2
    return (
        (
            c[:, None]
            * np.exp(2.0j * np.pi * phase[None, :] * np.arange(len(c))[:, None])
        )
        .sum(axis=0)
        .real
        * 2
        / n
    ).reshape(s)


def fftfit_full(
    template, profile, compute_scale=True, compute_uncertainty=True, std=None
):
    # We will upsample the cross-correlation function to ensure
    # that the highest peak is not missed
    upsample = 8

    t_c = np.fft.rfft(template)
    if len(template) % 2 == 0:
        t_c[-1] = 0
    p_c = np.fft.rfft(profile)
    if len(profile) % 2 == 0:
        p_c[-1] = 0
    n_c = min(len(t_c), len(p_c))
    t_c = t_c[:n_c]
    p_c = p_c[:n_c]

    ccf_c = np.conj(t_c).copy()
    ccf_c *= p_c
    ccf_c[0] = 0
    n_long = 2 ** int(np.ceil(np.log2(2 * (n_c - 1) * upsample)))
    ccf = np.fft.irfft(ccf_c, n_long)
    i = np.argmax(ccf)
    assert ccf[i] >= ccf[(i - 1) % len(ccf)]
    assert ccf[i] >= ccf[(i + 1) % len(ccf)]
    x = i / len(ccf)
    l, r = x - 1 / len(ccf), x + 1 / len(ccf)

    def gof(x):
        return -irfft_value(ccf_c, x, n_long)

    res = scipy.optimize.minimize_scalar(
        gof, bounds=(l, r), method="Bounded", options=dict(xatol=1e-5 / n_c)
    )
    if not res.success:
        raise ValueError("FFTFIT failed: %s" % res.message)
    # assert gof(res.x) <= gof(x)
    r = FFTFITResult()
    r.shift = res.x % 1

    if compute_scale or compute_uncertainty:
        # shifted template corefficients
        s_c = t_c * np.exp(-2j * np.pi * np.arange(len(t_c)) * r.shift)
        assert len(s_c) == len(p_c)
        n_data = 2 * len(s_c) - 1
        a = np.zeros((n_data, 2))
        b = np.zeros(n_data)
        a[0, 1] = len(template)
        a[0, 0] = s_c[0].real
        b[0] = p_c[0].real
        b[1 : len(p_c)] = p_c[1:].real
        b[len(p_c) :] = p_c[1:].imag
        a[1 : len(s_c), 0] = s_c[1:].real
        a[len(s_c) :, 0] = s_c[1:].imag

        lin_x, res, rk, s = scipy.linalg.lstsq(a, b)
        assert lin_x.shape == (2,)

        r.scale = lin_x[0]
        r.offset = lin_x[1]

        if compute_uncertainty:
            if std is None:
                resid = (
                    r.scale * shift(template, r.shift) + r.offset - profile
                )
                std = np.sqrt(np.mean(resid ** 2))

            J = np.zeros((2 * len(s_c) - 2, 2))
            J[: len(s_c) - 1, 0] = (
                -r.scale * 2 * np.pi * s_c[1:].imag * np.arange(1, len(s_c))
            )
            J[len(s_c) - 1 :, 0] = (
                r.scale * 2 * np.pi * s_c[1:].real * np.arange(1, len(s_c))
            )
            J[: len(s_c) - 1, 1] = s_c[1:].real
            J[len(s_c) - 1 :, 1] = s_c[1:].imag
            cov = scipy.linalg.inv(np.dot(J.T, J))
            assert cov.shape == (2, 2)
            # FIXME: std is per data point, not per real or imaginary
            # entry in s_c; check conversion
            r.uncertainty = std * np.sqrt(len(profile) * cov[0, 0] / 2)
            r.cov = cov

    return r


def fftfit_basic(template, profile):
    """Compute the phase shift between template and profile

    We should have fftfit_basic(template, shift(template, s)) == s
    """
    r = fftfit_full(template, profile, compute_scale=False, compute_uncertainty=False)
    return r.shift


def fftfit(profile, template):
    res = fftfit_full(template, profile)
    return res.scale, 0, normalize_phase_0d5(res.shift), res.uncertainty


def normalize_phase_0d5(phase):
    """Normalize phase between -0.5 and 0.5

    Examples
    --------
    >>> normalize_phase_0d5(0.5)
    0.5
    >>> normalize_phase_0d5(-0.5)
    0.5
    >>> normalize_phase_0d5(4.25)
    0.25
    >>> normalize_phase_0d5(-3.25)
    -0.25
    """
    while phase > 0.5:
        phase -= 1
    while phase <= -0.5:
        phase += 1
    return phase
