import numpy as np


NUSTAR_MJDREF = np.longdouble("55197.00076601852")


def sec_to_mjd(time, mjdref=NUSTAR_MJDREF, dtype=np.double):
    return np.array(time / 86400 + mjdref, dtype=dtype)


def splitext_improved(path):
    """
    Examples
    --------
    >>> np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    True
    >>> np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    True
    """
    from os.path import splitext
    if len(path.split('.')) > 2:
        return path.split('.')[0],'.' + '.'.join(path.split('.')[-2:])
    return splitext(path)


