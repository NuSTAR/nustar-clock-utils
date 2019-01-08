import numpy as np


NUSTAR_MJDREF = np.longdouble("55197.00076601852")


def sec_to_mjd(time, mjdref=NUSTAR_MJDREF, dtype=np.double):
    return np.array(np.asarray(time) / 86400 + mjdref, dtype=dtype)


def splitext_improved(path):
    """
    Examples
    --------
    >>> np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    True
    >>> np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    True
    >>> np.all(splitext_improved("a.f/a.tar") ==  ('a.f/a', '.tar'))
    True
    >>> np.all(splitext_improved("a.a.a.f/a.tar.gz") ==  ('a.a.a.f/a', '.tar.gz'))
    True
    """
    import os
    dir, file = os.path.split(path)

    if len(file.split('.')) > 2:
        froot, ext = file.split('.')[0],'.' + '.'.join(file.split('.')[-2:])
    else:
        froot, ext = os.path.splitext(file)

    return os.path.join(dir, froot), ext


