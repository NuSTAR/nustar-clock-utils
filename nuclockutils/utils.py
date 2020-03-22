import numpy as np
from astropy import log


NUSTAR_MJDREF = np.longdouble("55197.00076601852")


def fix_byteorder(table):
    import sys

    sys_byteorder = ('>', '<')[sys.byteorder == 'little']
    for col in table.colnames:
        if table[col].dtype.byteorder not in ('=', sys_byteorder):
            table[col] = table[col].byteswap().newbyteorder(sys_byteorder)
    return table


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


def get_wcs_from_col(hdu, col):
    from astropy.io.fits.column import KEYWORD_TO_ATTRIBUTE

    column = hdu.data.columns[col]
    res = type('wcsinfo', (), {})()
    res.form = getattr(column, KEYWORD_TO_ATTRIBUTE["TFORM"])
    res.crval = getattr(column, KEYWORD_TO_ATTRIBUTE["TCRVL"])
    res.crpix = getattr(column, KEYWORD_TO_ATTRIBUTE["TCRPX"])
    res.cdelt = getattr(column, KEYWORD_TO_ATTRIBUTE["TCDLT"])
    res.ctype = getattr(column, KEYWORD_TO_ATTRIBUTE["TCTYP"])
    res.cunit = getattr(column, KEYWORD_TO_ATTRIBUTE["TCUNI"])
    return res


def get_wcs_from_bintable(hdu, xcol, ycol):
    """Get WCS information from the columns (e.g. X and Y)."""
    from astropy import wcs
    xwcs = get_wcs_from_col(hdu, xcol)
    ywcs = get_wcs_from_col(hdu, ycol)

    w = wcs.WCS(naxis=2)

    w.wcs.crpix = [xwcs.crpix, ywcs.crpix]
    w.wcs.cdelt = np.array([xwcs.cdelt, ywcs.cdelt])
    w.wcs.crval = [xwcs.crval, ywcs.crval]
    w.wcs.ctype = [xwcs.ctype, ywcs.ctype]

    return w


def filter_with_region(evfile, regionfile, debug_plot=True,
                       outfile=None):
    """Filter event file by specifying a fk5 region."""
    from regions import read_ds9
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    label = regionfile.replace('.reg', '')
    root, ext = splitext_improved(evfile)
    if outfile is None:
        outfile = root + f'_{label}' + ext

    if outfile == evfile:
        raise ValueError("Invalid output file")

    log.info(f"Opening file {evfile}")
    with fits.open(evfile) as hdul:
        wcs = get_wcs_from_bintable(hdul['EVENTS'], 'X', 'Y')
        data = hdul['EVENTS'].data
        coords = SkyCoord.from_pixel(data['X'], data['Y'], wcs, mode='wcs')

        log.info(f"Reading region {regionfile}")
        region = read_ds9(regionfile)
        mask = region[0].contains(coords, wcs)
        masked = coords[mask]
        coordsx, coordsy = coords.to_pixel(wcs)
        x, y = masked.to_pixel(wcs)
        hdul['EVENTS'].data = data[mask]
        hdul.writeto(outfile, overwrite=True)
        log.info(f"Saving to file {outfile}")

    if debug_plot:
        import matplotlib.pyplot as plt
        center = region[0].center

        ddec = 0.1
        dra = 0.1 / np.cos(center.dec)
        figurename = f"{root}.png"
        log.info(f"Plotting data in {figurename}")
        fig = plt.figure(figurename, figsize=(10, 10))
        plt.style.use('dark_background')
        noise_ra = np.random.normal(coords.ra.value, 1/60/60) * u.deg
        noise_dec = np.random.normal(coords.dec.value, 1/60/60) * u.deg
        log.info(f"Randomizing scatter points by 1'' for beauty")
        plt.subplot(projection=wcs)
        plt.scatter(noise_ra, noise_dec, s=1, alpha=0.05)
        noise_ra = np.random.normal(masked.ra.value, 1/60/60) * u.deg
        noise_dec = np.random.normal(masked.dec.value, 1/60/60) * u.deg
        plt.scatter(noise_ra, noise_dec, s=1, alpha=0.05)
        plt.xlim([(center.ra - dra * u.deg).value, (center.ra + dra * u.deg).value])
        plt.ylim([(center.dec - ddec * u.deg).value, (center.dec + ddec * u.deg).value])
        plt.xlabel("RA")
        plt.ylabel("Dec")
        plt.grid()
        plt.savefig(figurename)
        plt.close(fig)