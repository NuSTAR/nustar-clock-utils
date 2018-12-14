from nuclockutils import get_orbital_functions
from nuclockutils import main_barycorr
import os
import numpy as np
import pytest


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6

        self.orbfile = os.path.join(datadir, "dummy_orb.fits")
        self.parfile = os.path.join(datadir, "dummy_par.par")
        self.evfile = os.path.join(datadir, "dummy_evt.evt")

    def test_read_orb_funcs(self):
        res = get_orbital_functions(self.orbfile)

    def test_barycorr(self):
        outfile = main_barycorr([self.evfile, self.orbfile, self.parfile])
        assert os.path.exists(outfile)
        with pytest.raises(RuntimeError):
            main_barycorr([self.evfile, self.orbfile, self.parfile,
                           '-o', outfile])
        main_barycorr([self.evfile, self.orbfile, self.parfile,
                       '-o', outfile, '--overwrite'])

        os.unlink(outfile)
