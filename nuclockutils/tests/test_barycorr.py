from nuclockutils import get_orbital_functions
import os
import numpy as np


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6

        self.orbfile = os.path.join(datadir, "dummy_orb.fits")

    def test_read_orb_funcs(self):
        res = get_orbital_functions(self.orbfile)
