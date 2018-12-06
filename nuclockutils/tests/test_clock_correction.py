from __future__ import absolute_import, division, print_function
import numpy as np
import os
import pytest
from nuclockutils import read_clock_offset_table, read_freq_changes_table
from nuclockutils import read_temptable, apply_clock_correction

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6
        self.times = np.arange(self.t0, self.t1, 0.01)
        self.clockfile = os.path.join(datadir, "dummy_clk.fits")
        self.tempfile = os.path.join(datadir, "dummy_temp.csv")
        self.orbfile = os.path.join(datadir, "dummy_orb.fits")

    def test_read_clock_offset_file(self):
        table = read_clock_offset_table()
        assert np.any(table['flag'])

    def test_read_freq_ch_table(self):
        tablefilt = read_freq_changes_table(filter_bad=True)
        table = read_freq_changes_table(filter_bad=False)
        assert len(tablefilt) < len(table)

    def test_read_temptables(self):
        table = read_temptable(mjdstart=self.t0, mjdstop=self.t1)
        table = read_temptable(temperature_file=self.tempfile,
                               mjdstart=self.t0, mjdstop=self.t1)

    def test_fun_from_file(self):
        outfile = apply_clock_correction(self.orbfile, outfile=None)
        assert os.path.exists(outfile)

