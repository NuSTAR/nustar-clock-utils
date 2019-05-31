from __future__ import absolute_import, division, print_function
import numpy as np
import os
import pytest
from nuclockutils import read_clock_offset_table, read_freq_changes_table
from nuclockutils import read_temptable, main_tempcorr, main_create_clockfile

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


def cleanup_hdf5():
    import glob
    hdf5_files = glob.glob('*.hdf5')
    for h5 in hdf5_files:
        os.unlink(h5)

class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6
        self.times = np.arange(self.t0, self.t1, 0.01)
        self.clockfile = os.path.join(datadir, "dummy_clk.fits")
        self.tempfile = os.path.join(datadir, "dummy_temp.csv")
        self.tempfits = os.path.join(datadir, "dummy_eng.hk")
        self.orbfile = os.path.join(datadir, "dummy_orb.fits")

    def test_read_clock_offset_file(self):
        table = read_clock_offset_table()
        assert np.any(table['flag'])
        cleanup_hdf5()

    def test_read_freq_ch_table(self):
        tablefilt = read_freq_changes_table(filter_bad=True)
        table = read_freq_changes_table(filter_bad=False)
        assert len(tablefilt) < len(table)
        cleanup_hdf5()

    @pytest.mark.remote_data
    def test_read_temptables(self):
        table = read_temptable(mjdstart=self.t0, mjdstop=self.t1)
        table = read_temptable(temperature_file=self.tempfile,
                               mjdstart=self.t0, mjdstop=self.t1)
        cleanup_hdf5()

    @pytest.mark.remote_data
    def test_fun_from_file(self):
        outfile = main_tempcorr([self.orbfile])
        assert os.path.exists(outfile)
        cleanup_hdf5()

    def test_command_line(self):
        outfile = main_tempcorr([self.orbfile, '--no-adjust',
                                 '--force-divisor', '24000000',
                                 '-t', self.tempfits])
        assert os.path.exists(outfile)
        cleanup_hdf5()
