from nuclockutils.nustarclock import main_update_temptable
import numpy as np
import os
from astropy.table import Table

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
        self.tempfile0 = os.path.join(datadir, "dummy_temp_small.csv")
        self.tempfile1 = os.path.join(datadir, "dummy_temp.csv")
        self.outfile = 'blablabla.hdf5'

    def test_prepare(self):
        main_update_temptable([self.tempfile0, '-o', self.outfile])

    def test_up_to_date(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        main_update_temptable([self.tempfile0, '-o', self.outfile])

        assert np.any(["is up to date" in rec.message
                       for rec in caplog.records])

    def test_read_temptables(self):
        main_update_temptable([self.tempfile1, '-o', self.outfile])
        assert len(Table.read(self.tempfile1, data_start=2)) == len(Table.read(self.outfile))

    @classmethod
    def teardown_class(self):
        cleanup_hdf5()
