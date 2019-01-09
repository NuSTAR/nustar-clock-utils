Documentation
=============

This is the documentation for nustar-clock-utils.
 Utilities for a precise timing calibration of NuSTAR data

Installation
------------

.. code-block::

    $ git clone https://github.com/matteobachetti/nustar-clock-utils
    $ cd nustar-clock-utils
    $ pip install -r requirements.txt
    $ pip install .

Usage
-----
You should have at least a (non-barycentered) event file and some temperature information.
The latter can be found in the engineering housekeeping file (``auxil/nuXXXXXXXXXX_eng.hk.gz``) in recent observations.
Otherwise the SOC can provide a comma-separated file with the temperature information.
There are two ways to correct the clock data, one for legacy data (<2018:10:01) and one for recent data.

Legacy:

.. code-block ::

    $ nustar_tempcorr event_file.evt

Recent-ish:

.. code-block ::

    $ nustar_tempcorr nu101010101010A01_cl.evt -t nu101010101010_eng.hk.gz


Very recent observation, before an update to the clock offset and clock divisor adjustment history:

.. code-block ::

    $ nustar_tempcorr nu101010101010A01_cl.evt -D 24000335 -t nu101010101010_eng.hk.gz --no-adjust

The -D option specifies the clock divisor (the frequency of the quartz oscillator in the spacecraft TCXO).
If unkown, give something around 24000300 and be aware that the clock will be stable but will run fast or slow.

.. toctree::
  :maxdepth: 2

  nuclockutils/index.rst

.. note:: The layout of this directory is simply a suggestion.  To follow
          traditional practice, do *not* edit this page, but instead place
          all documentation for the package inside ``nuclockutils/``.
          You can follow this practice or choose your own layout.
