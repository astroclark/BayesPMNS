#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
waveform_parser.py

Module to retrieve waveform data
"""

import os, sys
import glob

# ________________ - Class Definitions - ________________ #

class WaveData:
    """
    Object with details of available waveforms, locations, names etc
    """

    def __init__(self, eos=None, mass=None, viscosity=None):
        """
        Initialise to get all available EOS, masses.  User can then query the
        ojbect to retrieve pertinent information on any waveform, specified by
        EOS, masses and simulation type
        """

        self.eos = eos
        self.mass = mass
        self.viscosity = viscosity

        #
        # Get the waveform data location
        #

        try:
            wavedata_path = os.environ['PMNSPY_PREFIX'] + "/waveform_data"
        except KeyError:
            print >> sys.stderr, "PMNSPATH environment variable not" \
                    " set, please check env"

        # All the data files (NOTE that this will exclude the special cases due
        # to the directory structure):
        self.all_data_files = glob.glob(wavedata_path+'/*/*dat')

        # Parse the file names
        self.waves = self.parse_names(self.all_data_files)
        self.waves = self.select_wave(eos, mass, viscosity)

        self.nwaves = len(self.waves)


    def parse_names(self, all_data_files):
        """ 
        Parse the filenames.  This assumes a filename of the standard form:
        secderivqpoles_16384Hz_eos_mass_viscosity.dat,  returns
        """
        waves = []

        for data_file in all_data_files:

            wavename = data_file.split('/')[-1]
            wavename = wavename.replace('secderivqpoles_16384Hz_','')
            wavename = wavename.replace('.dat','')

            this_wave = dict()

            # Now walk backwards through the waveform label.  This makes it
            # easier to handle EOS names with underscores...

            # viscosity
            if wavename.endswith('lessvisc'):
                this_wave['viscosity'] = 'lessvisc'
            elif wavename.endswith('oldvisc'): 
                this_wave['viscosity'] = 'oldvisc'
            else:
                print >> sys.stderr, "ERROR: viscosity not labelled in waveform"
                sys.exit(-1)
            wavename = wavename.replace('_'+this_wave['viscosity'], '')

            # mass
            this_wave['mass'] = wavename.split('_')[-1]

            # eos
            this_wave['eos'] = wavename.replace('_'+this_wave['mass'], '')

            this_wave['data'] = data_file

            waves.append(this_wave)

        return waves


    # ----------------------------------------------------------------------
    def select_wave(self, eos=None, mass=None, viscosity=None):
        """
        Return a reduced list of waves corresponding to the specified EOS, mass
        and viscosity
        """

        # Use cases:
        #   1) eos, mass, visc
        #   2) eos, mass
        #   3) eos, visc
        #   4) eos
        #   5) mass, visc 
        #   6) mass
        #   7) visc
        
        #   1) eos, mass, visc
        if eos is not None and mass is not None and viscosity is not None:
            selected_waves = self._select_eos(self.waves, eos=eos)
            selected_waves = self._select_mass(selected_waves, mass=mass)
            selected_waves = self._select_viscosity(selected_waves,
                    viscosity=viscosity)

        #   2) eos, mass
        elif eos is not None and mass is not None and viscosity is None:
            selected_waves = self._select_eos(self.waves, eos=eos)
            selected_waves = self._select_mass(selected_waves, mass=mass)

        #   3) eos, viscosity
        elif eos is not None and mass is None and viscosity is not None:
            selected_waves = self._select_eos(self.waves, eos=eos)
            selected_waves = self._select_viscosity(selected_waves,
                    viscosity=viscosity)

        #   4) eos
        elif eos is not None and mass is None and viscosity is None:
            selected_waves = self._select_eos(self.waves, eos=eos)

        #   5) mass, visc 
        elif eos is None and mass is not None and viscosity is not None:
            selected_waves = self._select_mass(self.waves, mass=mass)
            selected_waves = self._select_viscosity(selected_waves,
                    viscosity=viscosity)

        #   6) mass
        elif eos is None and mass is not None and viscosity is None:
            selected_waves = self._select_mass(self.waves, mass=mass)

        #   7) viscosity
        elif eos is None and mass is None and viscosity is not None:
            selected_waves = self._select_viscosity(self.waves,
                    viscosity=viscosity)

        else:

            return self.waves

        return selected_waves


    #
    # Helpful subroutines to improve readability of waveform selection above:
    #
    @staticmethod
    def _select_viscosity(waves, viscosity):

        if viscosity=='lessvisc':
            return [wave for wave in waves if wave['viscosity']=='lessvisc']

        elif viscosity=='oldvisc':
            return [wave for wave in waves if wave['viscosity']=='oldvisc']

        else: 
            print >> sys.stderr, "viscosity not recognised"
            sys.exit()

    @staticmethod
    def _select_mass(waves, mass=None):

        return [wave for wave in waves if wave['mass']==mass]

    @staticmethod
    def _select_eos(waves, eos=None):

        return [wave for wave in waves if wave['eos']==eos]

    # ----------------------------------------------------------------------



    def remove_wave(self, wave):
        """
        Remove the item wave from the list of waves
        """
        self.waves.remove(wave)

    def copy(self):
        """
        Make a copy of this object
        """

        return WaveData(eos=self.eos, mass=self.mass, viscosity=self.viscosity)






# ________________ - Function Definitions - ________________ #

def main():

    #
    # Create WaveData instance
    #
    wavedata = WaveData()

    return wavedata


if __name__ == "__main__":

    wavedata = main()

