#!/usr/bin/env python
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

import optparse
import sys
import os
import numpy as np
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process

from pylal import git_version
from pylal.SimInspiralUtils import ExtractSimInspiralTableLIGOLWContentHandler

#
# Filenames: move sources in sys.argv[1] to locations in sys.argv[2]
#
original_sim_file=sys.argv[1]
location_sim_file=sys.argv[2]
outputname=sys.argv[3]#original_sim_file.replace(".xml","_relocated.xml")

#
# Read inspinj files
#
locations_xmldoc = utils.load_filename(location_sim_file, contenthandler =
        ExtractSimInspiralTableLIGOLWContentHandler, verbose=True)
locations_table  = table.get_table(locations_xmldoc, lsctables.SimInspiralTable.tableName)

original_sim_xmldoc = utils.load_filename(original_sim_file, contenthandler =
        ExtractSimInspiralTableLIGOLWContentHandler, verbose=True)
original_sim_table  = table.get_table(original_sim_xmldoc, lsctables.SimInspiralTable.tableName)

#
# Create a new document
#
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
sim_table = lsctables.New(lsctables.SimInspiralTable)
xmldoc.childNodes[0].appendChild(sim_table)

# Populate the new table with rows from the old one, but use the location from
# the SNR-scaled table

location_fields = ['distance', 'longitude', 'latitude', 'polarization', \
        'inclination']

for sim, location in zip(original_sim_table, locations_table):
    row = sim_table.RowType()
    for slot in row.__slots__: 
        setattr(row, slot, getattr(sim, slot))

    # Now move the source 
    for field in location_fields:
        setattr(row, field , getattr(location, field))
    sim_table.append(row)

#
# Write the new file
#

#outputname = original_sim_file.replace('INJECTIONS', 'INJECTIONS_ABOVE_SNR')
f = open(outputname,'w')
xmldoc.write(f)
f.close()




