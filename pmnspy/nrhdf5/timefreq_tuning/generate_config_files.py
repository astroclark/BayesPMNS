#!/usr/bin/env python

import fileinput
import shutil
import numpy as np

filein="flow_1024-srate_8192_seglen-4.00.ini"


for new_seglen in np.arange(0.25,4,0.25):
#new_seglen=2.0

    new_window=new_seglen/4.0

    fileout=filein.replace("seglen-4.00","seglen-%.2f"%new_seglen)
    shutil.copy(filein, fileout)

    f = open(filein,'r')
    filedata = f.read()
    f.close()

    newdata = filedata.replace("seglen=4.0","seglen=%.2f"%new_seglen)
    newdata = newdata.replace("window=1.0", "window=%.2f"%new_window)

    f = open(fileout,'w')
    f.write(newdata)
    f.close()

