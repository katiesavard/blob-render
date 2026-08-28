#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk

import sys
import argparse
from pyrap.tables import table
import numpy as np

def add_data_col(msname, modelcol, newcol):
    tt = table(msname, readonly=False)
    colnames = tt.colnames()

    if newcol in colnames:
#instead of replacing jsut remove it:
        print(f"{newcol} already exists — REMOVE it")
        tt.removecols(newcol)
#Here changed all of this renaming section to just removing the file as if not it just does not work.
        # Remove old column (not supported directly — workaround by renaming)
       # alt_name = newcol + "_OLD"
       # print(f"Renaming {newcol} → {alt_name} to avoid conflict")
       # tt.renamecol(newcol, alt_name)

    # Add new writable column
    print(f"Creating new column {newcol}")
    desc = tt.getcoldesc(modelcol)
    desc['name'] = newcol
    desc['comment'] = desc['comment'].replace(' ', '_')
    tt.addcols(desc)

    # Initialize with zeros
    model_data = tt.getcol(modelcol)
    new_data=model_data.copy()
#injecting core here is causing issues.
   # core_flux= 2e-3 #this is in Jy
    #new_data+=core_flux
    tt.putcol(newcol,new_data)
    print(f"{newcol} created successfully")
#    try:
 #       tt.putcol(newcol, np.zeros_like(model_data))
#        tt.putcol(newcol, new_data)
#        print(f"{newcol} created and initialized successfully with a pixel core of {core_flux} Jy.")
#    except: RuntimeError as e:
#        print(f"ERROR: Column was added, but could not be initialized: {e}")

    tt.close()


def main():
    parser = argparse.ArgumentParser(description="Add a new visibility column to a Measurement Set.")
    parser.add_argument('--ms_path', type=str, help='Path to the Measurement Set (e.g., my.ms)')
    parser.add_argument('--modelcol', type=str, default='MODEL_DATA', help='Name of the model data column (default: MODEL_DATA)')
    parser.add_argument('--newcol', type=str, default='NOISY_MODEL', help='Name of the new column to add (default: NOISY_MODEL)')

    args = parser.parse_args()
    modelcol = args.modelcol
    newcol = args.newcol
    ms_path = args.ms_path

    add_data_col(ms_path,modelcol,newcol)

if __name__ == '__main__':

    main()
