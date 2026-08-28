#!/usr/bin/env python

import numpy as np
import yaml
#from casacore.tables import table
from pyrap.tables import table
import argparse

def load_sefd_config(filepath="sefd.yaml"):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)['sefd']

def add_visibility_noise_from_names(ms_path, sefd_dict_filename, delta_nu, t_int, column='MODEL_DATA', rowchunk=1000):
    """
    Add thermal noise to visibilities in a Measurement Set using per-antenna SEFDs, writing in chunks.

    Parameters:
    -----------
    ms_path : str
        Path to the Measurement Set.

    sefd_dict_filename : str
        Path to YAML file with SEFDs keyed by antenna name.

    delta_nu : float
        Channel bandwidth in Hz.

    t_int : float
        Integration time per visibility.

    column : str
        Column to write noise into.

    rowchunk : int
        Number of rows to process at a time.
    """
    # Load SEFD dictionary
    sefd_dict = load_sefd_config(sefd_dict_filename)

    # Load antenna mapping
    ant_table = table(ms_path + '::ANTENNA')
    ant_names = ant_table.getcol("NAME")
    ant_index_to_name = {i: name for i, name in enumerate(ant_names)}
    ant_table.close()

    # Open MS table
    ms = table(ms_path, readonly=False)

    # Confirm column exists
    if column not in ms.colnames():
        raise ValueError(f"Column '{column}' does not exist in {ms_path}")

    # Optionally copy MODEL_DATA to column
    if column != 'MODEL_DATA' and 'MODEL_DATA' in ms.colnames():
        print(f"Initializing {column} with contents of MODEL_DATA")
        model_data = ms.getcol('MODEL_DATA')
        ms.putcol(column, model_data)

    ant1 = ms.getcol("ANTENNA1")
    ant2 = ms.getcol("ANTENNA2")
    nrows = ms.nrows()

    # Get the shape from the first row
    one_row = ms.getcell(column, 0)
    npol, nchan = one_row.shape

    denom = np.sqrt(2 * delta_nu * t_int)

    print(f"Adding noise to {column} in chunks of {rowchunk} rows...")

    for start in range(0, nrows, rowchunk):
        end = min(start + rowchunk, nrows)
        rows = end - start

        # Read chunk
        data_chunk = ms.getcol(column, startrow=start, nrow=rows)
        print(f"data_chunk shape: {data_chunk.shape}")
        for rel_row in range(rows):
            abs_row = start + rel_row
            i, j = ant1[abs_row], ant2[abs_row]
            if i == j:
                continue

            name1 = ant_index_to_name[i]
            name2 = ant_index_to_name[j]

            sefd1 = sefd_dict.get(name1)
            sefd2 = sefd_dict.get(name2)

            if sefd1 is None or sefd2 is None:
                raise ValueError(f"Missing SEFD for one of: {name1}, {name2}")
            eta=0.7 #included 0.7 efficiency factor from e-MERLIN calculator. For MeerKAT use 0.7 as well.
            sigma = np.sqrt(sefd1 * sefd2) /(eta* denom)
            nchan, npol = data_chunk.shape[1:3]
            noise_real = np.random.normal(0, sigma, (nchan, npol))
            noise_imag = np.random.normal(0, sigma, (nchan, npol))
            noise = noise_real + 1j * noise_imag

            data_chunk[rel_row, :, :] += noise

        # Write updated chunk
        ms.putcol(column, data_chunk, startrow=start, nrow=rows)

    ms.close()
    print(f"Noise added to column '{column}'")

def main():
    parser = argparse.ArgumentParser(description='Add noise to visibilities in a Measurement Set.')
    parser.add_argument('--ms_path', type=str, required=True, help='Path to the Measurement Set (e.g., eMERLIN.ms).')
    parser.add_argument('--sefd_dict_filename', type=str, required=True, help='YAML file with per-antenna SEFDs.')
    parser.add_argument('--delta_nu', type=float, help='Channel bandwidth in Hz (default: 1e6).')
    parser.add_argument('--t_int', type=float, help='Integration time per visibility in seconds.')
    parser.add_argument('--column', type=str, default='MODEL_DATA', help='Column to inject noise into.')
    parser.add_argument('--rowchunk', type=int, default=1000, help='Number of rows to process at once.')

    args = parser.parse_args()
    add_visibility_noise_from_names(
        ms_path=args.ms_path,
        sefd_dict_filename=args.sefd_dict_filename,
        delta_nu=args.delta_nu,
        t_int=args.t_int,
        column=args.column,
        rowchunk=args.rowchunk
    )

if __name__ == '__main__':
    main()
