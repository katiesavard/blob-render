from pyrap.tables import table
import argparse


def inject_core(msname, column, core_flux):
    #core flux called is in Jy. Follow the same recipe as the add_MS_column.py function:
    tt = table(msname, readonly=False)
    data = tt.getcol(column)
    data += core_flux #this time add the core hot pixel to the data.
    tt.putcol(column, data)
    print(f"Injected {core_flux} Jy core into {column} at the current phase centre")
    tt.close()


def main(): #follow the same recipe of building the main function as for add_MS_column.py, we can define the arguments that we want to call in predict_fromfits.py
    parser = argparse.ArgumentParser()

    parser.add_argument("--ms_path", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--core_flux", type= float , required=True)
    args = parser.parse_args()

    inject_core(args.ms_path, args.column, args.core_flux)

if __name__ == "__main__":
    main()
