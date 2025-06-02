import os
import numpy as np
import yaml
import argparse
from blobrender.help_strings import HELP_DICT, TYPES_DICT



def save_list(d,folder,name):
    """ saves a np array into a given location
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file = folder+'/'+name+'.npy'
    np.save(file,d)

def load_list(folder, name):
    """loads a numpy array from a given location
    """
    file = folder+'/'+name+'.npy'
    arr = np.load(file,allow_pickle=True)
    return arr

def loader_bar(i,range,modulo): #just for output purposes
    perc = int((i/range)*100)
    prev_perc = int(((i-1)/range)*100)
    if i==0:
        s = str(0)+"%"
        print(s,end="...",flush=True)
    if perc%modulo==0 and perc!=prev_perc:
        s = str(perc)+"%"
        print(s,end="...",flush=True)
        prev_perc=perc+modulo

def get_arguments(default_yaml_file,help_dict=None,description=' '):
    # Step 1: Pre-parse --config to determine which YAML file to use
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=default_yaml_file,
                            help="Path to the YAML config file (default: %(default)s)")
    args_config, remaining_argv = pre_parser.parse_known_args()
    yaml_file = args_config.config

    
    # Step 2: Load defaults from YAML
    with open(yaml_file, 'r') as f:
        defaults = yaml.safe_load(f)

    # Step 3: Enforce types from TYPES_DICT
    for key, value in defaults.items():
        desired_type = TYPES_DICT.get(key)
        if desired_type is not None and not isinstance(value, desired_type):
            try:
                # Special handling for bool, since bool('False') is True
                if desired_type is bool and isinstance(value, str):
                    value = value.lower() == "true"
                else:
                    value = desired_type(value)
            except Exception:
                pass  # If conversion fails, keep original
            defaults[key] = value      
    
    # Step 4: Filter help_dict to only keys present in the YAML
    filtered_help = {k: v for k, v in (help_dict or HELP_DICT).items() if k in defaults}
    
    # Step 5: Main parser with all arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=default_yaml_file,
                        help="Path to the YAML config file (default: %(default)s)")    
    for key, value in defaults.items():
        arg_type = TYPES_DICT.get(key, type(value))
        help_str = filtered_help.get(key, f'(default from YAML: {value})')
        # Handle booleans as store_true/store_false
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=lambda x: (str(x).lower() == 'true'), default=value, help=help_str)
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=value, help=help_str)

    args = parser.parse_args(remaining_argv)

    print(f"Using config file: {yaml_file}")

    # Step 6: Update YAML if any values changed
    updated = False
    changes = []
    for key in defaults:
        arg_val = getattr(args, key)
        if defaults[key] != arg_val:
            changes.append(f"{key}: {defaults[key]} -> {arg_val}")
            defaults[key] = arg_val
            updated = True

    if updated:
        with open(yaml_file, 'w') as f:
            yaml.safe_dump(defaults, f, default_flow_style=False)
        print(f"Updated {yaml_file} with new values:")
        for change in changes:
            print("  " + change)
    
    # Step 7: Print summary
    print("\nSummary of parameters used:")
    for key in defaults:
        arg_val = getattr(args, key)
        source = "command line" if any(f"{key}:" in change for change in changes) else "YAML"
        print(f"  {key}: {arg_val}   (from {source})")

    return args

def update_yaml(variable, new_value, yaml_path):
    """
    Update the fitsfile_name variable in the specified YAML file with the new fits_name.
    """
    with open(yaml_path, 'r') as f:
        pred_defaults = yaml.safe_load(f)
    pred_defaults[variable] = new_value 
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(pred_defaults, f, default_flow_style=False)
    print(f"Updated {variable} in {yaml_path} to {new_value}")