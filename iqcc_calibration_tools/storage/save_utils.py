from qualibrate_app.config import get_config_path, get_settings
from iqcc_calibration_tools.quam_config.components import Quam
import os
from pathlib import Path
import xarray as xr
import json
import numpy as np
import logging

# ANSI color codes
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Custom formatter for magenta colored logs
class MagentaFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{MAGENTA}{record.msg}{RESET}"
        return super().format(record)

# Configure logging with magenta color
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(MagentaFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def extract_string(input_string):
    # Find the index of the first occurrence of a digit in the input string
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)

    if index is not None:
        # Extract the substring from the start of the input string to the index
        extracted_string = input_string[:index]
        return extracted_string
    else:
        return None


def fetch_results_as_xarray(handles, qubits, measurement_axis):
    """
    Fetches measurement results as an xarray dataset.
    Parameters:
    - handles : A dictionary containing stream handles, obtained through handles = job.result_handles after the execution of the program.
    - qubits (list): A list of qubits.
    - measurement_axis (dict): A dictionary containing measurement axis information, e.g. {"frequency" : freqs, "flux",}.
    Returns:
    - ds (xarray.Dataset): An xarray dataset containing the fetched measurement results.
    """

    stream_handles = handles.keys()
    meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, qubit in enumerate(qubits)] for meas_var in meas_vars
    ]
    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}
    
    
    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )

    return ds
def fetch_single_shot_results_as_xarray(handles, qubits, measurement_axis): #FB saves also single shot data
    """
    Fetches measurement results as an xarray dataset.

    Parameters:
    - handles : dict
        A dictionary containing stream handles, e.g. job.result_handles
    - qubits : list
        A list of qubits.
    - measurement_axis : dict
        Dictionary describing measurement axes, e.g. {"t": idle_times, "repetition": n_reps}

    Returns:
    - ds : xarray.Dataset
        Dataset containing results with dims ["qubit", *measurement_axis.keys()]
    """

    stream_handles = handles.keys()
    meas_vars = list(
        set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None])
    )

    values = []
    for meas_var in meas_vars:
        qubit_data = []
        for i, qubit in enumerate(qubits):
            raw = handles.get(f"{meas_var}{i + 1}").fetch_all()
            arr = np.squeeze(raw)  # remove leading (1,1,...)
            # Try to match axis order
            expected_shape = [len(v) if not np.isscalar(v) else v for v in measurement_axis.values()]
            if list(arr.shape) != expected_shape:
                # Try transposing if it's just permuted
                if sorted(arr.shape) == sorted(expected_shape):
                    perm = [arr.shape.index(s) for s in expected_shape]
                    arr = np.transpose(arr, perm)
                else:
                    raise ValueError(
                        f"Shape mismatch for {meas_var}{i+1}: got {arr.shape}, expected {expected_shape}"
                    )
            qubit_data.append(arr)
        values.append(np.array(qubit_data))  # shape: (n_qubits, *axis_sizes)

    # Add qubit names once
    coords = {"qubit": [qubit.name for qubit in qubits], **measurement_axis}

    # Define final axes: qubit first, then all user axes
    ordered_axes = ["qubit"] + list(measurement_axis.keys())

    ds = xr.Dataset(
        {
            meas_var: (ordered_axes, values[i])
            for i, meas_var in enumerate(meas_vars)
        },
        coords=coords,
    )

    return ds


def get_storage_path():
    settings = get_settings(get_config_path())
    storage_location = settings.qualibrate.storage.location
    return Path(storage_location)


def find_numbered_folder(base_path, number):
    """
    Find folder that starts with '#number_'
    Will match '#number_something' but not '#number' alone
    """
    search_prefix = f"#{number}_"
    
    # Manual search for folder starting with #number_ and having something after
    for root, dirs, _ in os.walk(base_path):
        matching_dirs = [d for d in dirs if d.startswith(search_prefix) and len(d) > len(search_prefix)]
        if matching_dirs:
            return os.path.join(root, matching_dirs[0])
    
    return None



def load_dataset(serial_number, target_filename = "ds", parameters = None):
    """
    Loads a dataset from a file based on the serial number.
    
    Args:
        serial_number: The serial number to search for.
        base_folder: The base directory to search in.
    
    Returns:
        An xarray Dataset if found, None otherwise.
    """
    if not isinstance(serial_number, int):
        raise ValueError("serial_number must be an integer")
        
    base_folder = find_numbered_folder(get_storage_path(),serial_number)
    # Look for .nc files in the subfolder
    nc_files = [f for f in os.listdir(base_folder) if f.endswith('.h5')]
    
    # look for filename.h5
    is_present = target_filename in [file.split('.')[0] for file in nc_files]
    filename = [file for file in nc_files if target_filename == file.split('.')[0]][0] if is_present else None
    json_filename = "data.json"
    
    if nc_files:
        # Assuming there's only one .nc file per folder
        file_path = os.path.join(base_folder, filename)
        json_path = os.path.join(base_folder, json_filename)
        # Open the dataset
        ds = xr.open_dataset(file_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        try:
            machine = Quam.load(base_folder + "//quam_state.json")
        except Exception as e:
            print(f"Error loading machine: {e}")
            machine = None
        qubits = [machine.qubits[qname] for qname in ds.qubit.values]    
        if parameters is not None:
            for param_name, param_value in parameters:
                if param_name != "load_data_id":
                    if param_name in json_data["initial_parameters"]:
                        setattr(parameters, param_name, json_data["initial_parameters"][param_name])
            return ds, machine, json_data, qubits,parameters
        else:
            return ds, machine, json_data, qubits
    else:
        print(f"No .nc file found in folder: {base_folder}")
        return None