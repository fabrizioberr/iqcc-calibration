# import logging
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from calibration_utils.cryoscope import cryoscope_tools


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    iw_angle: float
    saturation_amp: float
    x180_amp: float
    success: bool



def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    # ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    # ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    ds = ds.assign_coords(
        {
            "full_freq": (  # Full frequency including RF and flux-induced shifts
                ["qubit", "detuning"],
                np.array([ds.detuning + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in node.namespace["qubits"]]),
            ),
            "full_detuning": (  # Frequency shift due to flux
                ["qubit", "detuning"],
                np.array([ds.detuning + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in node.namespace["qubits"]]),
            ),
            "flux": (  # Flux at given detuning
                ["qubit", "detuning"],
                np.array([np.sqrt(ds.detuning / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp**2) for q in node.namespace["qubits"]]),
            )
        }
    )
    ds.full_freq.attrs["long_name"] = "Full Frequency"
    ds.full_freq.attrs["units"] = "Hz"
    ds.full_detuning.attrs["long_name"] = "Full Detuning"
    ds.full_detuning.attrs["units"] = "Hz"
    ds.flux.attrs["long_name"] = "Flux"
    ds.flux.attrs["units"] = "V"

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # Extract frequency points and reshape data for analysis
    # freqs = ds['detuning'].values

    # Transpose to ensure ('qubit', 'time', 'freq') order for analysis
    stacked = ds.transpose('qubit', 'time', 'detuning')

    # Fit Gaussian to each spectrum to find center frequencies
    center_freqs = xr.apply_ufunc(
        lambda states: cryoscope_tools.fit_gaussian(ds['detuning'].values, states),
        stacked,
        input_core_dims=[['detuning']],
        output_core_dims=[[]],  # no dimensions left after fitting
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    ).rename({"state": "center_frequency"})

    # Add flux-induced frequency shift to center frequencies
    center_freqs = center_freqs.center_frequency + np.array([q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 * np.ones_like(ds.time) for q in node.namespace["qubits"]])

    # Calculate flux response from frequency shifts
    flux_response = np.sqrt(center_freqs / xr.DataArray([q.freq_vs_flux_01_quad_term for q in node.namespace["qubits"]], coords={"qubit": center_freqs.qubit}, dims=["qubit"]))

    # Store results in dataset
    ds = xr.Dataset({"center_freqs": center_freqs, "flux_response": flux_response})

    # Perform exponential fitting for each qubit
    fit_results = {}
    for q in node.namespace["qubits"]:
        fit_results[q.name] = {}
        t_data = flux_response.sel(qubit=q.name).time.values
        y_data = flux_response.sel(qubit=q.name).values
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = cryoscope_tools.optimize_start_fractions(
            t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
            )

        fit_results[q.name]["fit_successful"] = fit_successful
        fit_results[q.name]["best_fractions"] = best_fractions
        fit_results[q.name]["best_components"] = best_components
        fit_results[q.name]["best_a_dc"] = best_a_dc
        fit_results[q.name]["best_rms"] = best_rms

    node.results["fit_results"] = fit_results

    return ds, fit_results