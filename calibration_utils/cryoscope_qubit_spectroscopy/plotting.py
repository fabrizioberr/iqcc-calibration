from typing import List
import xarray as xr
import numpy as np
from matplotlib.figure import Figure
from typing import Literal

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_qubit_spectroscopy_vs_time(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots raw spectroscopy data vs time for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the qubit spectroscopy data.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data (no fit in this case).
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        im = ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].state.plot(
            ax=ax, add_colorbar=False, x="time", y="freq_GHz"
        )
        ax.set_ylabel("Frequency (GHz)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        cbar = grid.fig.colorbar(im, ax=ax)
        cbar.set_label("Qubit State")
    grid.fig.suptitle(f"Qubit spectroscopy vs time after flux pulse")
    grid.fig.tight_layout()

    return grid.fig


def plot_qubit_frequency_shift_vs_time(
    fit: xr.Dataset, 
    qubits: List[AnyTransmon], 
    scale: Literal["linear", "log"] = "linear"
    ) -> Figure:
    """
    Plots the extracted frequency shift vs time for the given qubits.

    Parameters
    ----------
    fit : xr.Dataset
        The dataset containing the fitted qubit frequency shift.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains fitted frequency shift vs time.
    """
    grid = QubitGrid(fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (fit.loc[qubit].center_freqs / 1e9).plot(ax=ax, marker='o')
        ax.set_ylabel("Frequency (GHz)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        if scale == "log":
            ax.set_xscale('log')
            ax.grid(True)
    grid.fig.suptitle(f"Qubit frequency shift vs time after flux pulse")
    grid.fig.tight_layout()
    
    return grid.fig


def plot_qubit_flux_response_vs_time(
    fit: xr.Dataset, 
    qubits: List[AnyTransmon], 
    fit_results: dict, 
    scale: Literal["linear", "log"] = "linear"
    ) -> Figure:
    """
    Plots the flux response vs time for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the flux response.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(fit, [q.grid_location for q in qubits])

    # Plot flux response and fitted curves for each qubit
    for ax, qubit in grid_iter(grid):
        # Plot measured flux response
        fit.loc[qubit].flux_response.plot(ax=ax)
        # flux_response_norm = ds.loc[qubit].flux_response / ds.loc[qubit].flux_response.values[-1]
        # flux_response_norm.plot(ax=ax)
        
        # Plot fitted curves and parameters if fits were successful    
        if fit_results[qubit["qubit"]]["fit_successful"]:
            t_data = fit.loc[qubit].time.values
            best_a_dc = fit_results[qubit["qubit"]]["best_a_dc"]
            t_offset = t_data - t_data[0]
            y_fit = np.ones_like(t_data, dtype=float) * best_a_dc  
            fit_text = f'a_dc = {best_a_dc:.3f}\n'
            for i, (amp, tau) in enumerate(fit_results[qubit["qubit"]]["best_components"]):
                y_fit += amp * np.exp(-t_offset/tau)
                fit_text += f'a{i+1} = {amp / best_a_dc:.3f}, τ{i+1} = {tau:.0f}ns\n'

            ax.plot(t_data, y_fit, color='r', label='Full Fit', linewidth=2)
            ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8)

        ax.set_ylabel("Flux (V)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        if scale == "log":
            ax.set_xscale('log')
            ax.grid(True)
    grid.fig.suptitle(f"Flux response vs time")
    grid.fig.tight_layout()
    
    return grid.fig