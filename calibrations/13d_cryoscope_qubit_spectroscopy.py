# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from datetime import datetime
import time
from typing import Literal, Optional, List

from qm.qua import *

from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.cryoscope_qubit_spectroscopy import (
    process_raw_dataset, 
    fit_raw_data, 
    plot_qubit_spectroscopy_vs_time,
    plot_qubit_frequency_shift_vs_time,
    plot_qubit_flux_response_vs_time,
)
from calibration_utils.cryoscope.cryoscope_tools import decompose_exp_sum_to_cascade
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates
start = time.time()


# %% {Initialisation}
description = """
Qubit Spectroscopy vs Flux Time - Long Time Cryoscope
===================================================

This sequence involves doing a qubit spectroscopy as a function of time after a flux pulse. 
The instantaneous frequency shift is used to extract the actual flux seen by the qubit as a function of time.
The flux response is then fitted to a single or double exponential decay.
The experiment is based on the description at https://arxiv.org/pdf/2503.04610.

Key Features:
------------
- Measures qubit frequency shift over time after flux pulse application
- Supports both single and double exponential decay fitting
- Provides visualization of raw data, frequency shifts, and flux response
- Updates exponential filter parameters in the system state

Prerequisites:
-------------
- Rough calibration of a pi-pulse, preferably with Gaussian envelope ("Power_Rabi_general_operation").
- Calibration of XY-Z delay.
- Identification of the approximate qubit frequency ("qubit_spectroscopy").
- The quadratic dependence of the qubit frequency on the flux ("Ramsey_vs_flux").

Before proceeding to the next node:
    - Update the exponential filter in the state.
"""

class Parameters(NodeParameters):
    """Configuration parameters for the spectroscopy experiment.
    
    Attributes:
        qubits (List[str]): List of qubits to measure
        num_shots (int): Number of measurement averages
        operation (str): Qubit operation to perform (default: "x180_Gaussian")
        operation_amplitude_factor (float): Scaling factor for operation amplitude
        duration_in_ns (int): Total measurement duration in nanoseconds
        frequency_span_in_mhz (float): Frequency sweep range in MHz
        frequency_step_in_mhz (float): Frequency step size in MHz
        flux_amp (float): Amplitude of flux pulse
        update_lo (bool): Whether to update local oscillator frequency
        fit_single_exponential (bool): Use single vs double exponential fit
        update_state (bool): Update system state with fit results
        flux_point_joint_or_independent (str): Flux point handling method
        simulate (bool): Run in simulation mode
        simulation_duration_ns (int): Simulation duration
        timeout (int): Operation timeout in seconds
        load_data_id (int): ID of data to load (optional)
        multiplexed (bool): Use multiplexed vs sequential measurement
        reset_type_active_or_thermal (str): Reset method to use
    """

    qubits: Optional[List[str]] = ['Q3']
    num_shots: int = 50
    operation: str = "x180"
    operation_amplitude_factor: Optional[float] = 1
    duration_in_ns: Optional[int] = 1000
    time_axis: Literal["linear", "log"] = "linear"
    time_step_in_ns: Optional[int] = 40 # for linear time axis
    time_step_num: Optional[int] = 200 # for log time axis
    frequency_span_in_mhz: float = 150
    frequency_step_in_mhz: float = 0.5
    flux_amp : float = 0.16
    update_lo: bool = True
    fitting_base_fractions: List[float] = [0.15, 0.05] # fraction of times from which to fit each exponential  
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    reset_type: Literal['active', 'active_simple', 'thermal'] = 'thermal'
    thermal_reset_extra_time_in_us: Optional[int] = 10_000
    min_wait_time_in_ns: Optional[int] = 32
    update_state: bool = False

# %% {Node initialisation}
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="13d_cryoscope_qubit_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    if node.parameters.operation_amplitude_factor:
        # pre-factor to the value defined in the config - restricted to [-2; 2)
        operation_amp = node.parameters.operation_amplitude_factor
    else:
        operation_amp = 1.0
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
    # Flux bias sweep
    if node.parameters.time_axis == "linear":
        times = np.arange(node.parameters.min_wait_time_in_ns // 4, node.parameters.duration_in_ns // 4, node.parameters.time_step_in_ns // 4, dtype=np.int32)
    elif node.parameters.time_axis == "log":
        times = np.logspace(np.log10(node.parameters.min_wait_time_in_ns // 4), np.log10(node.parameters.duration_in_ns // 4), node.parameters.time_step_num, dtype=np.int32)
        # Remove repetitions from times
        times = np.unique(times)

    # flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
    detuning = [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]
    
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "detuning sweep",  "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "time after flux pulse", "units": "ns"})
    }
    
    # Modify the lo frequency to allow for maximum detuning, this change will be reverted at the end of the node
    node.namespace["tracked_qubits"] = []
    if node.parameters.update_lo:
        for q in qubits:
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
                lo_band = q.xy.opx_output.band
                rf_frequency = q.xy.intermediate_frequency + q.xy.opx_output.upconverter_frequency
                lo_frequency = rf_frequency - 400e6
                if (lo_band == 3) and (lo_frequency < 6.5e9):
                    lo_frequency = 6.5e9
                elif (lo_band == 2) and (lo_frequency < 4.5e9):
                    lo_frequency = 4.5e9
                print(f"Updated LO frequency for {q.name}: {lo_frequency/1e9} GHz")
                
                # q.xy.intermediate_frequency = rf_frequency - lo_frequency
                q.xy.opx_output.upconverter_frequency = lo_frequency
                q.xy.opx_output.band = lo_band
                node.namespace["tracked_qubits"].append(q)

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for frequency scan
        t_delay = declare(int)  # QUA variable for delay time scan
        
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_each_(t_delay, times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)

                        # Qubit Manipulation
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detuning[i]) # Frequency update
                            qubit.align()
                            qubit.z.play("const", amplitude_scale=node.parameters.flux_amp / qubit.z.operations["const"].amplitude, duration=t_delay+200) # Flux pulse
                            qubit.xy.wait(t_delay) # Wait after flux pulse application
                            qubit.xy.play(operation, amplitude_scale=operation_amp) # excitation pulse
                            qubit.align()
                            qubit.wait(200) # NOTE: this is 800ns, why actually necessary?
                        
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        
                        # Wait for the resonator to deplete of photons
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)

   
# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["fit_results"] = fit_raw_data(node.results["ds_raw"], node)
    end = time.time()
    print(f"Script runtime: {end - start:.2f} seconds")


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_qubit_spec_vs_time = plot_qubit_spectroscopy_vs_time(node.results["ds_raw"], node.namespace["qubits"])
    fig_qubit_freq_shift_vs_time = plot_qubit_frequency_shift_vs_time(node.results["ds_fit"], node.namespace["qubits"], scale="linear")
    fig_qubit_freq_shift_vs_time_log = plot_qubit_frequency_shift_vs_time(node.results["ds_fit"], node.namespace["qubits"], scale="log")
    fig_flux_response_vs_time = plot_qubit_flux_response_vs_time(node.results["ds_fit"], node.namespace["qubits"], node.results["fit_results"], scale="linear")
    fig_flux_response_vs_time_log = plot_qubit_flux_response_vs_time(node.results["ds_fit"], node.namespace["qubits"], node.results["fit_results"], scale="log")
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "raw_qubit_spec_vs_time": fig_qubit_spec_vs_time,
        "fitted_qubit_freq_shift_vs_time": fig_qubit_freq_shift_vs_time,
        "fitted_qubit_freq_shift_vs_time_log": fig_qubit_freq_shift_vs_time_log,
        "flux_response_vs_time": fig_flux_response_vs_time,
        "flux_response_vs_time_log": fig_flux_response_vs_time_log,
    }


# %% extra plots

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
from qualibration_libs.plotting import QubitGrid, grid_iter
qubits = node.namespace["qubits"]
fit = node.results["ds_fit"]
fit_results = node.results["fit_results"]
scale = "linear"


grid = QubitGrid(fit, [q.grid_location for q in qubits])

# Plot flux response and fitted curves for each qubit
for ax, qubit in grid_iter(grid):
    # Plot measured flux response
    # fit.loc[qubit].flux_response.plot(ax=ax)
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
        magnitude = y_fit[-1]
        if node.parameters.update_state:
            ax.plot(t_data, y_fit / magnitude, color='r', label='Full Fit', linewidth=2)
        # ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
        #         verticalalignment='top', fontsize=8)
    (fit.loc[qubit].flux_response/magnitude).plot(ax=ax)
    ax.axhline(y=0.999, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=1.001, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel("Flux (V)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    if scale == "log":
        ax.set_xscale('log')
        ax.grid(True)
grid.fig.suptitle(f"Flux response vs time")
grid.fig.tight_layout()

plt.show()

scale = "log"


grid = QubitGrid(fit, [q.grid_location for q in qubits])

# Plot flux response and fitted curves for each qubit
for ax, qubit in grid_iter(grid):
    # Plot measured flux response
    # fit.loc[qubit].flux_response.plot(ax=ax)
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
        magnitude = y_fit[-1]
        if node.parameters.update_state:
            ax.plot(t_data, y_fit / magnitude, color='r', label='Full Fit', linewidth=2)
        # ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
        #         verticalalignment='top', fontsize=8)
    (fit.loc[qubit].flux_response/magnitude).plot(ax=ax)
    ax.axhline(y=0.999, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=1.001, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel("Flux (V)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    if scale == "log":
        ax.set_xscale('log')
        ax.grid(True)
grid.fig.suptitle(f"Flux response vs time")
grid.fig.tight_layout()

plt.show()

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for qubit in node.namespace.get("tracked_qubits", []):
        qubit.revert_changes()
    
    if node.parameters.update_state:
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                fit_results_per_qubit = node.results["fit_results"][q.name]
                if fit_results_per_qubit["fit_successful"]:
                    A_list = [component[0] / fit_results_per_qubit["best_a_dc"] for component in fit_results_per_qubit["best_components"]]
                    tau_list = [component[1] for component in fit_results_per_qubit["best_components"]]
                    A_c, tau_c, scale = decompose_exp_sum_to_cascade(A=A_list, tau=tau_list, A_dc=1)
                    q.z.opx_output.exponential_filter = list(zip(A_c, tau_c))
                    print("updated the exponential filter")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
