    # %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None

parameters = Parameters()

# Get the relevant QuAM components
if parameters.qubits is None:
    machine = Quam.load()
    parameters.qubits = [q.name for q in machine.active_qubits]


multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "thermal"
simulate = False


g = QualibrationGraph(
    name="retune_graph_qc_qwfix",
    parameters=parameters,
    nodes={
        "resonator_spectroscopy_vs_flux": library.nodes["02c_resonator_spectroscopy_vs_flux"].copy(name="resonator_spectroscopy_vs_flux",
        num_averages=20,
        min_flux_offset_in_v=-0.3,
        max_flux_offset_in_v=0.3,
        num_flux_points=101,
        simulate=simulate),
        "qubit_spectroscopy": library.nodes["03a_qubit_spectroscopy"].copy(name="qubit_spectroscopy",
        simulate=simulate),
         "qubit_spectroscopy_vs_flux": library.nodes["03b_qubit_spectroscopy_vs_flux"].copy(name="qubit_spectroscopy_vs_flux",
        num_averages=75,
        frequency_span_in_mhz=35,
        frequency_step_in_mhz=0.5,
        min_flux_offset_in_v=-0.0075,
        max_flux_offset_in_v=0.005,
        multiplexed=True,
        simulate=simulate), # this is a specific configuration found for the qwfix chip and also forced by the data error of the DGX QOP
        "power_rabi_x180": library.nodes["04b_power_rabi"].copy(    
            name="power_rabi_x180",
            simulate=simulate
        ),
        "T1": library.nodes["05_T1"].copy(name="T1", 
        multiplexed=True,
        simulate=simulate),
        "T2Ramsey": library.nodes["06_ramsey"].copy(name="T2Ramsey",
        simulate=simulate),
        "readout_frequency_optimization": library.nodes["08a_readout_frequency_optimization"].copy(name="readout_frequency_optimization",
        simulate=simulate),
        "readout_power_optimization": library.nodes["08b_readout_power_optimization"].copy(name="readout_power_optimization",
        simulate=simulate),
        # "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(
        #     flux_point_joint_or_independent=flux_point,
        #     multiplexed=multiplexed,
        #     name="IQ_blobs",
        #     reset_type_thermal_or_active="thermal",
        # ),
        "drag_calibration": library.nodes["10b_drag_calibration_180_minus_180"].copy(name="drag_calibration", 
        min_amp_factor=-1.99,
        max_amp_factor=1.99,
        reset_type_thermal_or_active=reset_type_thermal_or_active,
        simulate=simulate),
        
        "single_qubit_randomized_benchmarking": library.nodes["11a_single_qubit_randomized_benchmarking"].copy(
            
            multiplexed=False, 
            name="single_qubit_randomized_benchmarking",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            simulate=simulate
        ),
    },
    connectivity=[
        ("resonator_spectroscopy_vs_flux", "qubit_spectroscopy"),
        ("qubit_spectroscopy", "qubit_spectroscopy_vs_flux"),
        ("qubit_spectroscopy_vs_flux", "power_rabi_x180"),
        ("power_rabi_x180", "T1"),
        ("T1", "T2Ramsey"),
        ("T2Ramsey", "readout_frequency_optimization"),
        ("readout_frequency_optimization", "readout_power_optimization"),
        ("readout_power_optimization", "drag_calibration"),
        ("drag_calibration", "single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)


# %%

g.run()
# %%
