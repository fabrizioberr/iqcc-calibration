# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

# %%
library = QualibrationLibrary.get_active_library()

# %%
class Parameters(GraphParameters):
    qubits: List[str] = None

parameters = Parameters()

if parameters.qubits is None:
    machine = Quam.load()
    parameters.qubits = [q.name for q in machine.active_qubits]

multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="graph_retune_fine_arbel",
    parameters=parameters,
    nodes={
        "IQ_blobs": library.nodes["07_iq_blobs"].copy(
            multiplexed=multiplexed,
            name="IQ_blobs",
            reset_type="thermal",
        ),
        "ramsey_flux_calibration": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            multiplexed=multiplexed, 
            name="ramsey_flux_calibration",
            num_shots = 300,
            flux_span = 0.02,
            flux_num = 11,
            frequency_detuning_in_mhz = 4
        ),
        "power_rabi_x180": library.nodes["04b_power_rabi"].copy(
            operation="x180",
            name="power_rabi_x180",
            reset_type=reset_type_thermal_or_active,
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.002,
            max_number_pulses_per_sweep=50,
            update_x90=True,
            use_state_discrimination=False,
            multiplexed=multiplexed,
        ),
        "single_qubit_randomized_benchmarking": library.nodes["11d_Single_Qubit_Randomized_Benchmarking_legacy"].copy(
            flux_point_joint_or_independent=flux_point, 
            multiplexed=True, 
            num_random_sequences=300,
            log_scale=True,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            name="single_qubit_randomized_benchmarking"
        ),
    },
    connectivity=[
        ("IQ_blobs", "ramsey_flux_calibration"),
        ("ramsey_flux_calibration", "power_rabi_x180"),
        ("power_rabi_x180", "single_qubit_randomized_benchmarking")
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
