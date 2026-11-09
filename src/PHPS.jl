"""
    PHPS

Port-Hamiltonian Power System Simulator: power system simulation, stability studies and
region-of-attraction certificates from one generic model code. See `README.md`.
"""
module PHPS

using LinearAlgebra
using SparseArrays
using Printf: Printf
using ForwardDiff: ForwardDiff
import SciMLBase
import Sundials
import JLD2
using Dates: Dates
using Artifacts: Artifacts
using SHA: SHA
using Random: Random
import JSON3
import JSONSchema
import IntervalArithmetic

# io: inputs and parameters
include("io/expr.jl")
include("io/json.jl")
include("io/schema.jl")
include("io/case.jl")
include("io/scenario.jl")
include("io/contracts.jl")
include("io/params.jl")

# network
include("network/ybus.jl")
include("network/events.jl")

# power flow
include("powerflow/newton.jl")

# components
include("components/primitives.jl")
include("components/interface.jl")
include("components/machines/genrou.jl")
include("components/machines/gensal.jl")
include("components/exciters/ieeet1.jl")
include("components/governors/ieeeg1.jl")
include("components/governors/ieeeg3.jl")
include("components/loads/complexload.jl")
include("components/converters/common.jl")
include("components/converters/gfm_vsm.jl")
include("components/converters/gfm_droop.jl")
include("components/converters/gfm_voc.jl")
include("components/converters/gfl.jl")
include("components/dispatch.jl")
include("components/observables.jl")

# assembly
include("assembly/wiring.jl")
include("assembly/dae.jl")
include("assembly/sparsity.jl")

# initialisation
include("init/components.jl")
include("init/equilibrium.jl")

# simulation
include("sim/bdf1.jl")
include("sim/ida.jl")
include("sim/results.jl")

# port-Hamiltonian audits
include("ph/storage.jl")
include("ph/audit.jl")
include("ph/ports.jl")
include("ph/dissipativity.jl")
include("ph/terminal.jl")
include("ph/structure.jl")
include("ph/power.jl")

# ROA certificates
include("roa/interval.jl")
include("roa/section.jl")
include("roa/candidate.jl")
include("roa/enclosure.jl")
include("roa/containment.jl")
include("roa/certificate.jl")
include("roa/analytic.jl")
include("roa/taylor.jl")
include("roa/check.jl")
include("ph/exchange.jl")
include("ph/joint_storage.jl")
include("ph/polar.jl")
include("ph/sink.jl")
include("ph/strain.jl")
include("ph/completion.jl")
include("roa/system_candidate.jl")

# electromagnetic dq pilot of the physical extension (docs/records/early/PHYSICAL_EXTENSION.md)
include("emdq/pilot.jl")
include("emdq/gates.jl")
include("emdq/avr.jl")
include("emdq/nogo.jl")
include("emdq/krasovskii.jl")
include("emdq/phasefree.jl")
include("emdq/bm.jl")
include("emdq/network.jl")
# the two-machine pilot and the PCC coefficient maps (docs/active/PCC_VIEW_N2_FORMULATION.md)
include("emdq/pilot2.jl")
include("emdq/pccview.jl")
include("emdq/pccreduce.jl")
include("ph/component_ports.jl")

# Standalone physical resource primitives (module PhysicalSteam, self-contained); not registered as
# RMS case models. Loaded before the simulator, whose steam units use its IAPWS-95 kernel.
include("physical/steam_resource.jl")

# port-Hamiltonian grid model and simulator (PH_SIMULATOR_TODO.md, docs/active/PH_SIMULATOR_SPEC.md)
include("phsim/transformer.jl")
include("phsim/tline.jl")
include("phsim/motor.jl")
include("phsim/bridge.jl")
include("phsim/zeroseq.jl")
include("phsim/system.jl")
include("phsim/exciter.jl")
include("phsim/steamprops.jl")
include("phsim/steam.jl")
include("phsim/hydro.jl")
include("phsim/converter.jl")
include("phsim/control.jl")
include("phsim/midpoint.jl")
include("phsim/clearing.jl")
include("phsim/pccmanifold.jl")
include("phsim/fullmodel.jl")
include("roa/ivsteam.jl")              # interval methods of the steam lookups (practical-return proof, P0)
include("roa/ivfield.jl")              # interval methods of the field (practical-return proof, P0)
include("roa/ivlimit.jl")              # interval methods of the limiter primitives (window certificate, 13.31)
include("roa/tm1.jl")                  # first-order Taylor models (window certificate, 13.31.3)

# Rigid-shaft coupling of the physical steam component to the unchanged GENROU kernel.
include("physical/steam_machine_shaft.jl")
# The same steam component as the mechanical source of one machine in an assembled grid.
include("physical/steam_grid_unit.jl")


# cached compilation of the simulation path
include("precompile.jl")

# io
export parse_param_expr, param_value
export load_case, write_case, load_scenario, write_scenario, load_json_input, validate_json
export Case, CaseConfig, BusData, LoadData, SourceData, LineData, ShuntData, ComponentSpec, Wire
export Scenario, SolverSettings, AbstractEvent, BusFault, LineFault, OtherEvent
export load_contracts, contract, contract_key, ContractSet, ContractEntry, DomainClause
export component, param, hasparam, component_params, MODEL_TYPES, check_model_type,
       UnsupportedModelError
# network
export Network, nbus, bus_index, ybus, ybus_pf, ybus_dae, norton_stamps, NortonStamp
export load_admittances, LoadAdmittances, fault_admittance, fault_shunts, FaultShunt,
       with_fault, split_line_for_fault
# components
export AbstractComponent, build_component, with_params, model_type, state_names, input_names,
       output_names, nstates, ninputs, noutputs, params, param_dict, ports, bus, rhs!,
       outputs!, step_outputs!, modes, hamiltonian, grad_hamiltonian!, grad_hamiltonian,
       injection, norton_admittance, default_contracts
export GENROU_PHTRUE, GENSAL_PHTRUE, IEEET1_PHTRUE, IEEEG1_PHTRUE, IEEEG3_PHTRUE, COMPLEXLOAD,
       GFM_VSM_PHTRUE, GFM_DROOP_PHTRUE, GFM_VOC_PHTRUE, GFL_PHTRUE
export NoModes, ModeLog, UndecidedBranch, component_role
# assembly
export DAESystem, assemble, dae_residual!, dae_residual, nalg, resolve_wiring, InputSource,
       jacobian_pattern
# initialisation
export init_from_phasor, init_from_targets, MachineTargets, first_pass, solve_equilibrium,
       converter_init, converter_current, lag_states,
       EquilibriumResult, component_io
# simulation
export DAEWorkspace, SimResult, simulate_bdf1, simulate_ida, consistent_voltages!,
       simulate, csv_columns, csv_row, write_results_csv, write_results_jld2, run_metadata,
       observable_names, observable_values!
# port-Hamiltonian audits
export storage_components, total_hamiltonian, grad_total_hamiltonian, hessian_total_hamiltonian,
       solve_network, reduced_field, reduced_jacobian, PhysicalProjection, reservoir_states,
       physical_projection, shifted_storage_audit, PortModel, port_model, transfer,
       real_part_crossings, passivity_certificate, port_zeros, frequency_response,
       loop_port_model, MultiPortModel, open_loops_model, multiport_margin, kyp_riccati,
       port_storage, rest_storage, port_margin, loop_margin, TerminalModel, NetworkModel,
       sync_jacobian, terminal_models, terminal_margins, state_groups, storage_pattern,
       reference_section, lyapunov_check, structured_lyapunov, structured_completion, completion_rate_feasibility,
       minimal_completion, completion_margin,
       generalized_decay_rate, pow2_scaling, decay_margin,
       margin_residuals, verified_min_eig, pattern_certificate, verified_lyapunov,
       rank_couplings, add_coupling!, couple_states!, section_pattern,
       component_power, network_power, power_audit_samples, port_power_audit, rotor_structure,
       current_correction, kcl_solve, rotor_current_coordinates, exchange_one_form,
       one_form_exactness, one_form_path_test, lossless_variant, exchange_curl,
       rotor_energy, shifted_kinetic_energy, quotient_hessian, exchange_dissipation_forms,
       rotor_gradient_metric,
       network_potential, polar_balance, conductance_curl,
       strain_rotor, strain_metric, strain_energy, strain_balance, strain_reference,
       shifted_strain_energy, strain_decay_forms, ConstantPowerSink, integrable_loss_variant,
       certificate_only, completion_problem, completion_pattern, invariant_closure,
       StorageCompletion, storage_completion, completion_energy, completion_energy_state,
       completion_hessian_check, SystemCandidate, system_candidate, reduced_hessian_hull,
       taylor_level, certify_roa_taylor, taylor_recheck, taylor_record_check
# electromagnetic dq pilot
export WindingMachine, winding_machine, EMDQPilot, EMDQ_STATES, emdq_state_names,
       emdq_machine_block, emdq_machine, emdq_rhs!,
       emdq_rhs, emdq_energy_parts, emdq_energy, emdq_power, emdq_pilot, emdq_balance,
       emdq_shifted_energy, emdq_modulation_output, emdq_shifted_balance, emdq_local_forms,
       restricted_max_eig, emdq_reduction_check, emdq_angle_augmentation,
       emdq_angle_augmentation_rate, emdq_augmented_forms, EMDQAVR, emdq_avr,
       emdq_avr_energy, emdq_modulation, emdq_closed_loop_rhs!, emdq_closed_loop_rhs,
       emdq_closed_loop_balance, emdq_completion_data, emdq_repair_report, completion_farkas_check,
       emdq_local_obstruction, emdq_verify_obstruction, emdq_pilot_digest, emdq_big_data,
       emdq_nogo_certificate, check_nogo_certificate, emdq_radius_masks, emdq_verify_lyapunov,
       emdq_interval_data, emdq_component_blocks, emdq_component_mask, emdq_krasovskii_energy,
       emdq_krasovskii_balance, emdq_verify_krasovskii, block_lyapunov_farkas_check,
       block_lyapunov_conditioning, krasovskii_conditioning, block_trace_obstruction,
       verified_block_instability, block_union_abscissae, EMDQPhaseFree, emdq_phasefree,
       emdq_pf_nstates, emdq_pf_state_names, emdq_pf_rhs!, emdq_pf_rhs, emdq_pf_energy,
       emdq_pf_balance, emdq_pf_phasor_pairs, emdq_pf_rotate, EMDQPFClosedLoop,
       emdq_pf_closed_loop_rhs, emdq_pf_reduce, emdq_pf_lift, emdq_pf_reduced_rhs,
       emdq_pf_quotient_data, emdq_pf_masks, emdq_pf_interval_data, EMDQGovernor, emdq_governed,
       emdq_gov_torque, emdq_gov_rhs, emdq_gov_mode_margins, emdq_bm_signature, emdq_bm_Q,
       emdq_bm_potential, emdq_bm_terms, emdq_bm_balance, emdq_bm_sharp, emdq_bm_rate_check,
       bm_multiplier_feasibility, bm_farkas_check
export EMDQSite, EMDQPilot2, emdq_pilot2, emdq2_grid_layout, emdq2_nc, emdq2_nstates,
       emdq2_grid_state_names, emdq2_grid_rhs!, emdq2_rhs!, emdq2_rhs, emdq2_grid_energy_parts,
       emdq2_grid_energy, emdq2_energy, emdq2_grid_power, emdq2_balance, emdq2_site2_shifted,
       Jet, jet_flow,
       pcc_words, pcc_coefficient_values, pcc_coefficient_jacobian, pcc_two_input_jacobian,
       pcc_structural_zero, pcc_profile
export GridPH, emdq2_grid_ph, grid_ph_to_rotor, grid_ph_from_rotor, grid_ph_JRg, grid_ph_rhs,
       grid_ph_energy, grid_ph_power, ReducedGrid, reduced_grid, reduced_rhs, reduced_project, pcc_L,
       pcc_qorth, pcc_add_pair, pcc_krylov_pool
# physical network (component-first programme)
export PhysicalSteam
export SteamMachineShaft, steam_machine_balance, steam_machine_rhs!, steam_machine_energy,
       steam_machine_shift
export SteamGridUnit, steam_grid_sizes, steam_grid_algebraic, steam_grid_dae, steam_grid_jacobians,
       grid_rotation_direction, case_without_component
export PhysicalNetwork, physical_network, pn_sizes, pn_incidence, pn_flows, pn_energy,
       pn_balance, pn_descriptor, pn_quasistatic_ybus, pn_node_classes
export PCCSplit, PCCReduced, pcc_split, pcc_Q, pcc_rot, pcc_split3, pcc_qappend, pcc_protected, pcc_pod,
       pcc_reduced, pcc_nred, pcc_phi, pcc_project, pcc_rhs, pcc_power
export PHSystem, PHMachine, PHCut, PHInput, PHFault, phs_resolve_clearing, PHRun, ph_system, phs_nstates, phs_state_names,
       phs_flows, phs_rhs!, phs_rhs, phs_energy, phs_gradient, phs_matrices, phs_ledger, phs_network_mask,
       phs_terminal_voltage, phs_cut_voltage, phs_machine_signals, phs_phases, phs_equilibrium,
       phs_scales, phs_step_rule, phs_frame, phs_rotate_network, phs_simulate, phs_simulate_bdf, PHControlled, phs_controlled, phs_fixed_units, phs_full_model,
       phs_control_inputs, phs_control_signals, phs_modes, PHShuntExciter, PHBrushlessExciter,
       phs_write_run, phs_read_run,
       SteamTables, steam_tables, steam_sat, steam_state, steam_h_isentropic,
       PHSteamUnit, steam_initialise, steam_volumes_from_time_constants,
       PHHydroUnit, hydro_initialise, hydro_Tw_match, PHConverter, PHConverterDC, PHVsm, vsm_divider, phs_converter_signals, converter_dc_powers, PHDroop, droop_output, PHVoc, voc_rates, voc_sample!, PHGfl, gfl_call, gfl_sample!, gfl_emf, gfl_bridge, gfl_rates!, gfl_initial,
       PHCore, PHCoreCurve, core_curve_gmd, core_solve, core_energy, core_dgrad, phs_rest_residual, phs_core_signals
# component port passivity
export ComponentPort, component_port, port_transfer, port_hermitian_scan,
       port_witness_interval, polar_transfer, polar_scan, injected_current
# ROA certificates
export ProofFailure, is_interval_type, interval_solve, verified_max_eig, weyl_max_eig, verified_inverse_diagonal,
       ellipsoid_half_widths, SectionModel, section_model, lift, section_coordinates,
       section_field, state_field, project_field, section_residual, section_jacobian,
       LyapunovCandidate, QuadraticCandidate, quadratic_candidate, candidate_model,
       candidate_value, candidate_gradient, positivity_proof, sublevel_half_widths,
       gradient_matrix_hull, candidate_fingerprint, candidate_record, model_fingerprint,
       EquilibriumEnclosure, enclose_equilibrium, KCLBranch, enclose_kcl_branch, krawczyk,
       jacobian_hull, centered_hull, ContainmentAudit, containment_audit, box_digest, certify_level,
       certify_roa, section_record, software_record, write_certificate, eval_param_expr,
       system_digest, interval_cholesky, cholesky_positive_definite, roa_check,
       AbstractSectionModel, AnalyticModel, rotation_action, check_rotation_symmetry,
       certificate_claim, model_assumptions
# power flow
export solve_powerflow, PowerFlowSpec, PowerFlowResult, bus_power, BusType, PQ_BUS, PV_BUS,
       SLACK_BUS

end
