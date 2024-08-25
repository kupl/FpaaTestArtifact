import cirq
import numpy as np
from qruntest.utils import is_all_zero
from qruntest.fpaa_circuit import *
from qruntest.hyper_parameters import HypeParamConfig
from qruntest.utils import my_sub_state_vector, keep, circuit_merge_policy_for_count_metric
from qruntest.expectation import SymCostStatisticFPAA
from sympy import symbols
import logging


simulator = cirq.Simulator()


def assert_check_fpaa(
    expect_sv: np.ndarray,  # assumed to be zeros ; see apendix
    pgm: cirq.Circuit,  # assumed that input states to be zeros ; see apendix
    config: HypeParamConfig,
    logging_option=None,
    do_debug=False,
    do_without_simul=False,
) -> Tuple:
    # by the new policy, additional unitary transformation is not required as we will assume that it is already attached on pgm
    assert config
    assert pgm

    logger = logging.getLogger()
    logger.setLevel(logging_option)
    formatter = logging.Formatter("%(levelname)s : %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Do l-variant fpaa-based assertion check")
    logger.info("Hyper Paramter Settings")
    logger.info("{:<20} : {:<5}".format("DETLA", config.DELTA))
    logger.info("{:<20} :".format("STRAT"))
    logger.info(str(config.strat))

    if not do_without_simul:
        logger.info("{:<13} : {:<5}".format("Init SV", cirq.dirac_notation(cirq.final_state_vector(pgm), decimals=5)))
        init_sv = cirq.final_state_vector(pgm)
        init_sv_meas_non_zero = get_prob_non_target(init_sv, target=cirq.one_hot(index=0, shape=init_sv.shape, dtype=init_sv.dtype))

    num_pgm_qbits = len(pgm.all_qubits())

    all_gate_execution_count = 0
    cnot_gate_execution_count = 0
    moment_execution_count = 0
    iterate_num_cal = 0

    for idx, l in enumerate(config.l_SEQ):
        pgm_qbits = sorted(pgm.all_qubits(), key=lambda s: s.x)
        copied_pgm = pgm.copy()
        pgm_as_gate = PgmGate(copied_pgm, num_qbit=len(pgm.all_qubits()))
        runtime_check_qc_builder = cirq.Circuit()  ## IMPORTANT : NEW INIT
        runtime_check_qc_builder.append(pgm_as_gate(*pgm_qbits))
        logger.debug(f"\n=======================\nTrying for l-value {l}\n=======================")
        omega = get_omega(L=2 * l + 1, delta=config.DELTA)
        logger.debug("{:<20} : {:<5}".format("ω", omega))
        logger.debug("{:<20} : {:<5}".format("ω_in_amplitude", math.sqrt(omega)))
        fpaa_part = fpaa_in_circuit(pgm_qc=pgm, l=l, delta=config.DELTA, targ_reduce=True)  # give input as cirq.Circuit, not packaged into PgmGate
        runtime_check_qc_builder.append(fpaa_part)

        def assert_all_2qubit_is_cnot(decomposed_qc: cirq.Circuit()):
            for x in decomposed_qc.all_operations():
                if len(x.qubits) > 3:
                    return False
                elif len(x.qubits) == 1:
                    pass
                elif len(x.qubits) == 2:
                    if not (x.gate == cirq.CNOT or x.gate == cirq.CX):
                        return False
            return True

        assert len(runtime_check_qc_builder.all_qubits()) == num_pgm_qbits or len(runtime_check_qc_builder.all_qubits()) == num_pgm_qbits + (num_pgm_qbits - 1) + 1

        runtime_check_qc_builder.append(cirq.measure(*pgm_qbits))
        if not do_without_simul:
            curr_sv = cirq.final_state_vector(cirq.drop_terminal_measurements(runtime_check_qc_builder), dtype=np.complex128)  # for debugging purpose
            targ = cirq.one_hot(index=0, shape=curr_sv.shape, dtype=np.complex128)  # defined only for debug, TODO : delete later
            logger.debug("{:<15} : {:<5}".format("State After Amplified", cirq.dirac_notation(curr_sv, decimals=7)))
            logger.debug("{:<50} : {:<10}".format("Measuring non-targ", get_prob_non_target(curr_sv, target=targ)))
            # logger.debug("{:<13} : {:<5}".format("Measuring targ", get_prob(curr_sv, target = targ)))
            logger.debug("{:<50} : {:<10}".format("P_L (must be coincide with above)", prob_L(delta=config.DELTA, lamb=init_sv_meas_non_zero, L=2 * l + 1)))
            if l != 0:
                my_sub_state_vector(curr_sv, keep_indices=list(range(num_pgm_qbits)), atol=0.00005)  # practical atol according to our usage

        for curr_num_meas in range(1, config.NUM_measure_per_l[idx] + 1):
            curr_seq = (l, curr_num_meas)
            assert curr_seq in config.sequence

            if not do_without_simul:
                result = simulator.run(runtime_check_qc_builder, repetitions=1)
                does_pass_for_curr_iterate = is_all_zero(result=result, main_qbits=pgm_qbits)
                if not does_pass_for_curr_iterate:
                    print(f"BUG DETECTED! END. The l-value was {l} and sequence (starting from 0) is {iterate_num_cal}")
                    if not do_debug:
                        return l, iterate_num_cal, all_gate_execution_count, cnot_gate_execution_count, moment_execution_count
            iterate_num_cal += 1

    if not do_without_simul and not do_debug:
        print("Given Program is NOT buggy")
    else:
        print("The routine is ended. It was [ do_without_simul = False ] mode")
    return None, None, all_gate_execution_count, cnot_gate_execution_count, moment_execution_count
