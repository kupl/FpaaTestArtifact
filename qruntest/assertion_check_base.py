import cirq
import numpy as np
from qruntest.utils import *
from qruntest.fpaa_circuit import *
from qruntest.hyper_parameters import HypeParamConfig


def assert_base(
    expect_sv: np.ndarray,  # assumed to be zeros ; see apendix
    pgm: cirq.Circuit,  # assumed that input states to be zeros ; see apendix
    unitary_tranasform: cirq.Circuit = None,
    do_debug=False,
    only_get_cost=False,
    config: HypeParamConfig = None,
) -> Tuple:

    main_qbits_one = sorted(pgm.all_qubits(), key=lambda s: s.x)
    main_qbits_two = cirq.LineQubit.range(len(main_qbits_one), 2 * len(main_qbits_one))

    pgm_as_gate = PgmGate(pgm, num_qbit=len(pgm.all_qubits()))
    runtime_check_qc_builder = cirq.Circuit()
    runtime_check_qc_builder.append(pgm_as_gate(*main_qbits_one))
    runtime_check_qc_builder.append(pgm_as_gate(*main_qbits_two))
    runtime_check_qc_builder.append(cirq.measure(*(main_qbits_one + main_qbits_two)))
    print("running SAMPLE")
    for i in range(int(config.NUM_base_execution / 2)):
        simulator = cirq.Simulator()
        result = simulator.run(runtime_check_qc_builder, repetitions=1)
        does_pass_for_curr_iterate = is_all_zero(result=result, main_qbits=main_qbits_one + main_qbits_two)
        if does_pass_for_curr_iterate:
            pass
        else:
            print("BUG DETECTED!")
            print(f"Measurement data was : {str(result)}")
            return i * 2  # for each loop, we run circuit twice in a parallel, hence multiply 2 for derving number of P execution
    print("Assertion check passed")
    print("Given Program is NOT buggy")
    return
