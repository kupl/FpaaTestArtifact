import cirq, qiskit, tomlkit
import numpy as np
import qiskit.circuit
from qiskit.compiler import transpile
from qiskit import qasm3, qasm2
from cirq.contrib.qasm_import import circuit_from_qasm


def state_prep_gate_to_qasm(E_state: np.array):
    num_qubit = int(np.log2(len(E_state)))
    state_prep_qc = qiskit.circuit.QuantumCircuit(num_qubit)
    state_prep_qc.initialize(E_state, qubits=list(range(num_qubit)))
    state_prep_qc = transpile(state_prep_qc, basis_gates=["cx", "rx", "ry", "rz"])

    state_prep_qc_wo_reset = qiskit.circuit.QuantumCircuit(num_qubit)
    for x in state_prep_qc:
        if x.operation.name != "reset":
            state_prep_qc_wo_reset.append(x)
    inverted = state_prep_qc_wo_reset.inverse()
    qasm_res = qasm2.dumps(inverted)
    cirq_qc = circuit_from_qasm(qasm_res)
    return cirq_qc


def append_state_prep_gate(main_qc: cirq.Circuit, E_state):  # this will ensure that WLOG $|E>$ to be zeros(i.e, |00..0>)
    copied_qc = main_qc.copy()
    to_append = state_prep_gate_to_qasm(E_state=E_state)
    for m in to_append:
        copied_qc.append(m)
    return copied_qc


def import_spec_data(spec_id: str) -> cirq.Circuit:

    def name_qbit_to_line_qbit(qbit_name: str) -> int:
        qubit_num_id = int(qbit_name.split("_")[-1])
        return qubit_num_id

    with open(f"./programs/{spec_id}.toml", "rb") as f:
        data = tomlkit.load(f)
        pgm_dir = data["targ_pgm_QASM"]
        f = open(pgm_dir, "r")
        qc_to_test = circuit_from_qasm(f.read())
        f.close()
        E_state = np.array(eval(data["E_state"]))
        print(f"Specifed E state was {cirq.dirac_notation(E_state)}")
        final_qc_to_test = append_state_prep_gate(main_qc=qc_to_test, E_state=E_state)

        # conversion to line_qubit
        line_qubit_ver = cirq.Circuit()
        line_qbits = cirq.LineQubit.range(len(final_qc_to_test.all_qubits()))

        for gate_op in final_qc_to_test.all_operations():
            name_qubits_in_linequbit = [line_qbits[name_qbit_to_line_qbit(q.name)] for q in gate_op.qubits]
            to_append = gate_op.gate(*name_qubits_in_linequbit)
            line_qubit_ver.append(to_append)
        return line_qubit_ver
