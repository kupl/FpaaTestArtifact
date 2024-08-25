import cirq
import sys
sys.path.append(".")
import numpy as np
from qruntest.utils import *
from qruntest.assertion_check_fpaa import assert_check_fpaa
from casestudy_pgm.draper_related_utils import *


def draper_buggy(m : int, bug_control_param ) -> cirq.Circuit : # n for number of bits for a,b
    # bug_control_param \in [-1,1]
    b_qbits = cirq.LineQubit.range(m)
    a_qbits = cirq.LineQubit.range(m, 2 * m)
    pgm_qc = cirq.Circuit()
    QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(m, without_reverse=True)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)+1)))):
            controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 1)
            pgm_qc.append(controlled_R(b_i,a_qbits[idx_app]))
    pgm_qc.append((cirq.Z ** np.sin(math.pi * bug_control_param))(a_qbits[1])) ## bug injected!!
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc
    
def my_draper_buggy(m : int, bug_control_param) -> cirq.Circuit : # n for number of bits for a,b
    # bug_control_param \in [-1,1]

    b_qbits = cirq.LineQubit.range(m)
    a_qbits = cirq.LineQubit.range(m, 2 * m)
    pgm_qc = cirq.Circuit()
    # QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(m, without_reverse=True)
    QFT_WO_REV = my_QFT(m)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)+1)))):
            controlled_R = my_C_R_k_gate(j+1)
            # controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 1)
            pgm_qc.append(controlled_R(b_i,a_qbits[idx_app]))
    pgm_qc.append((cirq.Z ** np.sin(math.pi * bug_control_param))(a_qbits[1])) ## bug injected!!
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc

if __name__=='__main__':
    pass