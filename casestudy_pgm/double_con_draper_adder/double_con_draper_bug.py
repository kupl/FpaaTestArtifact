import cirq
import sys
sys.path.append(".")
import numpy as np
from qruntest.utils import *
from casestudy_pgm.draper_related_utils import *

def double_con_draper_bug(m : int, bug_control_param) -> cirq.Circuit : # m for number of bits for each a,b
    con     = cirq.LineQubit.range(2)
    b_qbits = cirq.LineQubit.range(2, m+2)
    a_qbits = cirq.LineQubit.range(m+2, 2 *m+2)
    
    pgm_qc = cirq.Circuit()
    QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(m, without_reverse=True)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)-1 )))):
            triple_controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 3)
            pgm_qc.append(triple_controlled_R(con[0], con[1], b_i, a_qbits[idx_app]))
    pgm_qc.append((cirq.Z ** np.sin(math.pi * bug_control_param))(a_qbits[1])) ## bug injected!!
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc



def my_double_con_draper_bug(m : int, bug_control_param) -> cirq.Circuit : # n for number of bits for a,b
    con     = cirq.LineQubit.range(2)
    b_qbits = cirq.LineQubit.range(2, m+2)
    a_qbits = cirq.LineQubit.range(m+2, 2 *m+2)
    
    pgm_qc = cirq.Circuit()
    # QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(m, without_reverse=True)
    QFT_WO_REV = my_QFT(m)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)-1 )))):
            # triple_controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 3)
            triple_controlled_R  = my_triple_C_R_k_gate(k=j+1)
            pgm_qc.append(triple_controlled_R(con[0], con[1], b_i, a_qbits[idx_app]))
    pgm_qc.append((cirq.Z ** np.sin(math.pi * bug_control_param))(a_qbits[1])) ## bug injected!!
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc
        


if __name__=='__main__':
    pass