import cirq
import sys
sys.path.append(".")
import numpy as np
from qruntest.utils import *
from casestudy_pgm.draper_related_utils import *


def R_k_gate(k : int):
    t = 2 / 2 ** k  
    return cirq.ZPowGate(exponent = t)

def draper_true(m : int) -> cirq.Circuit : # n for number of bits for a,b
    b_qbits = cirq.LineQubit.range(m)
    a_qbits = cirq.LineQubit.range(m, 2 * m)
    pgm_qc = cirq.Circuit()
    
    QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(m, without_reverse=True)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)+1)))):
            controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 1)
            pgm_qc.append(controlled_R(b_i,a_qbits[idx_app]))
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc


def my_draper_true(m : int) -> cirq.Circuit : # n for number of bits for a,b
    b_qbits = cirq.LineQubit.range(m)
    a_qbits = cirq.LineQubit.range(m, 2 * m)
    pgm_qc = cirq.Circuit()
    
    QFT_WO_REV = my_QFT(m)
    pgm_qc.append(cirq.Circuit(QFT_WO_REV(*a_qbits)))
    for b_i in b_qbits:
        for idx_app, j in enumerate(reversed(list(range(m- (m -b_i.x)+1)))):
            controlled_R = my_C_R_k_gate(j+1)
            # controlled_R  = cirq.ControlledGate(sub_gate = R_k_gate(j+1), num_controls = 1)
            pgm_qc.append(controlled_R(b_i,a_qbits[idx_app]))
    pgm_qc.append(cirq.Circuit((QFT_WO_REV ** -1)(*a_qbits)))
    return pgm_qc
        
if __name__=='__main__':
    print(draper_true(4))