import cirq, sys 
import numpy as np
import math
from  qruntest.utils import *
import matplotlib.pyplot as plt
from typing import List
from qruntest.fpaa_in_matrix import * 
from qruntest.qc_utils import PgmGate


def my_anticon_double_X(main_pgm_qbits, aux_qbit, aux_qbits_for_MCX, detach_head_con_encondes = False, detach_tail_con_encondes = False) -> List[cirq.GateOperation]:
    assert len(aux_qbit) == 1
    from qruntest.qc_utils import my_anticon_con_toff, my_toffoli, my_antianti_con_toff
    assert (detach_head_con_encondes == False and detach_tail_con_encondes == False) or (detach_head_con_encondes^detach_tail_con_encondes == True)

    to_return = list()
    # number of control will be #(main_pgm_qbits-1)
    # and thus number of reuqired ancila will be #(main_pgm_qbits-1)-1
    antianti_controlled_toffoli = my_antianti_con_toff()
    anticontrolled_controlled_toffoli = my_anticon_con_toff()
    if not detach_head_con_encondes :
        for idx, a in enumerate(aux_qbits_for_MCX[:-1]) : 
            if idx == 0 :
                to_return.append(antianti_controlled_toffoli(main_pgm_qbits[0],main_pgm_qbits[1], a))
            else : 
                to_return.append(anticontrolled_controlled_toffoli(main_pgm_qbits[idx+1],aux_qbits_for_MCX[idx-1], a))
    control_val_placeholder = aux_qbits_for_MCX[-2]
    to_return.append(cirq.CNOT(control_val_placeholder, main_pgm_qbits[-1]))
    to_return.append(cirq.CNOT(control_val_placeholder, aux_qbit[0]))
    if not detach_tail_con_encondes :
        for idx, a in reversed(list(enumerate(aux_qbits_for_MCX[:-1]))) : 
            if idx == 0 :
                to_return.append(antianti_controlled_toffoli(main_pgm_qbits[0],main_pgm_qbits[1], a))
            else : 
                to_return.append(anticontrolled_controlled_toffoli(main_pgm_qbits[idx+1],aux_qbits_for_MCX[idx-1], a))
    return to_return


def reflect_source_qc(source_state_qc : cirq.Circuit,
                      alpha : float,
                      main_pgm_qbits,
                      aux_qbit,
                      aux_qbits_for_MCX) ->  List[cirq.GateOperation]:

    assert(isinstance(source_state_qc, cirq.Circuit))
    pgm_as_gate = PgmGate(qc = source_state_qc,
                          num_qbit=len(source_state_qc.all_qubits()))
    pgm_as_gate_inv = PgmGate(qc = (source_state_qc) ** -1,
                                num_qbit=len(source_state_qc.all_qubits()),
                                # in_unitary= cirq.unitary(temp_qc_inv, dtype = np.complex256)
                                )

    # pgm_as_gate = PgmGate(qc = source_state_qc,
    #                       num_qbit=len(source_state_qc.all_qubits()))
    # pgm_as_gate_inv = PgmGate(qc = (source_state_qc) ** -1,
    #                             num_qbit=len(source_state_qc.all_qubits()),
    #                             in_unitary= cirq.unitary(temp_qc_inv)
    #                             )
    

    anti_controlled = cirq.ControlledGate(
                        sub_gate = cirq.XX
                        ,num_controls = len(main_pgm_qbits)-1
                        ,control_values = [ 0 for _ in range(len(main_pgm_qbits)-1)]
                    )
    R_z_minus_alpha_half = cirq.Rz(rads = - alpha / 2 )
    R_z_alpha = cirq.Rz(rads = alpha)

    # return ([pgm_as_gate(*main_pgm_qbits) ** -1
    #         , R_z_minus_alpha_half(main_pgm_qbits[-1])
    #         , anti_controlled(*(main_pgm_qbits+aux_qbit))
    #         , R_z_minus_alpha_half(main_pgm_qbits[-1])
    #         , R_z_minus_alpha_half(aux_qbit[0]) 
    #         , anti_controlled(*(main_pgm_qbits+aux_qbit))
    #        , R_z_alpha(main_pgm_qbits[-1])
    #        , pgm_as_gate(*main_pgm_qbits)]
    #        )
    
    # return ([pgm_as_gate_inv(*main_pgm_qbits)
    #        , R_z_minus_alpha_half(main_pgm_qbits[-1])]
    #        + my_anticon_double_X(main_pgm_qbits=main_pgm_qbits ,aux_qbit=aux_qbit, aux_qbits_for_MCX=aux_qbits_for_MCX, detach_tail_con_encondes=True)
    #         + [ R_z_minus_alpha_half(main_pgm_qbits[-1])
    #         , R_z_minus_alpha_half(aux_qbit[0]) ]
    #         + my_anticon_double_X(main_pgm_qbits=main_pgm_qbits ,aux_qbit=aux_qbit, aux_qbits_for_MCX=aux_qbits_for_MCX, detach_head_con_encondes=True)
    #        + [ R_z_alpha(main_pgm_qbits[-1])
    #        , pgm_as_gate(*main_pgm_qbits)]
    #        )

    return ([pgm_as_gate(*main_pgm_qbits) ** -1
           , R_z_minus_alpha_half(main_pgm_qbits[-1])]
           + my_anticon_double_X(main_pgm_qbits=main_pgm_qbits ,aux_qbit=aux_qbit, aux_qbits_for_MCX=aux_qbits_for_MCX, detach_tail_con_encondes=True)
            + [ R_z_minus_alpha_half(main_pgm_qbits[-1])
            , R_z_minus_alpha_half(aux_qbit[0]) ]
            + my_anticon_double_X(main_pgm_qbits=main_pgm_qbits ,aux_qbit=aux_qbit, aux_qbits_for_MCX=aux_qbits_for_MCX, detach_head_con_encondes=True)
           + [ R_z_alpha(main_pgm_qbits[-1])
           , pgm_as_gate(*main_pgm_qbits)]
           )
    
def reflect_target_qc(beta : float,
                      main_pgm_qbits,
                      aux_qbit,
                      aux_qbits_for_MCX) -> List[cirq.GateOperation]:
    # return [my_oracle_with_aux_for_MCX(main_pgm_qbits=main_pgm_qbits,  aux_qbit = aux_qbit, aux_qbits_for_MCX = aux_qbits_for_MCX)
    #         ,cirq.Rz(rads = beta)(*aux_qbit)
    #         ,my_oracle_with_aux_for_MCX(main_pgm_qbits=main_pgm_qbits, aux_qbit = aux_qbit, aux_qbits_for_MCX = aux_qbits_for_MCX) ]

    return [my_oracle_with_aux_for_MCX(main_pgm_qbits=main_pgm_qbits,  aux_qbit = aux_qbit, aux_qbits_for_MCX = aux_qbits_for_MCX, detach_tail_con_encondes=True)
            ,cirq.Rz(rads = beta)(*aux_qbit)
            ,my_oracle_with_aux_for_MCX(main_pgm_qbits=main_pgm_qbits, aux_qbit = aux_qbit, aux_qbits_for_MCX = aux_qbits_for_MCX, detach_head_con_encondes=True) ]



# def my_oracle(work_qbits, aux_qbit) -> cirq.GateOperation:
#     '''
#     Let U denote the oracle unitary.
#     Should satisfy: for $\ket{b}$ be state for (single) aux qbit,
#     U\ket{0}\ket{b} = \ket{0}\ket{b \oplus 1}
#     and for $k \neq 0$
#     U\ket{k}\ket{b} = \ket{k}\ket{b}
#     '''
#     gate = cirq.ControlledGate(
#         sub_gate = cirq.X,
#         num_controls = len(work_qbits),
#         control_values = [ 0 for _ in work_qbits]
#     )

#     return gate(*(work_qbits + aux_qbit))

def my_oracle_with_aux_for_MCX(main_pgm_qbits, aux_qbit, aux_qbits_for_MCX, detach_head_con_encondes = False, detach_tail_con_encondes = False) -> List[cirq.GateOperation]:
    '''
    an oracle version that requires extra ancila bits 
    for oracle, we need to implement #main_pgm + #aux_qbit-qubit toffoli,
    where #main_pgm will be number of control-bits and #aux_qbit = 1 will be target place.
    by N&C Figure~4.10, the required ancila bit will bie (#main_pgm_qbits-1)
    '''
    assert len(main_pgm_qbits)-1 == len(aux_qbits_for_MCX)
    assert (detach_head_con_encondes == False and detach_tail_con_encondes == False) or (detach_head_con_encondes^detach_tail_con_encondes == True)
    from qruntest.qc_utils import my_anticon_con_toff, my_toffoli, my_antianti_con_toff
    to_return = list()
    # TODO : for debugging prupose : delete later
    # gate = cirq.ControlledGate(
    #    sub_gate = cirq.X,
    #    num_controls = len(main_pgm_qbits),
    #    control_values = [ 0 for _ in main_pgm_qbits]
    # )
    # return gate(*(main_pgm_qbits + aux_qbit))

    
    # antianti_controlled_toffoli = cirq.ControlledGate(sub_gate = cirq.X,
    #                                               num_controls = 2, 
    #                                               control_values = [0,0])
    # anticontrolled_controlled_toffoli = cirq.ControlledGate(sub_gate = cirq.X,
    #                                               num_controls = 2, 
    #                                               control_values = [0,1])
    antianti_controlled_toffoli = my_antianti_con_toff()
    anticontrolled_controlled_toffoli = my_anticon_con_toff()
    if not detach_head_con_encondes : 
        for idx, a in enumerate(aux_qbits_for_MCX) : 
            if idx == 0 :
                to_return.append(antianti_controlled_toffoli(main_pgm_qbits[0],main_pgm_qbits[1], a))
            else :
                to_return.append(anticontrolled_controlled_toffoli(main_pgm_qbits[idx+1],aux_qbits_for_MCX[idx-1], a))
    to_return.append(cirq.CNOT(aux_qbits_for_MCX[-1], aux_qbit[0]))
    if not detach_tail_con_encondes : 
        for idx, a in reversed(list(enumerate(aux_qbits_for_MCX))) : 
            if idx == 0 :
                to_return.append(antianti_controlled_toffoli(main_pgm_qbits[0],main_pgm_qbits[1], a))
            else :
                to_return.append(anticontrolled_controlled_toffoli(main_pgm_qbits[idx+1],aux_qbits_for_MCX[idx-1], a))
    return to_return


def grover_iterate(pgm_qc : cirq.Circuit,
                   main_pgm_qbits,
                   aux_qbit, 
                   aux_qbits_for_MCX, # MCX as abbrev of Multiply-Controlled-NOT
                   alpha : float,
                   beta  : float) -> List[cirq.GateOperation]:
    # the generalized grover iterate by circuit
    return (reflect_target_qc(beta=beta, 
                              main_pgm_qbits=main_pgm_qbits,
                              aux_qbit=aux_qbit,
                              aux_qbits_for_MCX= aux_qbits_for_MCX)
            +reflect_source_qc(source_state_qc = pgm_qc, 
                               alpha=alpha, 
                               main_pgm_qbits=main_pgm_qbits,
                               aux_qbit=aux_qbit,
                               aux_qbits_for_MCX = aux_qbits_for_MCX))

def fpaa_in_circuit(pgm_qc : cirq.Circuit, 
                    l : int, 
                    delta : float,
                    targ_reduce : bool = False) -> cirq.Circuit(): 
    main_pgm_qbits = sorted(pgm_qc.all_qubits(), key = lambda s : s.x)
    aux_qbit = [cirq.LineQubit(len(pgm_qc.all_qubits()))]
    qbits_place_num_for_MCX_aux = len(main_pgm_qbits) + len(aux_qbit)
    aux_qbits_for_MCX =  [ cirq.LineQubit((qbits_place_num_for_MCX_aux-1) + i) for i in range(1, len(main_pgm_qbits)) ] # maybe not used?
    
    assert len(main_pgm_qbits + aux_qbit + aux_qbits_for_MCX ) == len(set(main_pgm_qbits + aux_qbit + aux_qbits_for_MCX))

    fpaa = cirq.Circuit()
    if not targ_reduce :
        alphas, betas = get_phase_anlges(l = l, delta = delta)
    else :
        alphas, betas = get_phase_anlges_rev_aa(l = l, delta = delta)

    for j in range(1, l+1):
        assert(j <= l)
        fpaa.append(grover_iterate(pgm_qc=pgm_qc,
                                   main_pgm_qbits=main_pgm_qbits, 
                                   aux_qbit=aux_qbit, 
                                   aux_qbits_for_MCX=aux_qbits_for_MCX,
                                   alpha=alphas[j], beta=betas[j]))

    return fpaa


def fpaa_qc_experiment_run(pgm_qc : cirq.Circuit, 
                           l : int, 
                           delta : float,
                           targ_reduce : bool = False):
    simulator= cirq.Simulator()
    work_qbits = sorted(pgm_qc.all_qubits(), key = lambda s : s.x)
    aux_qbit = [cirq.LineQubit(len(pgm_qc.all_qubits()))]
    fpaa = cirq.Circuit()
    fpaa.append(pgm_qc) ## TODO enhance here?

    if not targ_reduce :
        alphas, betas = get_phase_anlges(l = l, delta = delta)
    else :
        alphas, betas = get_phase_anlges_rev_aa(l = l, delta = delta)
    target = cirq.one_hot(index=0, shape = (2**(len(pgm_qc.all_qubits())+1),), dtype = np.complex128)

    for j in range(1, l+1):
        assert(j <= l)
        fpaa.append(grover_iterate(pgm_qc=pgm_qc, work_qbits=work_qbits, aux_qbit=aux_qbit, alpha=alphas[j], beta=betas[j]))
    aa_result = simulator.simulate(fpaa, qubit_order=work_qbits+aux_qbit,initial_state= None)
    if not targ_reduce:
        print(f"Probability of measuring {in_dirac_str(target)} on l={l}", get_prob(aa_result.final_state_vector, target) )
    else :
        print(f"Probability of measuring non-target on l={l}", get_prob_non_target(aa_result.final_state_vector, target))

def fpaa_qc_experiment(pgm_qc):
    simulator = cirq.Simulator()
    qbits = cirq.LineQubit.range(4)
    work_qbits = qbits[0:3]
    aux_qbit = qbits[3:]

    target = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    pgm_qc = cirq.Circuit()
    pgm_qc.append([cirq.H(a) for a in work_qbits])
    pgm_qc.append([cirq.CX(work_qbits[i],work_qbits[i+1]) for i in range(len(work_qbits)-1)])

    delta = math.sqrt(0.1)

    for l in range(1,50):
        fpaa = cirq.Circuit()
        fpaa.append(pgm_qc)
        alphas, betas = get_phase_anlges(l = l, delta = delta)
        assert(len(alphas.items()) ==  l)
        assert(len(betas.items())  ==  l)
        for j in range(1, l+1):
            fpaa.append(grover_iterate(pgm_qc=pgm_qc, work_qbits=work_qbits,aux_qbit=aux_qbit,alpha=alphas[j], beta=betas[j]))
        aa_result = simulator.simulate(fpaa,qubit_order=work_qbits+aux_qbit,initial_state= None)
        prob = get_prob(aa_result.final_state_vector, target)  

    

