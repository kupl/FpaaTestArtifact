from sympy import symbols
from typing import Tuple, List
import sympy
### Our Basic Operations = { CNOT }


### following is all all-gate cost (1qubit+2qubit) 



### following is all CNOT cost
def sym_static_cost_cnot_one_grover_iterate() -> sympy.core.basic.Basic:
    TOFF_CNOT_COST = 6
    n = symbols("n") # num of progam qubit
    pgm_cost = symbols("x")
    # nplusone_TOFFOLI_without_anc =  2 * ((n+1) **2) -6*(n+1) + 5  # in two-qubit gate 
    # n_TOFFOLI_without_anc =  2 * ( (n) **2) -6*(n) + 5            # in two-qubit gate
    expr = 0 
    ## S_t (target rotate) part, two oracle is applied across n+1 qubit 
    # expr += 2 * nplusone_TOFFOLI_without_anc
    # expr += 2 * nplusone_TOFFOLI_without_anc
    expr += TOFF_CNOT_COST * 2 * (n-1) + 1 
    expr +=1 
    ## S_s (source rotate) part
    expr += pgm_cost
    # expr += 2 * n_TOFFOLI_without_anc
    expr += TOFF_CNOT_COST * 2 * (n-2) +2
    expr +=2
    # expr += 2 * n_TOFFOLI_without_anc
    expr += pgm_cost
    return expr

def sym_static_all_gate_cost_one_grover_iterate() -> sympy.core.basic.Basic:
    TOFF_ALL_GATE_COST = 16
    n = symbols("n") # num of progam qubit
    pgm_cost = symbols("x")
    expr_for_target_relfect, expr_for_source_relfect = 0, 0
    expr_for_target_relfect += 2 * (4 + TOFF_ALL_GATE_COST)   # two antianti-toff.
    expr_for_target_relfect += 2 * (n-2) * (2 + TOFF_ALL_GATE_COST)   # other anti-con toff
    expr_for_target_relfect += 1 # one phased by beta
    expr_for_target_relfect += 2 # two cnot
    
    expr_for_source_relfect += 1 # one phase by alpha
    expr_for_source_relfect += pgm_cost
    expr_for_source_relfect += 2 * (4 + TOFF_ALL_GATE_COST)
    expr_for_source_relfect += 2 * (n-3) * (2 + TOFF_ALL_GATE_COST)  
    expr_for_source_relfect += 4 + 2 #four cnot and two phases by alpha
    expr_for_source_relfect += pgm_cost
    expr_for_source_relfect += 1 # one phase by alpha
    expr = expr_for_target_relfect + expr_for_source_relfect
    return expr    

def sym_static_moment_cost_one_grover_iterate() -> sympy.core.basic.Basic:
    num_qbit = symbols("n")
    pgm_cost = symbols("x") 
    
    # return 2*pgm_cost + 4* ( 2*(num_qbit **2)    + 10*num_qbit)
    expr_for_target_reflect = 194 + (num_qbit/2 -5) * 42 
    expr_for_source_reflect = 2 * pgm_cost + 90 + (num_qbit / 2  - 3) * 42
    
    return expr_for_target_reflect + expr_for_source_reflect

def sym_static_cnot_cost_fpaa_by_ord(ord : Tuple[int,int], l_list, m_list, hyp_param_seq) -> sympy.core.basic.Basic:
   # input : stopping order
    assert ord in hyp_param_seq
    assert len(l_list) == len(m_list)
    m = ord[1]
    l = ord[0]
    assert l in l_list
    num_qbit = symbols("n")
    pgm_cost = symbols("x") 
    expr = 0
    for prv_l_idx, prev_l in enumerate(l_list) :
        if prev_l != l:
            corr_m = m_list[prv_l_idx]
            expr += corr_m * (pgm_cost + (prev_l  *  sym_static_cost_cnot_one_grover_iterate() ))
        elif prev_l == l :
            expr += m      * (pgm_cost + (l  *  sym_static_cost_cnot_one_grover_iterate() ))
            break
    return expr 

def sym_static_all_gate_cost_fpaa_by_ord(ord : Tuple[int,int], l_list, m_list, hyp_param_seq) -> sympy.core.basic.Basic:
    assert ord in hyp_param_seq
    assert len(l_list) == len(m_list)
    m = ord[1]
    l = ord[0]
    assert l in l_list
    num_qbit = symbols("n")
    pgm_cost = symbols("x") 
    expr = 0
    for prv_l_idx, prev_l in enumerate(l_list) :
        if prev_l != l:
            corr_m = m_list[prv_l_idx]
            expr += corr_m * (pgm_cost + (prev_l  *  sym_static_all_gate_cost_one_grover_iterate() ))
        elif prev_l == l :
            expr += m      * (pgm_cost + (l  *  sym_static_all_gate_cost_one_grover_iterate() ))
            break
    return expr 

def old_sym_static_moment_cost_fpaa_by_ord(ord : Tuple[int,int], l_list, m_list, hyp_param_seq) -> sympy.core.basic.Basic:
    raise NotImplementedError
    assert ord in hyp_param_seq
    assert len(l_list) == len(m_list)
    m = ord[1]
    l = ord[0]
    assert l in l_list
    num_qbit = symbols("n")
    pgm_cost = symbols("x") 
    expr = 0
    for prv_l_idx, prev_l in enumerate(l_list) :
        if prev_l != l:
            corr_m = m_list[prv_l_idx]
            expr += corr_m * (pgm_cost + (prev_l  *  sym_static_moment_cost_one_grover_iterate() ))
        elif prev_l == l :
            expr += m      * (pgm_cost + (l  *  sym_static_moment_cost_one_grover_iterate() ))
            break
    return expr 

def sym_static_moment_cost_fpaa_by_ord(ord_idx : int,  hyp_param_seq : List[Tuple[int,int]]) -> sympy.core.basic.Basic:
    assert ord_idx <= len(hyp_param_seq)-1
    num_qbit = symbols("n")
    pgm_cost = symbols("x") 
    expr = 0
    for idx, ord_elt in enumerate(hyp_param_seq):
        if idx == (ord_idx+1) : break
        l, _ = ord_elt
        if l ==  0 : expr += (pgm_cost + (l   *  sym_static_moment_cost_one_grover_iterate() ))/2 # of course multiple by zero is zero
        else : expr += (pgm_cost + (l   *  sym_static_moment_cost_one_grover_iterate() ))
    return expr

def sym_cnot_cost_for_my_draper() -> sympy.core.basic.Basic:
    num_qbit_for_pgm = symbols("n") 
    num_qbit_for_qft = symbols("k")
    qft_cost = 0
    # drpaer utilizes two qft (one for inverse version, but same cost)
    qft_cost += (num_qbit_for_qft ** 2 -num_qbit_for_qft)
    qft_cost += (num_qbit_for_qft ** 2 -num_qbit_for_qft) 
    # for cphases in draper (sandwhiched by qft), it utilizes k(k+1)/2 many cphases and among them, there are k-many CZ.
    # following formula relfecs it 
    cphases_cost = 2 * ( num_qbit_for_qft*(num_qbit_for_qft+1)/2 - num_qbit_for_qft)  + num_qbit_for_qft
    expr = qft_cost + cphases_cost
    expr = expr.subs({num_qbit_for_qft : num_qbit_for_pgm/2 })
    return expr

def sym_all_gate_cost_for_my_draper(is_bug = False):
    # gate count model expressed by formula in $n$
    # the basis count is assume in 1-qubit gate + CNOT
    # initially, don't assume gate merge between 1-qubi gates
    num_qbit_for_pgm = symbols("n") 
    num_qbit_for_qft = symbols("k")
    qft_cost = 2 * (num_qbit_for_qft + 5 * ((num_qbit_for_qft**2 -num_qbit_for_qft)/ 2))
    cphases_cost =  5 * ( num_qbit_for_qft*(num_qbit_for_qft+1)/2 - num_qbit_for_qft)  + 3* num_qbit_for_qft
    expr = qft_cost +  cphases_cost
    expr = expr.subs({num_qbit_for_qft : num_qbit_for_pgm/2 })
    if not is_bug : 
        return expr
    else :
        return expr + 1

def sym_moment_cost_for_my_draper(is_bug = False):
    num_qbit_for_pgm = symbols("n") 
    expr = 20 + (num_qbit_for_pgm/2 - 2) * 20
    if not is_bug : 
        return expr
    else :
        return expr + 1
    
def sym_moment_cost_for_my_double_con_draper(is_bug= False):
    num_qbit_for_pgm = symbols("n") 
    fmla = (1/4) * ( 15 * (num_qbit_for_pgm ** 2) - 6 * num_qbit_for_pgm - 68)
    return fmla

def sym_moment_cost_for_my_temp(is_bug= False):
    num_qbit_for_pgm = symbols("n") 
    fmla = (1/2) * ( 15 * (num_qbit_for_pgm ** 2) - 6 * num_qbit_for_pgm - 68)
    return fmla