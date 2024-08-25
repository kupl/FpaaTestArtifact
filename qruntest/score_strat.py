from qruntest.hyper_parameters import HypeParamConfig
from qruntest.expectation import SymCostStatisticFPAA
from qruntest.fpaa_in_matrix import get_omega, prob_L
from qruntest.assurance_ftn import derive_credible_level
from qruntest.assurance_ftn import sym_P_L
import scipy.integrate as integrate
from typing import Tuple, List
import sympy, random, math, cirq
import numpy as np

def strat_space(naive_part : int, l_star : int, m_star : int, delta = math.sqrt(0.05)) -> List[HypeParamConfig]:
    from random import randrange
    max_len_of_strat = randrange(1,10)
    strat_builder       = list()
    curr_l = 0
    limit_l = l_star -1
    for _ in range(max_len_of_strat):
        l_to_append = randrange(start=min(curr_l + 1, limit_l-1), stop= limit_l)
        m_to_append = randrange(1,7)
        strat_builder.append((l_to_append,m_to_append ))
        curr_l = l_to_append
        if (l_to_append > l_star -1) or curr_l+1 >= l_star-5:
            break
    # strat_builder.append((l_star, m_star))
    strat_builder.insert(0, (0, naive_part))
    return HypeParamConfig(DELTA= delta, strat=strat_builder, NUM_base_execution=None)

def expected_num_g_by_b(b:float, strat : HypeParamConfig):
    cost_model = SymCostStatisticFPAA(prob_measure_zero=(1-b),
                                    sym_pgm_cnot_cost=None,
                                    sym_pgm_all_gate_cost=None,
                                    sym_pgm_moment_cost=None,
                                    pgm_qc=None,
                                    config = strat)
    return cost_model.expected_num_g_iterate()


def score_bug(strat : HypeParamConfig, w : float) :
    assert 1 > w and w> 0 
    by_b = lambda x : expected_num_g_by_b(b=x, strat=strat)
    range_to_integral = 1
    assert w < range_to_integral and 3*w < range_to_integral
    return (integrate.quad(by_b, w ,3*w)[0]  + integrate.quad(by_b, 3*w ,range_to_integral)[0] ) / (range_to_integral-w)

def score_ver(strat : HypeParamConfig) -> int:
    score = 0
    for x in strat.strat:
        score += x[0] * x[1]
    return score


def score(strat : HypeParamConfig, w : float)-> float :
    # return score_bug(strat, w=w)
    l_star, m_star = strat.strat[-1]
    if score_ver(strat) <= l_star * (m_star * 1.8) :
        return score_bug(strat, w=w) 
    else : return 100000


def event_dist(l :int, m : int, b: float) :
    prob_l_res = prob_L(delta=math.sqrt(0.05), lamb = b, L = 2*l+1)
    return (1-prob_l_res) ** m


def derive_l_star(w: float, delta=math.sqrt(0.05)): 
    for l in range(1000): 
        omega = get_omega(L = 2*l+1, delta = delta)
        if omega < w : 
            return l

def derive_m_star(l_star : int, confidence : float, w: float, naive_part : int,  delta = math.sqrt(0.05)): 
    for m in range(4, 10):
        print(f"Assurance for m = {m}")
        temp_strat = HypeParamConfig(DELTA=delta, strat=[(0,naive_part), (l_star, m)], NUM_base_execution=None, label=None)
        const_res = constraint_for_ver(strat=temp_strat, w = w, confidence=confidence)
        if const_res : return m
    raise AttributeError


def constraint_for_bug(strat : HypeParamConfig, w : float, ref_prob_strat : float):
    # for b in [w, 5*w, 1] :
    for b in np.linspace(start=w, stop=1,num=100, endpoint=True) :
        cost_model = SymCostStatisticFPAA(prob_measure_zero=(1-b),
                                        sym_pgm_cnot_cost=None,
                                        sym_pgm_all_gate_cost=None,
                                        sym_pgm_moment_cost=None,
                                        pgm_qc=None,
                                        config = strat)
        ref_prob_sum = SymCostStatisticFPAA(prob_measure_zero=(1-b),
                                        sym_pgm_cnot_cost=None,
                                        sym_pgm_all_gate_cost=None,
                                        sym_pgm_moment_cost=None,
                                        pgm_qc=None,
                                        config = ref_prob_strat).event_prob_sum()
        event_prob_sum = cost_model.event_prob_sum()
        # if not ref_prob_sum <= event_prob_sum : return False
        if not (math.isclose(ref_prob_sum , event_prob_sum, abs_tol =1e-8) or ref_prob_sum <= event_prob_sum ) : return False

    return True

def constraint_for_ver(strat : HypeParamConfig, w: float,  confidence :float):
    
    def val_by_b(b : float) : 
        builder = 1
        for l_idx, l in enumerate(strat.l_SEQ) :
            builder = builder * event_dist(l =l, m= strat.NUM_measure_per_l[l_idx], b=b)
        return builder 
    
    # numerator = integrate.quad(val_by_b, w, 2*w)[0] + integrate.quad(val_by_b, 2*w,3*w)[0] + integrate.quad(val_by_b, 3*w,1)[0]
    numerator = integrate.quad(val_by_b, w,1)[0] 
    # this integral may not be 100% accurate, due to inherent limitaiton in numertical algorithm. However, by reference to wolframalpha, this was best choice as setting
    marignal_constnat = integrate.quad(val_by_b, 0,w)[0] + integrate.quad(val_by_b, w,2*w)[0] + integrate.quad(val_by_b, 2*w,3*w)[0] + integrate.quad(val_by_b, 3*w,1)[0]
    derived_confi_res = numerator / marignal_constnat
    # print(derived_confi_res)
    if derived_confi_res >1 :
        return False
    if derived_confi_res < confidence : 
        return True
    else : #try in correct value which gives in
        return False
    
