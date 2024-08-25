import sys

sys.path.append(".")
from qruntest.hyper_parameters import HypeParamConfig
from qruntest.score_strat import strat_space, score, derive_l_star, derive_m_star, constraint_for_bug, constraint_for_ver
from qruntest.fpaa_in_matrix import get_omega
from qruntest.assurance_ftn import derive_credible_level
from qruntest.expectation import SymCostStatisticFPAA
import random, math


def main_search(naive_part: int, w: float, confidence: float, l_star: int, m_star: int, seed: int = 2023):
    # naive_strat_config = HypeParamConfig(DELTA=math.sqrt(0.05), strat=[ (0,naive_part), (l_star,m_star)], NUM_base_execution=None)
    strat_for_ref_prob = HypeParamConfig(DELTA=math.sqrt(0.05), strat=[(l_star, m_star)], NUM_base_execution=None)
    print("Strat of Ref Prob is")
    print(strat_for_ref_prob.str_in_tuple())
    # ref_prob_sum =  SymCostStatisticFPAA(prob_measure_zero=(1-w),
    #                                     sym_pgm_cnot_cost=None,
    #                                     sym_pgm_all_gate_cost=None,
    #                                     sym_pgm_moment_cost=None,
    #                                     pgm_qc=None,
    #                                     config = strat_for_ref_prob).event_prob_sum()
    assert naive_part >= 100
    print(f"naive_part = {naive_part}")
    print(f"w = {w}")
    print(f"Seed = {seed}")
    cur_min = 1000000
    num_trial = 500
    print(f"Number of Trial for Search {num_trial}")
    print(f"confidence = {confidence}")
    # print(f"Ref prob sum : {ref_prob_sum}")
    random.seed(seed)
    curr_strat_to_return = None
    for idx, _ in enumerate(range(num_trial)):
        res_strat = strat_space(naive_part=naive_part, l_star=l_star, m_star=m_star)
        cur_score = score(res_strat, w=w)
        if cur_score < cur_min and constraint_for_ver(res_strat, w=w, confidence=confidence) and constraint_for_bug(strat=res_strat, w=w, ref_prob_strat=strat_for_ref_prob):
            print("=======")
            print(res_strat)
            print(f"Found at : {idx}")
            print("Score : ", cur_score)
            cur_min = cur_score
            curr_strat_to_return = res_strat
    return curr_strat_to_return


if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()
    res = main_search(naive_part=100, w=0.001, confidence=0.0001, l_star=34, m_star=6, seed=3000)
    print(res)
