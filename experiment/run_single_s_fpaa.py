import sys, cirq, math, logging
import numpy as np

sys.path.append(".")
import argparse
from qruntest.assertion_check_fpaa import assert_check_fpaa
from qruntest.hyper_parameters import HypeParamConfig
from qruntest.fpaa_in_matrix import get_prob, get_omega
from experiment.run_strat_search import main_search, derive_l_star, derive_m_star
from casestudy_pgm.double_con_draper_adder.double_con_draper_bug import double_con_draper_bug
from casestudy_pgm.double_con_draper_adder.double_con_draper_correct import double_con_draper_true
from casestudy_pgm.draper_adder.draper_bug import draper_buggy
from casestudy_pgm.draper_adder.draper_correct import draper_true
from programs.utils.state_prep_gate_to_qasm import import_spec_data

from qruntest.expectation import SymCostStatisticFPAA
import importlib


DELTA = math.sqrt(0.05)


def main(pgm: cirq.Circuit, w: float, confidence: float, naive_part: int, seed: int):  # WLOG we just assume input sv is always zeros
    l_star = derive_l_star(w=w)
    m_star = derive_m_star(l_star=l_star, confidence=confidence, w=w, naive_part=0, delta=DELTA)
    print(f"Derived l ={l_star} and {m_star} for Reference Prob")
    num_qbit = len(pgm.all_qubits())
    targ = cirq.one_hot(index=0, shape=(2**num_qbit,), dtype=np.complex128)
    prob_zero = get_prob(curr_state=cirq.final_state_vector(pgm), target=targ)
    print(f"Prob Zero is {prob_zero}")
    found_strat = main_search(naive_part=naive_part, w=w, confidence=confidence, l_star=l_star, m_star=m_star, seed=seed)
    print("Found Strat is")
    print(str(found_strat))
    assert_check_fpaa(expect_sv=None, pgm=pgm, config=found_strat, logging_option=logging.INFO)


if __name__ == "__main__":
    case_study_pgm_mapper = {"draper_bug": draper_buggy, "draper_correct": draper_true, "con_draper_bug": double_con_draper_bug, "con_draper_correct": double_con_draper_true}

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_qasm", default=None, required=False, type=str)
    parser.add_argument("--target_case_study", type=str, required=False, default=None)
    parser.add_argument("--num_qubit", type=int, required=False, default=None)
    parser.add_argument("--t", type=float, required=False, default=None)
    parser.add_argument("--naive_part", type=int, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--assurance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    assert not ((args.target_case_study != None) ^ (args.num_qubit != None))
    assert (args.from_qasm != None) ^ (args.target_case_study != None)

    if args.from_qasm:
        qc = import_spec_data(spec_id=args.from_qasm)
    else:
        if args.target_case_study in ["draper_correct", "con_draper_correct"]:
            if args.t:
                raise ValueError("t value shall not be given for correct implementation testing")
            if args.target_case_study == "draper_correct":
                assert args.num_qubit >= 4
                qc = case_study_pgm_mapper[args.target_case_study](int(args.num_qubit / 2))
            elif args.target_case_study == "con_draper_correct":
                assert args.num_qubit >= 6
                qc = case_study_pgm_mapper[args.target_case_study](int((args.num_qubit - 2) / 2))
        elif args.target_case_study in ["draper_bug", "con_draper_bug"]:
            if not args.t:
                raise ValueError("For buggy case study program, t value should be specified.")
            if args.target_case_study == "draper_bug":
                assert args.num_qubit >= 4
                qc = case_study_pgm_mapper[args.target_case_study](int(args.num_qubit / 2), np.arcsin(args.t / math.pi))
            elif args.target_case_study == "con_draper_bug":
                assert args.num_qubit >= 6
                qc = case_study_pgm_mapper[args.target_case_study](int((args.num_qubit - 2) / 2), np.arcsin(args.t / math.pi))
        else:
            raise NotImplementedError(f"Program {args.target_case_study} is not valid case study program name")

    print("Target quantum circuit program to test")
    print(qc)
    print("check final state vector (note we wlog assume |E>=|00..0>)")
    print(cirq.dirac_notation(cirq.final_state_vector(qc)))

    main(pgm=qc, w=args.tolerance, confidence=args.assurance, naive_part=args.naive_part, seed=args.seed)
