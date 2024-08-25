import sys, cirq, math, logging
import numpy as np

sys.path.append(".")
import argparse
from qruntest.assertion_check_base import assert_base
from qruntest.assurance_ftn import derive_required_num_sample_for_base
from qruntest.hyper_parameters import HypeParamConfig
from qruntest.fpaa_in_matrix import get_prob, get_omega
from qruntest.expectation import SymCostStatisticFPAA
from casestudy_pgm.double_con_draper_adder.double_con_draper_bug import double_con_draper_bug
from casestudy_pgm.double_con_draper_adder.double_con_draper_correct import double_con_draper_true
from casestudy_pgm.draper_adder.draper_bug import draper_buggy
from casestudy_pgm.draper_adder.draper_correct import draper_true
from programs.utils.state_prep_gate_to_qasm import import_spec_data


DELTA = math.sqrt(0.05)


def main(pgm: cirq.Circuit, w: float, confidence: float):  # WLOG we just assume input sv is always zeros
    num_sample_required = math.ceil(derive_required_num_sample_for_base(confidence=confidence, w=w))
    print(f"Derived Required Measurement is {num_sample_required}")
    config = HypeParamConfig(DELTA=None, strat=[], NUM_base_execution=num_sample_required, label=None)
    num_qbit = len(pgm.all_qubits())
    targ = cirq.one_hot(index=0, shape=(2**num_qbit,), dtype=np.complex128)
    prob_zero = get_prob(curr_state=cirq.final_state_vector(pgm), target=targ)
    print(f"Prob Zero is {prob_zero}")
    measurement_trial = assert_base(expect_sv=None, pgm=pgm, config=config)
    if measurement_trial:
        print(f"Num Measurement Untill Bug Detection {measurement_trial}")


if __name__ == "__main__":
    case_study_pgm_mapper = {"draper_bug": draper_buggy, "draper_correct": draper_true, "con_draper_bug": double_con_draper_bug, "con_draper_correct": double_con_draper_true}

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_qasm", default=None, required=False, type=str)
    parser.add_argument("--target_case_study", type=str, required=False, default=None)
    parser.add_argument("--num_qubit", type=int, required=False, default=None)
    parser.add_argument("--t", type=float, required=False, default=None)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--assurance", type=float, required=True)
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
    print(cirq.dirac_notation(cirq.final_state_vector(qc), decimals=3))
    main(pgm=qc, w=args.tolerance, confidence=args.assurance)
