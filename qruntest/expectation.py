import cirq, numpy
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from qruntest.utils import prob_L
from qruntest.fpaa_in_matrix import get_prob_non_target, get_prob
from qruntest.qc_utils import PgmGate
from qruntest.hyper_parameters import HypeParamConfig
from qruntest.utils import keep
import sympy
import math


######################################
# below symbolic version of cost model


class SymCostStatisticBase:  # for symolic verison, we only must care the probability value, not as in amplitude

    def __init__(
        self,
        prob_measure_zero: float,
        sym_pgm_cnot_cost: sympy.core.basic.Basic,
        sym_pgm_all_gate_cost: sympy.core.basic.Basic,
        sym_pgm_moment_cost: sympy.core.basic.Basic,
        pgm_qc: cirq.Circuit,
        config: HypeParamConfig,
    ) -> None:
        self.zero_measure_prob = prob_measure_zero
        self.sym_pgm_cnot_cost = sym_pgm_cnot_cost
        self.sym_pgm_all_gate_cost = sym_pgm_all_gate_cost
        self.sym_pgm_moment_cost = sym_pgm_moment_cost
        self.is_true_zero_case = math.isclose(a=prob_measure_zero, b=1, abs_tol=1e-06)
        self.config = config

        if sym_pgm_cnot_cost:
            assert sympy.symbols("n") in sym_pgm_cnot_cost.free_symbols
        if sym_pgm_all_gate_cost:
            assert sympy.symbols("n") in sym_pgm_all_gate_cost.free_symbols

        # assert 0.7<= self.zero_measure_prob and self.zero_measure_prob <= 1 # this assertion only for debugging purpose
        # self.pgm_qc = pgm_qc
        # temp_qc = cirq.Circuit([x for x in cirq.decompose(pgm_qc, keep=keep)])
        # self.moment_of_pgm_qc = len(temp_qc.moments)

    def derive_event_prob(self, num_meas) -> float:
        assert 1 <= num_meas
        b = 1 - self.zero_measure_prob
        a = self.zero_measure_prob
        return (a ** (num_meas - 1)) * b

    def expected_measurement_num_baseline(self) -> float:
        """
        Returns expected number of required measurements untill the error term is measured
        assuming random variable X in {1,2,3,...}
        Args:
            zero_measure_prob (float): probability of measuring |0\rangle
        """
        if not self.is_true_zero_case:
            b = 1 - self.zero_measure_prob
            a = self.zero_measure_prob
            return (a / b) + 1  # floating error may hapen, when a~=1, so handle by self.is_true_zero_case
        else:
            return float("inf")

    # def sym_moment_cost_for_each_event(self, m, moment_cost : int) -> sympy.core.basic.Basic:
    #     return m * moment_cost

    def sym_moment_expected_execution_cost(self) -> sympy.core.basic.Basic:
        if not self.is_true_zero_case:
            return self.sym_pgm_moment_cost * (self.expected_measurement_num_baseline()) / 2
        else:
            return ((self.config.NUM_base_execution) / 2) * self.sym_pgm_moment_cost


class SymCostStatisticFPAA:

    def __init__(
        self,
        prob_measure_zero: float,
        sym_pgm_cnot_cost: sympy.core.basic.Basic,
        sym_pgm_all_gate_cost: sympy.core.basic.Basic,
        sym_pgm_moment_cost: sympy.core.basic.Basic,
        pgm_qc: cirq.Circuit,
        config: HypeParamConfig,
    ) -> None:
        assert len(config.l_SEQ) != 0 and len(config.l_SEQ) == len(config.NUM_measure_per_l)
        if sym_pgm_cnot_cost:
            assert sympy.symbols("n") in sym_pgm_cnot_cost.free_symbols

        self.sym_pgm_cnot_cost = sym_pgm_cnot_cost
        self.sym_pgm_all_gate_cost = sym_pgm_all_gate_cost
        self.sym_pgm_moment_cost = sym_pgm_moment_cost
        assert 0 <= prob_measure_zero and prob_measure_zero <= 1
        self.prob_measure_zero = prob_measure_zero  # probability of measureing zero from given program state $P$
        self.is_true_zero_case = math.isclose(a=prob_measure_zero, b=1, abs_tol=1e-06)
        self.l_sequence = config.l_SEQ
        self.num_meas_per_l = config.NUM_measure_per_l

        self.hyp_param_sequence = config.sequence
        self.config = config
        self.delta = config.DELTA
        self.amplified_prob_per_l = dict()  # build 'amplfied' probability of measuring error term (i.e, non-zeros)
        for l in self.l_sequence:
            self.amplified_prob_per_l[l] = prob_L(delta=config.DELTA, lamb=1 - self.prob_measure_zero, L=2 * l + 1)
        self.pgm_qc = pgm_qc
        self.init_prob_sym = sympy.symbols("b")

    def derive_symbolic_prob_per_ord(self, ord_idx: int) -> float:
        from qruntest.assurance_ftn import sym_P_L

        prob_builder = 1
        assert ord_idx <= len(self.config.sequence)
        for idx_of_prv_ords in range(ord_idx):
            curr_l = self.hyp_param_sequence[idx_of_prv_ords][0]
            curr_amplified_sym = sym_P_L(delta=self.config.DELTA, lamb=self.init_prob_sym, L=2 * curr_l + 1)
            prob_builder = prob_builder * (1 - curr_amplified_sym)
        l_of_given_idx = self.hyp_param_sequence[ord_idx][0]
        corr_amplified_sym = sym_P_L(delta=self.config.DELTA, lamb=self.init_prob_sym, L=2 * l_of_given_idx + 1)
        prob_builder = prob_builder * corr_amplified_sym
        return prob_builder

    def derive_event_prob_per_ord(self, ord_idx: int) -> float:  # return probability value
        prob_builder = 1
        for idx_of_prv_ords in range(ord_idx):
            curr_l = self.hyp_param_sequence[idx_of_prv_ords][0]
            curr_b_amplified = self.amplified_prob_per_l[curr_l]
            prob_builder = prob_builder * (1 - curr_b_amplified)
        l_of_given_idx = self.hyp_param_sequence[ord_idx][0]
        corr_b_amplified = self.amplified_prob_per_l[l_of_given_idx]
        prob_builder = prob_builder * corr_b_amplified
        assert 0 <= prob_builder and prob_builder <= 1
        return prob_builder

    def event_prob_sum(self) -> float:
        prob_sum = 0
        for idx, _ord in enumerate(self.hyp_param_sequence):
            l_val, m_val = _ord
            curr_event_prob = self.derive_event_prob_per_ord(ord_idx=idx)
            prob_sum += curr_event_prob
            # print(f"On idx {idx} ord ({l_val}, {m_val}) : {curr_event_prob}")
        # print(f"Check inf Prob sum is close to 1 : {prob_sum}")
        return prob_sum

    def print_event_prob(self) -> float:
        prob_sum = 0
        # assuming index starts 1 in here!!
        for idx, _ord in enumerate(self.hyp_param_sequence):
            l_val, m_val = _ord
            curr_event_prob = self.derive_event_prob_per_ord(ord_idx=idx)
            prob_sum += curr_event_prob
            print(f"On idx {idx+1} ord ({l_val}, {m_val}) : {curr_event_prob}")
        print(f"Check inf Prob sum is close to 1 : {prob_sum}")
        return None

    def expected_order_in_seq(self) -> float:
        # assuming index starts 1 in here!!
        expect_val_builder = 0
        for idx, _ord in enumerate(self.hyp_param_sequence):
            l_val, m_val = _ord
            curr_event_prob = self.derive_event_prob_per_ord(ord_idx=idx)
            expect_val_builder += (idx + 1) * curr_event_prob
        return expect_val_builder

    def closest_corr_expected_idx(self):
        expected_idx = self.expected_order_in_seq()
        idx = round(expected_idx) - 1
        return self.hyp_param_sequence[idx]

    def ord_to_num_grover(self, ord_idx: int) -> int:  # Order to Number of Grover
        num_grover_iterate = 0
        for i in range(ord_idx + 1):
            l, _ = self.hyp_param_sequence[i]
            num_grover_iterate += l
        return num_grover_iterate

    def expected_num_g_iterate(self) -> float:
        expected_val_builder = 0
        for idx, _ord in enumerate(self.hyp_param_sequence):
            l_val, m_val = _ord
            corr_num_iterate = self.ord_to_num_grover(ord_idx=idx)
            corr_event_prob = self.derive_event_prob_per_ord(ord_idx=idx)
            expected_val_builder += corr_num_iterate * corr_event_prob
        return expected_val_builder

    def sym_moment_cost_for_each_event(self, ord_idx: int) -> sympy.core.basic.Basic:
        from qruntest.symbolic_cost_model import sym_static_moment_cost_fpaa_by_ord

        # cost_formula = sym_static_moment_cost_fpaa_by_ord(ord = (l, num_meas), l_list=self.l_sequence, m_list=self.num_meas_per_l, hyp_param_seq=self.hyp_param_sequence)
        cost_formula = sym_static_moment_cost_fpaa_by_ord(ord_idx=ord_idx, hyp_param_seq=self.hyp_param_sequence)
        subsed_cost_formula = cost_formula.subs({sympy.symbols("x"): self.sym_pgm_moment_cost})
        return subsed_cost_formula

    def sym_moment_expected_execution_cost(self) -> sympy.core.basic.Basic:
        sym_expect_val_builder = 0  # floating error may hapen, when a~=1, so handle by self.is_true_zero_case
        if not self.is_true_zero_case:
            for idx, _ord in enumerate(self.hyp_param_sequence):
                prob = self.derive_event_prob_per_ord(ord_idx=idx)
                cost = self.sym_moment_cost_for_each_event(ord_idx=idx)
                sym_expect_val_builder += prob * cost
            return sym_expect_val_builder
        else:
            last_hyp_param_seq = self.hyp_param_sequence[-1]
            return self.sym_moment_cost_for_each_event(ord_idx=len(self.hyp_param_sequence) - 1)

    def sym_to_bn_moment_expected_execution_cost(self) -> sympy.core.basic.Basic:
        sym_expect_val_builder = 0  # floating error may hapen, when a~=1, so handle by self.is_true_zero_case
        if not self.is_true_zero_case:
            for idx, _ord in enumerate(self.hyp_param_sequence):
                prob = self.derive_symbolic_prob_per_ord(ord_idx=idx)
                cost = self.sym_moment_cost_for_each_event(ord_idx=idx)
                sym_expect_val_builder += prob * cost
            return sym_expect_val_builder
        else:
            last_hyp_param_seq = self.hyp_param_sequence[-1]
            return self.sym_moment_cost_for_each_event(ord_idx=len(self.hyp_param_sequence) - 1)
