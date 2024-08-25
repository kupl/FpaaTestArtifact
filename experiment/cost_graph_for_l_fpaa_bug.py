import sys, math

sys.path.append(".")
from qruntest.fpaa_in_matrix import get_prob, get_omega
from qruntest.expectation import SymCostStatisticBase, SymCostStatisticFPAA
import argparse
from sympy.plotting import plot
from sympy import symbols, lambdify
from matplotlib import pyplot as plt
import numpy as np
from qruntest.hyper_parameters import HypeParamConfig, DELTA
from qruntest.symbolic_cost_model import sym_moment_cost_for_my_draper, sym_moment_cost_for_my_double_con_draper, sym_moment_cost_for_my_temp
from experiment.run_strat_search import main_search, derive_l_star, derive_m_star
from qruntest.assurance_ftn import derive_required_num_sample_for_base

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
from datetime import datetime
from prettytable import PrettyTable


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--tolerance", type=float, required=True)
parser.add_argument("--assurance", type=float, required=True)
parser.add_argument("--target_n", type=int, required=True)
parser.add_argument("--from_saved_data", type=bool, required=False)


args = parser.parse_args()

line_colors = ["blue", "green", "red", "magenta", "gray", "black"]
line_styles = [":", "--", "-", "-."]
if args.id == "draper":
    pgm_cost = sym_moment_cost_for_my_draper()
elif args.id == "con_draper":
    pgm_cost = sym_moment_cost_for_my_double_con_draper()
else:
    raise NotImplementedError


print(f"Assurance (eta) = {args.assurance *100}%")
### Below, in-system parmeter
num_b_chunk = 1000
n = symbols("n")
b = symbols("b")
x_lim = 1
b_vals = np.linspace(0.001, x_lim, num_b_chunk, endpoint=True)
near_zero = 0.1  # in percentange


### Building Strats ####
if args.from_saved_data:
    pass
else:
    l_star = derive_l_star(w=args.tolerance)
    m_star_for_wo_base = derive_m_star(l_star=l_star, confidence=args.assurance, w=args.tolerance, naive_part=0, delta=DELTA)
    print(f"Derived m_star_for_wo_base = {m_star_for_wo_base}")
    naive_without_base = [(l_star, m_star_for_wo_base)]
NUM_sample_for_baseline = math.ceil(derive_required_num_sample_for_base(confidence=args.assurance, w=args.tolerance))
print(f"Required num sample for baseline by given w and confi: {NUM_sample_for_baseline}")

naive_wo_base_config = HypeParamConfig(DELTA=DELTA, strat=naive_without_base, NUM_base_execution=NUM_sample_for_baseline, label=r"$l$-FPAA")


list_of_strats = list()
list_of_strats.append(naive_wo_base_config)


def run_bug_detect():

    print("Drawing Bug Detection Cost Analysis...")

    plt.xlabel("b=" + r"$|\langle E_{\bot}|P\rangle|^2$ " + " value")
    plt.ylabel("Reduction(%)")

    num_qbit = args.target_n
    print(f"The cost formula assumes n = {num_qbit}")
    avg_spd_up_by_strat = dict()
    max_vals_by_strat = dict()
    min_vals_by_strat = dict()

    for param_idx, strat in enumerate(list_of_strats):
        spd_up_for_curr_strat = list()
        for idx, b_val in enumerate(b_vals):

            base_cost = SymCostStatisticBase(
                prob_measure_zero=1 - b_val,
                sym_pgm_cnot_cost=None,
                sym_pgm_all_gate_cost=None,
                sym_pgm_moment_cost=pgm_cost + 1,  # for buggy case addd one moment, due to $Z$ misplacement
                pgm_qc=None,
                config=strat,
            )

            fpaa_cost = SymCostStatisticFPAA(
                prob_measure_zero=1 - b_val,
                sym_pgm_cnot_cost=None,
                sym_pgm_all_gate_cost=None,
                sym_pgm_moment_cost=pgm_cost + 1,  # for buggy case addd one moment, due to $Z$ misplacement
                pgm_qc=None,
                config=strat,
            )
            base_res = base_cost.sym_moment_expected_execution_cost().subs({n: num_qbit})
            fpaa_res = fpaa_cost.sym_moment_expected_execution_cost().subs({n: num_qbit})
            spd_up = (1 - (fpaa_res / base_res)) * 100  # data saved in percentage
            spd_up_for_curr_strat.append(spd_up)

        # find critical point index
        critical_point_idx = None
        for spd_up_idx, val in reversed(list(enumerate(spd_up_for_curr_strat))):
            if abs(val) > near_zero:
                critical_point_idx = spd_up_idx
                print(f"Critical Point :  {critical_point_idx} with Reduction {val}")
                break
        min_spd_up_val = min(spd_up_for_curr_strat)
        max_spd_up_val = max(spd_up_for_curr_strat)

        plt.plot(b_vals, spd_up_for_curr_strat, label=r"$l$-FPAA")
        avg_spd_up = sum(spd_up_for_curr_strat[: critical_point_idx + 1]) / len(spd_up_for_curr_strat[: critical_point_idx + 1])
        print(f"AVG Reduction for {strat.label} : {round(avg_spd_up    ,2)}%")
        print(f"Max Reduction for {strat.label} : {round(max_spd_up_val,2)}%")
        print(f"Min Reduction for {strat.label} : {round(min_spd_up_val,2)}%")

        avg_spd_up_by_strat[strat.label] = avg_spd_up
        min_vals_by_strat[strat.label] = min_spd_up_val
        max_vals_by_strat[strat.label] = max_spd_up_val

    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    b_vals_shrink = np.linspace(0.001, 0.0025, num_b_chunk)
    for param_idx, strat in enumerate(list_of_strats):
        spd_up_for_curr_strat = list()
        for idx, b_val in enumerate(b_vals_shrink):
            base_cost = SymCostStatisticBase(
                prob_measure_zero=1 - b_val,
                sym_pgm_cnot_cost=None,
                sym_pgm_all_gate_cost=None,
                sym_pgm_moment_cost=pgm_cost + 1,  # for buggy case addd one moment, due to $Z$ misplacement
                pgm_qc=None,
                config=strat,
            )

            fpaa_cost = SymCostStatisticFPAA(
                prob_measure_zero=1 - b_val,
                sym_pgm_cnot_cost=None,
                sym_pgm_all_gate_cost=None,
                sym_pgm_moment_cost=pgm_cost + 1,  # for buggy case addd one moment, due to $Z$ misplacement
                pgm_qc=None,
                config=strat,
            )
            base_res = base_cost.sym_moment_expected_execution_cost().subs({n: num_qbit})
            fpaa_res = fpaa_cost.sym_moment_expected_execution_cost().subs({n: num_qbit})
            spd_up = (1 - (fpaa_res / base_res)) * 100  # data saved in percentage
            spd_up_for_curr_strat.append(spd_up)
    a = plt.axes([0.65, 0.6, 0.2, 0.2])
    plt.plot(b_vals_shrink, spd_up_for_curr_strat)
    plt.grid(True)
    if args.id == "draper":
        plt.ylim(math.floor(float(spd_up_for_curr_strat[-1]) / 10) * 10, math.ceil(float(spd_up_for_curr_strat[0]) / 10) * 10)
    elif args.id == "con_draper":
        plt.ylim(math.floor(float(spd_up_for_curr_strat[-1]) / 10) * 10, math.ceil(float(spd_up_for_curr_strat[0]) / 10) * 10)

    return avg_spd_up_by_strat, min_vals_by_strat, max_vals_by_strat, list_of_strats


if __name__ == "__main__":
    print("Run by Strat")
    avg_spd_up_by_strat, min_vals_by_strat, max_vals_by_strat, list_of_strats = run_bug_detect()
    print(avg_spd_up_by_strat)
    print(min_vals_by_strat)
    print(max_vals_by_strat)
    print(list_of_strats)
    print("====Tried Strats====")
    for strat in list_of_strats:
        print(f"Strate {strat.label}")
        print(str(strat))
        print("--------------")
    print("====Table for Bug Detection====")
    t_bug = PrettyTable(["Reduction"] + [s._label for s in list_of_strats])
    t_bug.add_row(["Max"] + [f"{round(max_vals_by_strat[list_of_strats[i]._label], 2)}%" for i in range(len(list_of_strats))])
    t_bug.add_row(["Min"] + [f"{round(min_vals_by_strat[list_of_strats[i]._label], 2)}%" for i in range(len(list_of_strats))])
    print(t_bug)
    plt.figure(1)
    plt.savefig(f"pgm_{args.id}_tolerance_{args.tolerance}_confi_{args.assurance}_targetn_{args.target_n}_lfpaa_bug.pdf")
    plt.show()
