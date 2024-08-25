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
# parser.add_argument('--eta',          type=float ,required=True)
parser.add_argument("--naive_part", type=int, required=True)
parser.add_argument("--tolerance", type=float, required=True)
parser.add_argument("--assurance", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--target_n", type=int, required=True)
parser.add_argument("--from_saved_data", type=bool, required=False)


args = parser.parse_args()

line_colors = ["blue", "green", "red", "magenta", "gray", "black"]
line_styles = [":", "--", "-", "-.", (0, (3, 1, 1, 1, 1, 1))]
if args.id == "draper":
    print("Running on CASESTUDY 1 : Draper Adder")
    pgm_cost = sym_moment_cost_for_my_draper()
elif args.id == "con_draper":
    print("Running on CASESTUDY 2 : Con Draper Adder")
    pgm_cost = sym_moment_cost_for_my_double_con_draper()
else:
    raise NotImplementedError


print(f"Assurance (eta) = {args.assurance *100}%")
### Below, in-system parmeter
num_b_chunk = 600
num_s_sample = 3
num_n_lim_for_ver = 500
# naive_part_for_strat = int((1/args.eta) * 4)
naive_part_for_strat = args.naive_part
n = symbols("n")
b = symbols("b")
n_vals = np.linspace(4, num_n_lim_for_ver, 5000, endpoint=True)
x_lim = 11 * 0.01
b_vals = np.linspace(0.001, x_lim, num_b_chunk, endpoint=True)
near_zero = 0.1  # in percentange


### Building Strats ####
if args.from_saved_data:
    l_star = 34
    m_star = 5
    print(f"Preset l_* = {l_star} , m_* = {m_star}")
    naive_with_base = [(0, args.naive_part), (l_star, m_star)]
    m_star_for_wo_base = 6
    naive_without_base = [(l_star, m_star_for_wo_base)]
    print(f"Preset m for w/o base l-fpaa = {m_star_for_wo_base}")
else:
    l_star = derive_l_star(w=args.tolerance)
    m_star = derive_m_star(l_star=l_star, confidence=args.assurance, w=args.tolerance, naive_part=args.naive_part, delta=DELTA)
    m_star_for_wo_base = derive_m_star(l_star=l_star, confidence=args.assurance, w=args.tolerance, naive_part=0, delta=DELTA)
    naive_without_base = [(l_star, m_star_for_wo_base)]
    print(f"Derived l_* = {l_star} , m_* = {m_star}")
    print(f"Derived m_star_for_wo_base = {m_star_for_wo_base}")
    naive_with_base = [(0, naive_part_for_strat), (l_star, m_star)]
    naive_without_base = [(l_star, m_star_for_wo_base)]

NUM_sample_for_baseline = math.ceil(derive_required_num_sample_for_base(confidence=args.assurance, w=args.tolerance))
print(f"Required num sample for baseline by given tolerance and assurance: {NUM_sample_for_baseline}")


naive_w_base_config = HypeParamConfig(DELTA=DELTA, strat=naive_with_base, NUM_base_execution=NUM_sample_for_baseline, label=r"$s_{naive}$-FPAA")
naive_wo_base_config = HypeParamConfig(DELTA=DELTA, strat=naive_without_base, NUM_base_execution=NUM_sample_for_baseline, label=r"$l$-FPAA")

list_of_strats = list()
list_of_strats.append(naive_w_base_config)

temp_latex_label_dict = {0: r"$s_{{1}}$-FPAA", 1: r"$s_{{2}}$-FPAA", 2: r"$s_{{3}}$-FPAA", 3: r"$s_{{4}}$-FPAA", 4: r"$s_{{5}}$-FPAA"}

if args.from_saved_data:
    # s1_star_config = HypeParamConfig(DELTA=math.sqrt(0.05), strat= [(0, 100), (9, 2), (11, 3), (26, 1), (32, 5)], NUM_base_execution=NUM_sample_for_baseline, label= temp_latex_label_dict[0])
    # s2_star_config = HypeParamConfig(DELTA=math.sqrt(0.05), strat= [(0, 100), (9, 1), (13, 1), (18, 2), (26, 1), (32, 5)], NUM_base_execution=NUM_sample_for_baseline, label= temp_latex_label_dict[1])
    # s3_star_config = HypeParamConfig(DELTA=math.sqrt(0.05), strat=  [(0, 100), (1, 1), (9, 2), (18, 4), (32, 5)], NUM_base_execution=NUM_sample_for_baseline, label= temp_latex_label_dict[2])
    # list_of_strats.append(s1_star_config)
    # list_of_strats.append(s2_star_config)
    # list_of_strats.append(s3_star_config)
    s1_star_config = HypeParamConfig(DELTA=math.sqrt(0.05), strat=[(0, 100), (9, 3), (21, 3), (32, 6)], NUM_base_execution=NUM_sample_for_baseline, label=temp_latex_label_dict[0])
    list_of_strats.append(s1_star_config)
else:
    for i in range(num_s_sample):
        start = datetime.now()
        print(f"Finding strat in seed {args.seed+i*100}")
        found_strat = main_search(naive_part=args.naive_part, w=args.tolerance, confidence=args.assurance, l_star=l_star, m_star=m_star_for_wo_base, seed=args.seed + i * 100)
        found_strat._NUM_base_execution = NUM_sample_for_baseline
        found_strat._label = temp_latex_label_dict[i]
        list_of_strats.append(found_strat)
        print(f"Strat Sampling Time was {datetime.now()-start}")

strat_for_bug_find = list_of_strats.copy()
list_of_strats.append(naive_wo_base_config)
strat_for_ver = list_of_strats


def run_bug_detect():

    print("Drawing Bug Detection Cost Analysis...")
    plt.figure(1)
    plt.xlabel("b=" + r"$|\langle E_{\bot}|P\rangle|^2$ " + " value")
    plt.ylabel("Reduction(%)")
    num_qbit = args.target_n
    print(f"The cost formula assumes n = {num_qbit}")
    print("Strats to be evaluted are")
    for strat in strat_for_bug_find:
        print(strat.label)
        print(strat.str_in_tuple())
    avg_spd_up_by_strat = dict()
    max_vals_by_strat = dict()
    min_vals_by_strat = dict()
    print("Now evaluating")
    for param_idx, strat in enumerate(strat_for_bug_find):
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
        # plt.axhline(y=min_spd_up_val, color="red")
        max_spd_up_val = max(spd_up_for_curr_strat)
        plt.plot(b_vals, spd_up_for_curr_strat, color=line_colors[param_idx], label=strat.label, linestyle=line_styles[param_idx])

        avg_spd_up = sum(spd_up_for_curr_strat[: critical_point_idx + 1]) / len(spd_up_for_curr_strat[: critical_point_idx + 1])
        print(f"AVG Reduction for {strat.label} : {round(avg_spd_up    ,2)}%")
        print(f"Max Reduction for {strat.label} : {round(max_spd_up_val,2)}%")
        print(f"Min Reduction for {strat.label} : {round(min_spd_up_val,2)}%")

        avg_spd_up_by_strat[strat.label] = avg_spd_up
        min_vals_by_strat[strat.label] = min_spd_up_val
        max_vals_by_strat[strat.label] = max_spd_up_val

    plt.xlim(0, x_lim)
    upper = int(max([math.ceil(x * 0.1) for x in max_vals_by_strat.values()]) * 10)
    if args.id == "draper":
        plt.ylim(-150, upper)
    elif args.id == "con_draper":
        plt.ylim(-40, upper)
    else:
        raise NotImplementedError
    # plt.yticks(list(plt.yticks()[0]) + [ round(float(x),3) for x in min_vals])

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return avg_spd_up_by_strat, min_vals_by_strat, max_vals_by_strat, list_of_strats


def run_ver():
    print("Drawing Verification Cost Analysis...")
    plt.figure(2)
    # plt.title("Strategies :" +
    #           strat_title_builder + "\n"+
    #           f"Target Pgm :  {args.id}" + "\n")
    plt.xlabel("num of qubit")
    plt.ylabel("Reduction(%)")
    val_of_naive = None
    conv_values = list()
    conv_values_by_strat = dict()
    for param_idx, strat in enumerate(strat_for_ver):
        ver_base_cost = SymCostStatisticBase(prob_measure_zero=1, sym_pgm_cnot_cost=None, sym_pgm_all_gate_cost=None, sym_pgm_moment_cost=pgm_cost, pgm_qc=None, config=strat)
        ver_fpaa_cost = SymCostStatisticFPAA(prob_measure_zero=1, sym_pgm_cnot_cost=None, sym_pgm_all_gate_cost=None, sym_pgm_moment_cost=pgm_cost, pgm_qc=None, config=strat)

        def spd_up_by_n():
            ver_base_cost_fmla = ver_base_cost.sym_moment_expected_execution_cost()
            ver_fpaa_cost_fmla = ver_fpaa_cost.sym_moment_expected_execution_cost()
            return (1 - (ver_fpaa_cost_fmla / ver_base_cost_fmla)) * 100

        ver_spd_up_lamb = lambdify(n, spd_up_by_n(), modules=["numpy"])
        ver_spd_up = ver_spd_up_lamb(n_vals)
        # if param_idx == 0 :
        #     val_of_naive = ver_spd_up[-1]
        print(f"The covereged val of spd_up for strat {strat.label} is {round(ver_spd_up[-1],2) }%")
        conv_values.append(ver_spd_up[-1])
        conv_values_by_strat[strat.label] = ver_spd_up[-1]
        plt.plot(n_vals, ver_spd_up, color=line_colors[param_idx], label=strat.label, linestyle=line_styles[param_idx])
        # if param_idx > 0 :
        #     print(f"For Curr {param_idx}th strat cost for ver gap is {ver_spd_up[-1] - val_of_naive}")
        # plt.axhline(y=ver_spd_up[-1], color="black")

    upper = int(max([math.ceil(x * 0.1) for x in conv_values_by_strat.values()]) * 10)
    if args.id == "draper":
        plt.ylim(50, upper)
    elif args.id == "con_draper":
        plt.ylim(60, upper)
    plt.xlim(0, num_n_lim_for_ver)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.yticks(list(plt.yticks()[0]) + [ round(float(x),3) for x in conv_values])

    return conv_values_by_strat


if __name__ == "__main__":
    print("Run by Strat")
    avg_spd_up_by_strat, min_vals_by_strat, max_vals_by_strat, list_of_strats = run_bug_detect()
    ver_cost_by_strat = run_ver()
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
    t_bug = PrettyTable(["Reduction"] + [s._label for s in strat_for_bug_find])
    t_bug.add_row(["Max"] + [f"{round(max_vals_by_strat[strat_for_bug_find[i]._label], 2)}%" for i in range(len(strat_for_bug_find))])
    t_bug.add_row(["Min"] + [f"{round(min_vals_by_strat[strat_for_bug_find[i]._label], 2)}%" for i in range(len(strat_for_bug_find))])
    print(t_bug)
    print("====Table for Verification====")
    t_ver = PrettyTable([s._label for s in strat_for_ver])
    t_ver.add_row([f"{round(ver_cost_by_strat[strat_for_ver[i]._label], 2)}%" for i in range(len(strat_for_ver))])
    print(t_ver)

    plt.figure(1)
    plt.savefig(f"pgm_{args.id}_seed_{args.seed}_targetn_{args.target_n}_tolerance_{args.tolerance}_assure_{args.assurance}_naive_{args.naive_part}_bug.pdf")
    plt.figure(2)
    plt.savefig(f"pgm_{args.id}_seed_{args.seed}_ver.pdf")
    print("Fig Saved.")

    # plt.show()
