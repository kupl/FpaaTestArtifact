import cirq
import numpy as np
import math
from  qruntest.utils import *
import matplotlib.pyplot as plt

def reflect_source(source : cirq.QuantumState , alpha : float) -> np.array:
    return np.identity(n = source.shape[0]) - (1- np.exp(- 1j* alpha )) * np.outer(source, source)

def reflect_target(target : cirq.QuantumState, beta : float) -> np.array:
    return np.identity(n = target.shape[0]) - (1- np.exp(1j* beta )) * np.outer(target, target)

def get_prob(curr_state, target) -> float:
    return np.abs(np.vdot(target, curr_state)) ** 2

def get_prob_non_target(curr_state, target):
    return 1 - get_prob(curr_state, target)

def lower_bound_L(delta, source, target):
    return np.log(2/delta) / math.sqrt(get_prob(source, target))

def lower_bound_l(delta, source, target):
    lower_L = lower_bound_L(delta, source, target)
    return (lower_L-1)/2

def lower_bound_L_non_targ(delta, source, target):
    return np.log(2/delta) / math.sqrt(1-get_prob(source, target))

def lower_bound_l_non_targ(delta, source, target):
    lower_L_non_target = lower_bound_L_non_targ(delta, source, target)
    return (lower_L_non_target-1)/2

def get_omega(L : int, delta):
    # for all lambda s.t lamda >= omega, the amplified probability $P_L$ is guaranteed to meet our critierion
    # where lambda is probability (not the amplitude) to measure zero in source state
    # assert L % 2 == 1 
    gamma = 1 / (np.cosh((1 / L) * np.arccosh(1 / delta)))
    return (1 - gamma**2)

def get_required_l_from_omega(x : float, delta)-> int:
    to_return = (np.arccosh(1/delta) / np.arccosh(1/ math.sqrt(1-x)) - 1) / 2
    return to_return

def get_phase_anlges(l, delta):
    alphas, betas = dict(), dict()
    L = (2 * l) + 1
    gamma = 1 / np.cosh((1 / L) * np.arccosh(1 / delta))
    sg = np.sqrt(1 - gamma**2)
    for j in range(1, l+1):
        alphas[j]     = 2 * np.arctan2(1, (np.tan(2 * np.pi * j / L)   * sg)) ## why use  np.arctan2?
        betas[l-j+1]  = - alphas[j] 
    return alphas, betas

def get_phase_anlges_rev_aa(l, delta):
    alphas, betas = dict(), dict()
    L = (2 * l) + 1
    gamma = 1 / np.cosh((1 / L) * np.arccosh(1 / delta))
    sg = np.sqrt(1 - gamma**2)
    for j in range(1, l+1):
        alphas[j]     = 2 * np.arctan2(1, (np.tan(2 * np.pi * j / L)   * sg)) ## why use  np.arctan2?
        betas[l-j+1]  = alphas[j] 
    return alphas, betas

def get_grover_phase_angles(l : int ):
    alphas = []
    betas = []
    for _ in range(l):
        alphas.append(np.pi)
        betas.append(np.pi)


    return alphas, betas

def plot_fp_aa_experiment(source : np.array, target : np.array, delta : float, amp_non_targ : bool = False, num_l : int = 50):

    num_qbits_source = int(math.log(source.shape[0],2))
    num_qbits_target = int(math.log(target.shape[0],2))
    
    assert(num_qbits_source == num_qbits_target)

    x_axis = list()
    P_l_data = list()
    simulator = cirq.Simulator()
    flag = False
    title_builder = f"Source : {in_dirac_str(source)} Target : {in_dirac_str(target)}"
    collect_all_prob_l = list()
    collect_all_x_axis = list()
    print("Prob of measuring Target:", get_prob(source, target) )
    print("Prob of measuring Non-Target:", get_prob_non_target(source, target) )
    for l in range(1,num_l):
        aa_qc = cirq.Circuit()
        qbits = cirq.LineQubit.range(num_qbits_source)
        if amp_non_targ :
            alphas, betas = get_phase_anlges_rev_aa(l = l, delta = delta)
        else :
            alphas, betas = get_phase_anlges(l = l, delta = delta)
        # assert len(alphas.items()) == l
        # assert len(alphas.items()) == len(betas.items())
        for j in range(1, l+1):
            aa_qc.append(cirq.MatrixGate(reflect_target(target, betas[j]), unitary_check_atol = 1e-03 )(*qbits))
            aa_qc.append(cirq.MatrixGate(-reflect_source(source, alphas[j]), unitary_check_atol = 1e-03)(*qbits))
        aa_result = simulator.simulate(aa_qc, initial_state= source)
        print(in_dirac_str(aa_result.final_state_vector))
        prob_l = get_prob(aa_result.final_state_vector, target)
        if amp_non_targ: ## Amplifying non-target \bar{T} state
            flag = True
            x_axis.append(l)
            print("Prob of measuring Target:", prob_l )
            print("Prob of measuring Non-Target:", get_prob_non_target(aa_result.final_state_vector, target) )
            P_l_data.append(get_prob_non_target(aa_result.final_state_vector, target))
        else: ## Amplifying target T state
            if prob_l >= 0.4 or flag :
                flag = True
                x_axis.append(l)
                P_l_data.append(prob_l)
            collect_all_prob_l.append(prob_l)
            collect_all_x_axis.append(l)
            print("Prob of measuring Target:", prob_l)

    if amp_non_targ :
        plt.ylim(ymax = 1, ymin = 0.1)
        plt.ylabel(r'$|\langle \bar{t} \mid U \mid s \rangle|^2$')
        plt.xlabel(r'$l$')
        title_builder += "/ Ampliyfy Non Target" + r'$|\langle \bar{t} \mid U \mid s \rangle|^2$'
    else : 
        plt.ylim(ymax = 1, ymin = 0.1)
        title_builder += "/ Ampliyfy Target" + r'$|\langle t \mid U \mid s \rangle|^2$'
        plt.ylabel(r'$|\langle t \mid U \mid s \rangle|^2$')
        plt.xlabel(r'$l$')
    if x_axis:
        plt.plot(x_axis, [ 1 - delta ** 2 for _ in x_axis], '--', color='red', label='Dashed')
        plt.plot(x_axis, P_l_data, '-', color = 'blue' )
        plt.xticks(range(x_axis[0],x_axis[-1]))
    else :
        plt.plot(collect_all_x_axis, [ 1 - delta ** 2 for _ in collect_all_x_axis], '--', color='red', label='Dashed')
        plt.plot(collect_all_x_axis, collect_all_prob_l, '-', color = 'blue' )
        plt.xticks(range(collect_all_x_axis[0],collect_all_x_axis[-1]))
    plt.title(label=title_builder)
    if amp_non_targ:
        lb_l = lower_bound_l_non_targ(delta, source, target)
        print("BOUND FOR L :", lower_bound_L_non_targ(delta, source, target) )
    else:
        lb_l = lower_bound_l(delta, source, target)
        print("BOUND FOR L :", lower_bound_L(delta, source, target) )
    plt.axvline(x=round(lb_l), color='black', label='lowerbound_l')
    plt.show()

# def plot_fpaa_by_a(source_list, target, l = 10, delta = 0.5):
    # prob_l_data = list()
    # x_axis =list()
    # qbit =cirq.LineQubit.range(1) 
    # simulator = cirq.Simulator()
    # for source in source_list :
    #     aa_qc = cirq.Circuit()
    #     alphas, betas = get_phase_anlges(l = l, delta = delta)
    #     print("Source",source)
    #     print("Target",target)
    #     print("Val a", get_prob(source, target))
    #     for j in range(1, l+1):
    #         aa_qc.append(cirq.MatrixGate(reflect_target(target, betas[j]), unitary_check_atol = 1e-03 )(qbit[0]))
    #         aa_qc.append(cirq.MatrixGate(-reflect_source(source, alphas[j]), unitary_check_atol = 1e-03)(qbit[0]))
    #     aa_result = simulator.simulate(aa_qc, initial_state= source)
    #     prob_l = get_prob(aa_result.final_state_vector, target)
    #     print(prob_l)
    #     prob_l_data.append(prob_l)
    #     x_axis.append(math.sqrt(get_prob(source, target)))
    # plt.ylim(ymax = 1, ymin = 0.0)
    # plt.plot(x_axis , prob_l_data, '-', color = 'blue' )
    # plt.show()

def sv_gen(a : float):
    print(a)
    assert( 0 <= a and a <=1 )
    G_13 = cirq.ry(2 * np.arccos( math.sqrt(a) ))
    a,b,c = cirq.LineQubit.range(3)
    simulator = cirq.Simulator()
    qc = cirq.Circuit()
    qc.append(G_13(a))
    result = simulator.simulate(qc)
    return result.final_state_vector

if __name__ == '__main__':
    G_13 = cirq.ry(2 * np.arccos( math.sqrt(99) / math.sqrt(100)))
    a,b,c = cirq.LineQubit.range(3)
    simulator = cirq.Simulator()
    qc = cirq.Circuit()
    qc.append(G_13(a))
    qc.append(cirq.SWAP(b,c))
    result = simulator.simulate(qc)
    source = result.final_state_vector
    # source = np.array([1/math.sqrt(4),0,0,0,0,1/math.sqrt(4),1/math.sqrt(4),1/math.sqrt(4)])
    target = cirq.one_hot(index=0, shape = (8,), dtype = np.complex128) # target is |000>
    delta = math.sqrt(0.05)
    

    print("Source : " ,in_dirac_str(source))
    print("Target : " ,in_dirac_str(target))
    print("lower l", lower_bound_l(delta, source, target))
    plot_fp_aa_experiment(source=source, target=target, delta=delta, rev_ampli = True)

    # npts = 400
    # target = np.array([1, 0], dtype=np.complex128)
    # target = cirq.one_hot(index=0, shape = (2,), dtype = np.complex128)
    # sources =list()
    # for n in range(0,npts):
    #     source_temp = sv_gen(n * (0.999/npts)  + 0.0001)
    #     sources.append(source_temp)
    
    # for l in [1,2,3,4,5,6,7,8,9,10]:
    #     plot_fpaa_by_a(sources, target,l= l ,delta=(0.5))


        