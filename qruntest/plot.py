import sys
sys.path.append(".")
from qruntest.fpaa_in_matrix import get_omega
from hyper_parameters import DELTA
import numpy as np
import matplotlib.pyplot as plt
import math
from qruntest.utils import prob_L

epsilon = 0.06

def plot_omega_aux(l : int) :
    # L = 2l + 1
    return get_omega(L = 2*l +1 ,delta = DELTA) 

def plot_omega():
    l_domain = np.arange(1, 40)
    omega_image = np.array(list(map(plot_omega_aux, l_domain)))
    plt.plot(l_domain, [epsilon **2 for _ in l_domain] ,  '--', color='red', label='Dashed' )
    plt.plot(l_domain, omega_image)
    plt.title('plot omega by '+ r'$\delta = $' +  str(DELTA))
    plt.xlabel('l')
    plt.ylabel('omega')
    plt.grid(True)
    plt.show()    

def get_omega_list():
    l_domain = np.arange(1, 40)
    for l in l_domain:
        # print(f"==== l : {l} ====")
        omega = get_omega(L = 2*l +1 ,delta = DELTA) 
        # print("{:<13} : {:<5}".format("ω", omega))
        # print("{:<13} : {:<5}".format("ω_in_amplitude", math.sqrt(omega)))
        print(f"{l},{math.sqrt(omega)},{omega}")


def get_P_L_val_aux(init_sv_meas_non_zero : float, l :int) :
    prob_L_res = prob_L(delta=math.sqrt(0.05), lamb = init_sv_meas_non_zero, L = 2*l+1)
    return (1-prob_L_res) ** 3

def get_P_L_val():
    l = 18
    omega_on_l = get_omega(L = 2*l +1 ,delta = DELTA) 
    print(f"Omgea on l : {l} is {omega_on_l}")
    init_sv_meas_non_zero_domain = np.arange(0.0000,0.005,0.0001)
    lambda x : get_P_L_val_aux(init_sv_meas_non_zero=x, l=l) 
    val_image = np.array(list(map(lambda x : get_P_L_val_aux(init_sv_meas_non_zero=x, l=l) , init_sv_meas_non_zero_domain)))
    print(f"Value on 1w/8 : {get_P_L_val_aux(init_sv_meas_non_zero=1*(omega_on_l/8),l=l)}")
    print(f"Value on 3w/8 : {get_P_L_val_aux(init_sv_meas_non_zero=3*(omega_on_l/8),l=l)}")
    print(f"Value on 5w/8 : {get_P_L_val_aux(init_sv_meas_non_zero=5*(omega_on_l/8),l=l)}")
    print(f"Value on 7w/8 : {get_P_L_val_aux(init_sv_meas_non_zero=7*(omega_on_l/8),l=l)}")
    plt.plot(init_sv_meas_non_zero_domain, val_image)
    plt.axvline(x=omega_on_l, linestyle= '--', color='red')
    plt.xlabel('init_meas_val')
    plt.ylabel('P_L')
    plt.grid(True)
    plt.show() 

if __name__=='__main__':
    get_P_L_val()   
    # get_omega_list()