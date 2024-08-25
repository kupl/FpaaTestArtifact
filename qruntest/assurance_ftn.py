import numpy as np
from cirq.linalg import predicates
from cirq import protocols
from typing import List, Union, Tuple
import math, sympy, cirq
from sympy import symbols
from sympy.core.symbol import Symbol
from sympy.functions.elementary.hyperbolic import  cosh, acosh
from sympy import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy import acos, cos
import scipy


# def prob_L(delta : float, lamb : float, L : int):
#    #input lambda value may be given in probability, NOT IN AMPLITUDE
#    T_1L = np.cosh((1 / L) * np.arccosh(1 / delta), dtype=np.complex128) # T_{1/L}(1/delta)
#    T_L = np.cosh((L) * np.arccosh(T_1L * math.sqrt( 1- lamb )   ) , dtype=np.complex128 )
#    return 1 - (delta**2) * (np.abs(T_L) ** 2)



def sym_P_L(delta : float, lamb : Symbol, L : int) -> sympy.core.basic.Basic:
    #input lambda value may be given in probability, NOT IN AMPLITUDE
    T_1L = cosh((1 / L) * acosh(1 / delta)) # T_{1/L}(1/delta)
    T_L = cos((L) * acos(T_1L * sqrt( 1- lamb )   )  )
    return 1 - (delta**2) * (Abs(T_L) ** 2)


def derive_credible_level(delta : float, L : int, k : int, w : float) -> float:
    # Pr(b in [w,1]| M)
    assert 0<= w and w<=1
    assert k >= 1
    assert 3*w < 1 # give small enough one for stable numerical integral..
    lamb = symbols("p")
    likelihood = ( 1 -sym_P_L(delta =delta, lamb=lamb, L = L)) ** k
    by_b = lambda x : likelihood.subs({lamb : x})
    numerator = scipy.integrate.quad(by_b, w, 2*w)[0] + scipy.integrate.quad(by_b, 2*w,3*w)[0] + scipy.integrate.quad(by_b, 3*w,1)[0]
    marignal_constnat = scipy.integrate.quad(by_b, 0,w)[0] + scipy.integrate.quad(by_b, w,2*w)[0] + scipy.integrate.quad(by_b, 2*w,3*w)[0] + scipy.integrate.quad(by_b, 3*w,1)[0]
    return numerator / marignal_constnat


def derive_required_num_sample_for_base(confidence :float, w: float):
    return (np.log(confidence)/np.log(1-w)) - 1