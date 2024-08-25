import numpy as np
import cirq
from cirq.linalg import predicates
from cirq import protocols
from typing import List, Union, Tuple
import math

class QuantumRuntimeError(Exception):
    def __init__(self, 
                 pgm_qc : cirq.Circuit,
                 measure_data : cirq.Result):
        super().__init__('Runtime Assertion Error.' +
                         '\n Qunatum circuit was : \n'
                         + str(pgm_qc)
                         + '\n Measurement data was : \n'
                         + str(measure_data))

def dirac_notation_in_prob(sv : np.array):
   return cirq.dirac_notation(cirq.state_vector_to_probabilities(sv), decimals = 10)

def sample_measure_untill_nonzero(qc : cirq.Circuit):
   simulator = cirq.Simulator()
   for idx in range(1,10000):
      result = simulator.run(qc, repetitions=1)
      if not is_all_zero(result=result, work_qbits= qc.all_qubits()) :
         return idx
   raise ValueError("why no non-zero measurement?")

def give_in_prob(result : Union[cirq.StateVectorTrialResult, np.ndarray ] )-> bool :
   if isinstance(result, np.ndarray):
      res_vec = result
   else :
      res_vec = result.final_state_vector
   conj_res_vec = np.conjugate(res_vec)
   prob_vec = np.multiply(res_vec, conj_res_vec)
   return prob_vec

def is_all_zero(result : cirq.Result, main_qbits : List[cirq.LineQubit])-> bool :
    qbits_idx_str = list(result.measurements.keys())[-1]
    assert(result.measurements[qbits_idx_str].shape == (1,len(main_qbits))) 
    measurement_data = result.measurements[qbits_idx_str][0]
    return not np.any(measurement_data) # True if all zero

def in_dirac_str(sv : np.ndarray):
   return cirq.dirac_notation(sv, decimals=10)

def prob_L(delta : float, lamb : float, L : int):
   # ref : https://github.com/ichuang/pyqsp/blob/6ec2499c153b4359a11c914ff59bff16545508cc/pyqsp/phases.py#L60
   #input lambda value may be given in probability, NOT IN AMPLITUDE
   # T_1L = np.cosh((1 / L) * np.arccosh(1 / delta), dtype=np.complex128) # T_{1/L}(1/delta)
   # T_L = np.cosh((L) * np.arccosh(T_1L * math.sqrt( 1- lamb )   ) , dtype=np.complex128 )
   # return 1 - (delta**2) * (np.abs(T_L) ** 2)
   T_1L = np.cosh((1 / L) * np.arccosh(1 / delta), dtype=np.complex128) # T_{1/L}(1/delta)
   T_L = np.cos((L) * np.arccos(T_1L * math.sqrt( 1- lamb )   ) , dtype=np.complex128 )
   return 1 - (delta**2) * (np.abs(T_L) ** 2)



def ket_zeros(qbits):
   prod_state_builder = None
   for idx, qbit in enumerate(qbits):
      if idx != 0:
         prod_state_builder = prod_state_builder * cirq.KET_ZERO(qbit)
      else:
         prod_state_builder = cirq.KET_ZERO(qbit)
   return prod_state_builder

def classic_state_to_product(sv, qbits):
   # asssert sv is classic state
   sv_in_dirac_str = in_dirac_str(sv)
   index_mid = sv_in_dirac_str.index('|')
   index_langle = sv_in_dirac_str.index('⟩') 
   binary_str = sv_in_dirac_str[index_mid+1:index_langle]
   prod_state_builder = None
   for bin, qbit in zip(binary_str, qbits):
      if bin == "0" :
         if prod_state_builder:
            prod_state_builder = prod_state_builder * cirq.KET_ZERO(qbit)
         else:
            prod_state_builder = cirq.KET_ZERO(qbit)
      elif bin == "1":
         if prod_state_builder:
            prod_state_builder = prod_state_builder * cirq.KET_ONE(qbit)
         else:
            prod_state_builder = cirq.KET_ONE(qbit)
   return prod_state_builder


RaiseValueErrorIfNotProvided: np.ndarray = np.array([])

class EntangledStateError(ValueError):
    """Raised when a product state is expected, but an entangled state is provided."""

def my_sub_state_vector(
    state_vector: np.ndarray,
    keep_indices: List[int],
    *,
    default: np.ndarray = RaiseValueErrorIfNotProvided,
    atol: Union[int, float] = 1e-6,
) -> np.ndarray:
   
   # ref : https://github.com/quantumlib/Cirq/blob/v1.2.0/cirq-core/cirq/linalg/transformations.py#L479-L578


    if not np.log2(state_vector.size).is_integer():
        raise ValueError(
            "Input state_vector of size {} does not represent a "
            "state over qubits.".format(state_vector.size)
        )

    n_qubits = int(np.log2(state_vector.size))
    keep_dims = 1 << len(keep_indices)
    ret_shape: Union[Tuple[int], Tuple[int, ...]]
    if state_vector.shape == (state_vector.size,):
        ret_shape = (keep_dims,)
        state_vector = state_vector.reshape((2,) * n_qubits)
    elif state_vector.shape == (2,) * n_qubits:
        ret_shape = tuple(2 for _ in range(len(keep_indices)))
    else:
        raise ValueError("Input state_vector must be shaped like (2 ** n,) or (2,) * n")

    keep_dims = 1 << len(keep_indices)
    if not np.isclose(np.linalg.norm(state_vector), 1, atol=atol):
        raise ValueError("Input state must be normalized.")
    if len(set(keep_indices)) != len(keep_indices):
        raise ValueError(f"keep_indices were {keep_indices} but must be unique.")
    if any([ind >= n_qubits for ind in keep_indices]):
        raise ValueError("keep_indices {} are an invalid subset of the input state vector.")

    other_qubits = sorted(set(range(n_qubits)) - set(keep_indices))
    candidates = [
        state_vector[predicates.slice_for_qubits_equal_to(other_qubits, k)].reshape(keep_dims)
        for k in range(1 << len(other_qubits))
    ]
    # The coherence measure is computed using unnormalized candidates.
    best_candidate = max(candidates, key=lambda c: float(np.linalg.norm(c, 2)))
    best_candidate = best_candidate / np.linalg.norm(best_candidate)
    left = np.conj(best_candidate.reshape((keep_dims,))).T
    coherence_measure = sum([abs(np.dot(left, c.reshape((keep_dims,)))) ** 2 for c in candidates])

    if protocols.approx_eq(coherence_measure, 1, atol=atol):
        return np.exp(2j * np.pi * np.random.random()) * best_candidate.reshape(ret_shape)

    # Method did not yield a pure state. Fall back to `default` argument.
    if default is not RaiseValueErrorIfNotProvided:
        return default

    raise EntangledStateError(
        "Input state vector could not be factored into pure state over "
        "indices {}".format(keep_indices)
    )


def keep(op : cirq.GateOperation)-> bool:
   if op.gate == cirq.CNOT or op.gate.num_qubits()==1:
         return True

def circuit_merge_policy_for_count_metric(qc : cirq.Circuit) -> cirq.Circuit:
   return qc


def required_number_of_base_execute(nonzero_prob : float):
   assert 0 <= nonzero_prob and nonzero_prob <= 1
   return (1.9) ** 2 / nonzero_prob