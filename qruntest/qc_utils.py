import cirq
from qruntest.utils import keep
import numpy as np

class PgmGate(cirq.Gate):
   def __init__(self, qc : cirq.Circuit
                     , num_qbit : int
                     , in_unitary : np.array  = None 
                     , id : str = None):
      self.qc = qc
      self.in_unitary = in_unitary
      self.num_qbit = num_qbit
      super(PgmGate, self)

   def _num_qubits_(self):
      return self.num_qbit

   def _unitary_(self):
      if type(self.in_unitary) is np.ndarray :
         return self.in_unitary
      else :
         return cirq.unitary(self.qc)
   
   def _decompose_(self, work_qbits):
      assert len(work_qbits) == self.num_qbit
      for op in self.qc.all_operations() :
         qubits_idx = [ q.x for q in op.qubits]
         qbits_to_apply = [ work_qbits[x] for x in qubits_idx]
         yield op.gate(*qbits_to_apply)

   def __pow__(self, power):
      if power != -1 :
         raise NotImplementedError
      else :
         return type(self)(qc = (self.qc ** -1), num_qbit = self.num_qbit)

   def _circuit_diagram_info_(self, args):
      return ["PGM"] * self.num_qubits() ## TODO : improve printing
   
   def count_CNOT(self):  ## Validate Here
      cnt = 0
      # for x in self._decompose_(work_qbits=cirq.LineQubit.range(self.num_qbit)):
      for x in cirq.decompose(self.qc, preserve_structure=True):
         if x.gate in [cirq.CZ, cirq.CX, cirq.CNOT]:
            cnt+=1
         elif len(x.qubits) >= 2:
            if isinstance(x.gate , cirq.CZPowGate):
               cnt +=1
            else :
               raise ValueError(f"Invalid one? {x}")
         else :
            ValueError() 
      return cnt
   

class my_toffoli(cirq.Gate):
   def __init__(self):
      super(my_toffoli, self)

   def _num_qubits_(self):
      return 3
   
   
   def _decompose_(self, qubits):
      # from qcqi : Figure 4.9. Implementation of the Toffoli gate using Hadamard, phase, controlled-
      action_one, action_two, targ = qubits
      yield cirq.H(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield (cirq.T **-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.H(targ)
      yield (cirq.T**-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.T(action_one)
      yield cirq.S(action_two)

   def _circuit_diagram_info_(self, args):
      return "@","@","X"

class my_antianti_con_toff(cirq.Gate):
   def __init__(self):
      super(my_antianti_con_toff, self)

   def _num_qubits_(self):
      return 3
   
   def _unitary_(self) :
      to_return = cirq.ControlledGate(sub_gate = cirq.X,
                                                  num_controls = 2, 
                                                  control_values = [0,0])
      return cirq.unitary(to_return)
   
   def _decompose_(self, qubits):
      # from qcqi : Figure 4.9. Implementation of the Toffoli gate using Hadamard, phase, controlled-

      action_one, action_two, targ = qubits
      yield cirq.X(action_one)
      yield cirq.X(action_two)
      yield cirq.H(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield (cirq.T **-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.H(targ)
      yield (cirq.T**-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.T(action_one)
      yield cirq.S(action_two)
      yield cirq.X(action_one)
      yield cirq.X(action_two)

   def _circuit_diagram_info_(self, args):
      return "(0)","(0)","X"
   
class my_anticon_con_toff(cirq.Gate):
   def __init__(self):
      super(my_anticon_con_toff, self)

   def _num_qubits_(self):
      return 3
   
   def _unitary_(self):
      anticontrolled_controlled_toffoli = cirq.ControlledGate(sub_gate = cirq.X,
                                                  num_controls = 2, 
                                                  control_values = [0,1])
      return cirq.unitary(anticontrolled_controlled_toffoli)

   def _decompose_(self, qubits):
      action_one, action_two, targ = qubits
      yield cirq.X(action_one)
      yield cirq.H(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield cirq.CNOT(action_two, targ)
      yield (cirq.T **-1)(targ)
      yield cirq.CNOT(action_one, targ)
      yield cirq.T(targ)
      yield (cirq.T **-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.H(targ)
      yield (cirq.T**-1)(action_two)
      yield cirq.CNOT(action_one, action_two)
      yield cirq.T(action_one)
      yield cirq.S(action_two)
      yield cirq.X(action_one)

   def _circuit_diagram_info_(self, args):
      return "(0)","@","X"