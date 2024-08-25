import cirq
import numpy as np
import math

def R_k_gate(k : int):
    t = 2 / 2 ** k  
    return cirq.ZPowGate(exponent = t)

class my_C_R_k_gate(cirq.Gate):

    def __init__(self, k : int, put_minus = False):
        '''
            we will be generating c-phase gate that is
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & e^{i * self.deg}
        '''
        super(my_C_R_k_gate, self)
        assert isinstance(k, int)
        self.k = k
        self.deg = (2 / (2 ** k))  * math.pi
        self.to_feed_in_ZPOW = (2 / (2 ** k))
        if put_minus :
            self.deg = -self.deg
            self.to_feed_in_ZPOW = - self.to_feed_in_ZPOW
        
        self.put_minus = put_minus

    def _num_qubits_(self):
        return 2
    
    # def _unitary_(self):
    #     temp = cirq.ZPowGate(exponent = (2 / (2 ** self.k)) )
    #     res = cirq.ControlledGate(sub_gate =temp, num_controls = 1)
    #     return cirq.unitary(res)
    
    def _decompose_(self, qubits):
        con, targ = qubits
        if self.k == 0 :
            raise NotImplementedError
        if self.k == 1  or self.k == -1:
            yield cirq.H(targ)
            yield cirq.CNOT(con,targ)
            yield cirq.H(targ) 
        else:
            yield cirq.ZPowGate(exponent = self.to_feed_in_ZPOW   / 2 )(con)
            yield cirq.ZPowGate(exponent = self.to_feed_in_ZPOW   / 2 )(targ)
            yield cirq.CNOT(con,targ)
            yield cirq.ZPowGate(exponent = - self.to_feed_in_ZPOW / 2 )(targ)
            yield cirq.CNOT(con,targ)

            # yield cirq.ZPowGate(exponent = self.to_feed_in_ZPOW   / 2 )(con)
            # yield cirq.CNOT(con,targ)
            # yield cirq.ZPowGate(exponent = - self.to_feed_in_ZPOW / 2 )(targ)
            # yield cirq.CNOT(con,targ)
            # yield cirq.ZPowGate(exponent = self.to_feed_in_ZPOW   / 2 )(targ)

    def __pow__(self, exponent):
        if not (exponent == -1) :
            raise NotImplementedError
        return type(self)(k = self.k, put_minus=True)
    
    def _circuit_diagram_info_(self, args):
        if not self.put_minus :
            return "@", str(self.k)
        else :
            return "@", str("minus"+str(self.k))
        


class my_double_C_R_k_gate(cirq.Gate):
    def __init__(self, k : int, put_minus = False):
        assert isinstance(k, int)
        self.k = k
        self.deg = (2 / (2 ** k))  * math.pi
        self.to_feed_in_ZPOW = (2 / (2 ** k))
        if put_minus :
            self.deg = -self.deg
            self.to_feed_in_ZPOW = - self.to_feed_in_ZPOW
        
        self.put_minus = put_minus

    def _num_qubits_(self):
        return 3
    
    def _decompose_(self, qubits):
        # ref : barceno et al
        if self.k == 0 :
            raise NotImplementedError

        con_one, con_two, targ =qubits
        
        C_R_k_plus_1     = my_C_R_k_gate(k=self.k+1, put_minus=False)
        C_R_k_plus_1_dag  = my_C_R_k_gate(k=self.k+1, put_minus=True)

        yield C_R_k_plus_1(con_two, targ)
        yield cirq.CNOT(con_one, con_two)
        yield C_R_k_plus_1_dag(con_two, targ)
        yield cirq.CNOT(con_one, con_two)
        yield C_R_k_plus_1(con_one, targ)

    def _circuit_diagram_info_(self, args):
        if not self.put_minus :
            return "@", "@", str(self.k)
        else :
            return "@", "@", str("minus"+str(self.k))

class my_triple_C_R_k_gate(cirq.Gate):
    def __init__(self, k : int, put_minus = False):
        assert isinstance(k, int)
        self.k = k
        self.deg = (2 / (2 ** k))  * math.pi
        self.to_feed_in_ZPOW = (2 / (2 ** k))
        if put_minus :
            self.deg = -self.deg
            self.to_feed_in_ZPOW = - self.to_feed_in_ZPOW
        self.put_minus = put_minus

    def _num_qubits_(self):
        return 4
    
    def _decompose_(self, qubits):
        # ref : barceno et al
        if self.k == 0 :
            raise NotImplementedError
        con_one, con_two, con_three, targ = qubits

        C_R_k_plus_2     = my_C_R_k_gate(k=self.k+2, put_minus=False)
        C_R_k_plus_2_dag  = my_C_R_k_gate(k=self.k+2, put_minus=True)

        yield C_R_k_plus_2(con_one, targ)
        yield cirq.CNOT(con_one, con_two)
        yield C_R_k_plus_2_dag(con_two, targ)
        yield cirq.CNOT(con_one, con_two)
        yield C_R_k_plus_2(con_two, targ)
        yield cirq.CNOT(con_two, con_three)
        yield C_R_k_plus_2_dag(con_three,targ)
        yield cirq.CNOT(con_one,con_three)
        yield C_R_k_plus_2(con_three, targ)
        yield cirq.CNOT(con_two, con_three)
        yield C_R_k_plus_2_dag(con_three, targ)
        yield cirq.CNOT(con_one, con_three)
        yield C_R_k_plus_2(con_three, targ)

    def _circuit_diagram_info_(self, args):
        if not self.put_minus :
            return "@", "@", "@", str(self.k)
        else :
            return "@", "@", "@", str("minus"+ self.k)

class my_QFT(cirq.Gate):

    def __init__(self, n : int):
        # n : qubit length
        super(my_QFT, self)
        self.n = n 
    def _num_qubits_(self):
        return self.n

    def _unitary_(self):
        QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(self.n, without_reverse=True)
        return cirq.unitary(QFT_WO_REV)

    def _decompose_(self, qubits):
        qreg = list(qubits)
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for k, qubit in enumerate(qreg):
                # yield (cirq.CZ ** (1 / 2 ** (k + 1)))(qubit, q_head)
                yield (my_C_R_k_gate(k+2))(qubit, q_head) 
                
    def __pow__(self, exponent):
        if not (exponent == -1) :
            raise NotImplementedError
        return my_QFT_INV(n=self.n)
    
    def _circuit_diagram_info_(self, args):
      return ["qft"] *  self.num_qubits()


class my_QFT_INV(cirq.Gate):

    def __init__(self, n : int):
        # n : qubit length
        super(my_QFT_INV, self)
        self.n = n 
    def _num_qubits_(self):
        return self.n
    
    def _unitary_(self):
        REF_QFT_WO_REV = cirq.ops.QuantumFourierTransformGate(self.n, without_reverse=True)  
        return cirq.unitary(REF_QFT_WO_REV**-1)
    
    def _decompose_(self, qubits):
        qreg = list(qubits)
        to_collect = list()
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            to_collect.append( cirq.H(q_head))
            for k, qubit in enumerate(qreg):
                # yield (( cirq.CZ ** (1 / 2 ** (k + 1))  ** -1 ))(qubit, q_head)
                to_collect.append(((my_C_R_k_gate(k+2))**-1)(qubit, q_head))
        to_collect = reversed(to_collect)
        for x in list(to_collect):
            yield x

    def __pow__(self, exponent):
        if not (exponent == -1) :
            raise NotImplementedError
        return my_QFT(n=self.n)

    def _circuit_diagram_info_(self, args):
      return ["qft_inv"] *  self.num_qubits()

def R_k_gate(k : int):
    t = 2 / 2 ** k  
    return cirq.ZPowGate(exponent = t)


