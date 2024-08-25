import math
from typing import List, Tuple

class HypeParamConfig():
    
    def __init__(self, DELTA : float, 
                        strat  : List[Tuple[int,int]],
                        NUM_base_execution : int ,
                        label : str = None) -> None:
        self._DELTA = DELTA
        self._strat =strat
        l_SEQ = list()
        NUM_measure_per_l = list()
        for elt in strat :
            l_SEQ.append(elt[0])
            NUM_measure_per_l.append(elt[1])
        
        self._l_SEQ = l_SEQ
        self._NUM_measure_per_l = NUM_measure_per_l
        self._label = label

        self._NUM_base_execution = NUM_base_execution
        assert len(self._l_SEQ) ==  len(self._NUM_measure_per_l)
        
        seq_builder =list()
        for idx_of_l, l_val in enumerate(self.l_SEQ):
            for i in range(1, self.NUM_measure_per_l[idx_of_l]+1):
                seq_builder.append((l_val, i)) # For example, [(0,5)] => [(0,1),(0,2),(0,3),(0,4), (0,5)]
        self.sequence = seq_builder
        
    @property
    def DELTA(self):
        return self._DELTA
    
    @property
    def label(self):
        return self._label

    @property
    def strat(self):
        return self._strat

    @property
    def l_SEQ(self):
        return self._l_SEQ

    @property
    def NUM_measure_per_l(self):
        return self._NUM_measure_per_l
    
    @property
    def NUM_base_execution(self): # sequential 실행 기준 (not pararelle 실행)
        return self._NUM_base_execution
    
    def __str__(self) -> str:
        str_builder = ""
        str_builder += f"delta : {str(self.DELTA)}" + "\n"
        str_builder += f"strat : {str(self.strat)}"
        return str_builder
        
    def str_in_tuple(self) -> str:
        tuple_li_builder = list()
        for idx, l in enumerate(self.l_SEQ) :
            tuple_li_builder.append((l, self.NUM_measure_per_l[idx]))
        return str(tuple_li_builder)

DELTA = math.sqrt(0.05)