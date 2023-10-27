'''
Flows.py and the other files in this folder allow for particle trajectories
to be simulated from analytical functions.  Any new function should be implemented
in its own file similar to Gyre.py.  Flows may be 2 or 3 dimensional, or, if you are 
feeling ambitious, higher dimensional!  Custom functions can be defined in the 
main file and should be initialized using the custom_function() class method.
'''

import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Callable

sys.path.append(os.path.join(os.getcwd(), "Flows"))

# Predefined functions 
from Gyre import *

predefined_function_map = {
    "Gyre" : (Gyre, Gyre_defaultParameters),
}

class Flow:
    
    def __init__(self):
        # set the class attributes
        self.flowname: str = None
        self.function_generator: Callable = None
        self.include_gradv: bool = False
        self.flow_function: Callable = None
        self.gradv_function: Callable = None
        self.parameters: dict[str, any] = None
        self.initial_conditions: np.ndarray = None
        self.time_vector: np.ndarray = None
        self.integrator_options: dict[str, any] = None
        self.states: np.ndarray = None
        
    def predefined_function(self, function_name: str, 
                            initial_conditions: np.ndarray,
                            time_vector: np.ndarray,
                            parameters: dict[str, any]=None,
                            include_gradv: bool=False,
                            integrator_options: dict[str, any]=None):
        
        if function_name in predefined_function_map.keys():
            self.flowname = function_name
            self.function_generator = predefined_function_map[function_name][0]
            self.initial_conditions = initial_conditions
            self.time_vector = time_vector
            self.include_gradv = include_gradv
            self.integrator_options = integrator_options
            if not parameters is None:
                self.parameters = parameters
            else:
                self.parameters = predefined_function_map[function_name][1]
            
            # compute the flow function
            if self.include_gradv:
                self.flow_function, self.gradv_function = self.function_generator(self.parameters, self.include_gradv)
            else:
                self.flow_function = self.function_generator(self.parameters)
                
        else: 
            ValueError(f"{function_name} is not recognized as a predefined function.")
        
    def custom_function(self, function_name: str, 
                        function_generator: Callable,
                        initial_conditions: np.ndarray,
                        time_vector: np.ndarray,
                        parameters: dict[str, any],
                        include_gradv: bool=False,
                        integrator_options: dict[str, any]=None):
        pass # TODO
        
    def integrate_trajectories(self):
        # initialize the states array
        self.states = np.zeros([np.shape(self.initial_conditions)[0],
                                len(self.time_vector),
                                len(self.initial_conditions[0])])

        if self.integrator_options is None:
            def integrate(q):
                state = odeint(self.flow_function, 
                                q, 
                                self.time_vector)
                return state
        else:
            def integrate(q):
                state = odeint(self.flow_function, 
                                q, 
                                self.time_vector
                                **self.integrator_options)
                return state
            
        for i, ic in enumerate(self.initial_conditions):
            self.states[i,:,:] = integrate(ic)
        

        

        