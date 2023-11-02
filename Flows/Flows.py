'''
Flows.py and the other files in this folder allow for particle trajectories
to be simulated from analytical functions.  Any new function should be
implemented in its own file similar to Gyre.py.  Flows may be 2 or 3
dimensional, or, if you are feeling ambitious, higher dimensional!
Custom functions can be defined in the main file and should be
initialized using the custom_function() class method.
'''

import numpy as np
from scipy.integrate import odeint
from typing import Callable

# Predefined functions
from Flows.Gyre import Gyre, Gyre_defaultParameters

predefined_function_map = {
    "Gyre": (Gyre, Gyre_defaultParameters),
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
                            initial_conditions: np.ndarray = None,
                            time_vector: np.ndarray = None,
                            parameters: dict[str, any] = None,
                            include_gradv: bool = False,
                            integrator_options: dict[str, any] = None):

        if function_name in predefined_function_map.keys():
            self.flowname = function_name
            self.function_generator = predefined_function_map[function_name][0]
            self.initial_conditions = initial_conditions
            self.time_vector = time_vector
            self.include_gradv = include_gradv
            self.integrator_options = integrator_options
            if parameters is not None:
                self.parameters = parameters
            else:
                self.parameters = predefined_function_map[function_name][1]

            # compute the flow function
            if self.include_gradv:
                fun, gradfun = self.function_generator(self.parameters,
                                                       self.include_gradv)
                self.flow_function = fun
                self.gradv_function = gradfun
            else:
                self.flow_function = self.function_generator(self.parameters)

        else:
            ValueError(f"{function_name} is not a predefined function.")

    def custom_function(self, function_name: str,
                        function_generator: Callable,
                        initial_conditions: np.ndarray = None,
                        time_vector: np.ndarray = None,
                        parameters: dict[str, any] = None,
                        include_gradv: bool = False,
                        integrator_options: dict[str, any] = None):
        self.flowname = function_name
        self.function_generator = function_generator
        self.initial_conditions = initial_conditions
        self.time_vector = time_vector
        self.parameters = parameters
        self.include_gradv = include_gradv
        self.integrator_options = integrator_options
        if self.include_gradv:
            try:
                fun, gradfun = self.function_generator(self.parameters,
                                                       self.include_gradv)
                self.flow_function = fun
                self.gradv_function = gradfun
            except Exception:
                ValueError("function isn't configured for gradients.")
        else:
            try:
                self.flow_function = self.function_generator(self.parameters)
            except Exception:
                ValueError("function_generator is not properly configured.")

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
                               self.time_vector,
                               ** self.integrator_options)
                return state

        for i, ic in enumerate(self.initial_conditions):
            self.states[i, :, :] = integrate(ic)
