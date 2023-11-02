
'''
Here the gyre flow common to the LCS literature is implemented.
'''

import numpy as np
import sympy as sym
from typing import Dict, Callable

Gyre_defaultParameters: Dict[str, float] = {
    "A": 0.1,
    "epsilon": 0.1,
    "omega": 2*np.pi/10
}

Gyre_domain = {
    "domain": [[0, 1], [0, 2]],
    "periodic": [0, 0]
}


def Gyre(parameters: Dict[str, float] = Gyre_defaultParameters,
         gradV: bool = False) -> Callable:

    # Read the parameters
    A = parameters["A"]
    epsilon = parameters["epsilon"]
    omega = parameters["omega"]

    # Symbolic Gyre Function
    xs = sym.symbols('x, y')
    x, y, t = sym.symbols('x, y, t')

    a = epsilon*sym.sin(omega*t)
    b = 1-2*epsilon*sym.sin(omega*t)
    f = a*xs[0]**2 + b*xs[0]

    u = sym.pi*A*sym.sin(sym.pi*f)*sym.cos(sym.pi*xs[1])
    v = -sym.pi*A*sym.cos(sym.pi*f)*sym.sin(sym.pi*xs[1])*(2*a*xs[0] + b)
    U = [u, v]

    if gradV:
        gradV = [[sym.diff(u, x), sym.diff(u, y)],
                 [sym.diff(v, x), sym.diff(v, y)]]

    # Lambda Functions
    U_fun = sym.lambdify([xs, t], U)
    gradV_fun = sym.lambdify([xs, t], gradV)

    def flowFun(q, t):
        return U_fun(q, t)

    def gradVFun(q, t):
        return np.array(gradV_fun(q, t))

    if not gradV:
        return flowFun
    else:
        return flowFun, gradVFun
