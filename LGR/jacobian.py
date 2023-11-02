#  functions for computing the flow-map jacobian as in Harms et al. (2023)

import numpy as np


# Generate a callable regression function for use in LGR and in
# planet-satellite approaches.
def setRegressionFunction(kernel: str = None, lam: float = None,
                          sig: float = None):

    # Specify regularizer
    if lam is None:
        lam = 0

    # Kernel functions to be used below
    def gen_phi(r, sigma):
        return np.exp(-(r**2)/(2*sigma**2))

    # Make a function using X0 and X1. This is what will be returned by the
    # parent function.
    def regFun(X0, X1):

        # naming variables
        dims, M = np.shape(X0)
        I_d = np.eye(dims)

        # Set the kernel for the regression.
        K = np.eye(M)
        if kernel == 'radialGaussian':
            if sig is None:
                # Compute kernel standard deviation from data
                sigma = np.std(np.linalg.norm(X0, axis=0))
                if sigma == 0:
                    sigma = 1
            else:
                sigma = sig
            # Radial Gaussian Weighting
            Phi = np.diag(gen_phi(np.linalg.norm(X0, axis=0), sigma)+0.00001)
            K = Phi.T * Phi

        # Return the function
        if lam is not None:
            return (X1 @ K @ X0.T) @ np.linalg.inv(X0 @ K @ X0.T + lam*M*I_d)
        else:
            if kernel is None:
                return X1 @ np.linalg.pinv(X0)
            else:
                return (X1 @ K @ X0.T) @ np.linalg.inv(X0 @ K @ X0.T)

    return regFun
