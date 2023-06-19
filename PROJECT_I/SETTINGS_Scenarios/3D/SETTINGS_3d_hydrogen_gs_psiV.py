use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, tol=1e-12):
    return -1/(cnp.linalg.norm(grid, axis=-1)+tol)

mus = cnp.array([0,0,0])
sigmas = cnp.array([1/cnp.sqrt(2)]*3)
ps = cnp.array([0,0,0])

# Initial Wavefunction
def psi0(grid, a0=1.0):
    return (1/(np.sqrt(np.pi*a0**3)))*cnp.exp(-cnp.linalg.norm(grid, axis=-1)/a0, dtype=cnp.csingle)




