use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, tol=1e-12):
    return -1/(cnp.linalg.norm(grid, axis=-1)+tol)

# Initial Wavefunction
def psi0(grid):
    return ((1/(4*np.sqrt(2*np.pi)))*grid[:,:,:,2]*cnp.exp(-cnp.linalg.norm(grid, axis=-1)/2, dtype=cnp.csingle)).astype(cnp.csingle)




