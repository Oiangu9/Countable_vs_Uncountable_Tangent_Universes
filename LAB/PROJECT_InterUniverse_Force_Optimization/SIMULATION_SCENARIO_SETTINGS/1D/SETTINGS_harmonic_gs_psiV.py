use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid):
    return 0.5*cnp.sum(grid**2, -1)

# Initial Wavefunction
mus = cnp.array([0])
sigmas = cnp.array([1/np.sqrt(2)])
ps = cnp.array([0.0])

def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,.., n]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)


