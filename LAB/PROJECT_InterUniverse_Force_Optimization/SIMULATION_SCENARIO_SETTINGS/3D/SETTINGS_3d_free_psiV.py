use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, dtype=cnp.single):
    return cnp.zeros(grid.shape[:-1], dtype=real_dtype)

mus = cnp.array([0]*3)
sigmas = cnp.array([1/cnp.sqrt(2)]*3)
ps = cnp.array([0,0,0.0])

# Initial Wavefunction
def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,Nz, 2]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)



