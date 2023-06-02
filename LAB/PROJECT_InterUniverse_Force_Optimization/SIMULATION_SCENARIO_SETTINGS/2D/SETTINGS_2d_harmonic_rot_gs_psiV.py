use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid):
    return 0.5*cnp.sum(grid**2, -1)

mus = cnp.array([0,0])
sigmas = cnp.array([1/np.sqrt(2),1/np.sqrt(2)])
ps = cnp.array([0,0])

# Initial Wavefunction
def psi0(grid, mus, sigmas, ps, hbar):
    # grid is [Nx,Ny, 2]
    phis = cnp.arctan2(grid[:,:,1], grid[:,:,0])
    rho = cnp.linalg.norm(grid-mus, axis=-1).astype(dtype=cnp.single )
    return rho*cnp.exp(1j*phis/hbar, dtype=cnp.csingle )*np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)




