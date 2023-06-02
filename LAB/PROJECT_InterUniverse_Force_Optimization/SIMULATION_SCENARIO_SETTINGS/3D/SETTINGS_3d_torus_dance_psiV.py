use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, R=4):
    return 0.5*(grid[:,:,:,2]**2+(cnp.sqrt(grid[:,:,:,0]**2+grid[:,:,:,1]**2)-R)**2).astype(real_dtype)

mus = cnp.array([4, 0, 0])
sigmas = cnp.array([1.3]*3)
ps = cnp.array([0,4,0])

# Initial Wavefunction
def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,Nz, 2]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)



