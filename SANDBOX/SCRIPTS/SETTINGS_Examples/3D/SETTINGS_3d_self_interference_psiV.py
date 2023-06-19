use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, real_dtype=cnp.single):
    return cnp.zeros(grid.shape[:-1], dtype=real_dtype)

mus1 = cnp.array([0,3,3])
sigmas1 = cnp.array([1,1,1])
ps2 = cnp.array([0,-10,-10])

mus2 = cnp.array([0,-3,-3])
sigmas2 = cnp.array([1,1,1])
ps1 = cnp.array([0,10,10])

# Initial Wavefunction
def psi0(grid, mus1=mus1, mus2=mus2, sigmas1=sigmas1, sigmas2=sigmas2, ps1=ps1, ps2=ps2, hbar=1, complex_dtype=cnp.csingle):
    return np.prod(1/(sigmas1*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus1)**2/(4*sigmas1**2))*cnp.exp(1j*(ps1*grid)/hbar), 
                   axis=-1, dtype=complex_dtype)+\
            np.prod(1/(sigmas2*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus2)**2/(4*sigmas2**2))*cnp.exp(1j*(ps2*grid)/hbar), 
                    axis=-1, dtype=complex_dtype)



