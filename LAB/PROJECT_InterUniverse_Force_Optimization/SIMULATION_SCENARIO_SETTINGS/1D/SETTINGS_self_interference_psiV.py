use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, real_dtype=cnp.single):
    return cnp.zeros(grid.shape[:-1], dtype=real_dtype)

# parameters for 2 gaussians
muss = cnp.array([[-2.7], [2.7]]) 
sigmass = cnp.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])
pss = cnp.array([[1],[-1]])


def psi0(grid, muss=muss, sigmass=sigmass, pss=pss, hbar=1, complex_dtype=cnp.csingle):
    psi = cnp.zeros(grid.shape[:-1], dtype=complex_dtype)
    for mus, sigmas, ps in zip(muss, sigmass, pss):
        psi+=np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=complex_dtype)
    return psi/np.sqrt(len(muss))


