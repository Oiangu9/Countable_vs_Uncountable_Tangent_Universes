use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, Vmax1=1.5, Vmax2=1.5, xbarrier1=2, xbarrier2=6,
                    width1=0.7, width2=0.7, k_super=6):
    return Vmax1*cnp.exp(-cnp.abs((grid[:,0]-xbarrier1)/width1)**k_super)+\
                Vmax2*cnp.exp(-cnp.abs((grid[:,0]-xbarrier2)/width2)**k_super)

# Initial Wavefunction
mus = cnp.array([-2])
sigmas = cnp.array([1])
ps = cnp.array([1.0])

def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,.., n]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)


