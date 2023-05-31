use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, Vmax=20, k_super=6, 
                   widthxbar=0.3, xbarrier=1, slitSep=2.5,
                  widthSlit=0.7):
    '''
    widths are half widths
    widthSlit should be smaller than half slitSeparation
    '''
    Vy = 1-cnp.exp(-cnp.abs((grid[:,:,1]-slitSep/2)/widthSlit)**k_super)-\
            cnp.exp(-cnp.abs((grid[:,:,1]+slitSep/2)/widthSlit)**k_super)
    return Vmax*cnp.exp(-cnp.abs((grid[:,:,0]-xbarrier)/widthxbar)**k_super)*Vy

# Initial Wavefunction
mus = cnp.array([-2,0])
sigmas = cnp.array([1,2])
ps = cnp.array([5,0])

def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,.., n]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)


