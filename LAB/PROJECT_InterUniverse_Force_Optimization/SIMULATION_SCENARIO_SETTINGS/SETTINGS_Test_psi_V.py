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
def psi0(x,y, mus=[-2,0], sigmas=[1,2], ps=[5,0], hbar=1):
    return 1/(sigmas[0]*cnp.sqrt(2*cnp.pi))**0.5*cnp.exp(-(x-mus[0])**2/(4*sigmas[0]**2))*\
            1/(sigmas[1]*cnp.sqrt(2*cnp.pi))**0.5*cnp.exp(-(y-mus[1])**2/(4*sigmas[1]**2))*\
                cnp.exp(1j*(ps[0]*x+ps[1]*y)/hbar)

