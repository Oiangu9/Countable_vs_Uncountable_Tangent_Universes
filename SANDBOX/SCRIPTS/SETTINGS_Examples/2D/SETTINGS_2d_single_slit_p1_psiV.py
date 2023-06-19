use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, Vmax=20, xbottnk=1, adiabty=2, 
                   wmax=10, wmin=1, k_super=6):
    '''wmin is half the slit width
    - wmax>wmin is expected
    - grid supposed to be [Nx,Ny,2]
    - default params for xlowers=[0,-L] xuppers=[2*xbottnk, L]
    - should place adiabaticity<xbottlneck
    '''
    wx = wmax-(wmax-wmin)*cnp.exp(-cnp.abs((grid[:,:,0]-xbottnk)/adiabty)**k_super)
    return Vmax*(1-cnp.exp(-cnp.abs(grid[:,:,1]/wx)**k_super)) #[Nx,Ny]

# parameters for gaussian
mus = cnp.array([-2.5,0])
sigmas = cnp.array([1/np.sqrt(2),1/np.sqrt(2)])
ps = cnp.array([1,0])


def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,.., n]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)


