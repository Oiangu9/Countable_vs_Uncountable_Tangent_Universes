use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid,Vmax=20, Rshell=2, wshell=0.4,
            k_super=6, th1=17*np.pi/180, th2=40*np.pi/180, wslit=8*np.pi/180):
    Rs = np.sqrt(np.sum(grid**2, -1))
    Rs = np.where(Rs==0, 1e-4, Rs)
    thetas = cnp.arccos(grid[:,:,:,2]/Rs)
    return cnp.where(thetas<cnp.pi/2, 
            Vmax*cnp.exp(-cnp.abs((Rs-Rshell)/wshell)**k_super)*\
                (1-cnp.exp(-cnp.abs((thetas-th1)/wslit)**k_super)-\
                    cnp.exp(-cnp.abs((thetas-th2)/wslit)**k_super)), 0)


mus = cnp.array([0,0,4.5])
sigmas = cnp.array([1,1,0.7])
ps = cnp.array([0,0,-2.5])

# Initial Wavefunction
def psi0(grid, mus=mus, sigmas=sigmas, ps=ps, hbar=1):
    # grid is [Nx,Ny,Nz, 2]
    return np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)



