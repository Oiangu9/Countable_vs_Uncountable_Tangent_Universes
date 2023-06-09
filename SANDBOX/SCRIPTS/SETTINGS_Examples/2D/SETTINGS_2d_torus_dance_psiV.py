use_cuda_numpy = False
import numpy as np

if use_cuda_numpy:
    import cupy as cnp

else:
    import numpy as cnp

def chosenV(grid, R_torus=4):
    return 0.5*(cnp.sqrt(grid[:,:,0]**2+grid[:,:,1]**2)-R_torus)**2

# parameters for 2 gaussians
muss = cnp.array([[4, -0.5], [-4, 0.5]]) 
sigmass = cnp.array([[1.5,1.5],[1.5,1.5]])
#pss = cnp.array([[0,1.5],[0,1.5]])
pss = cnp.array([[0,-3],[0,3]])

def psi0(grid, muss=muss, sigmass = sigmass, pss = pss, hbar=1,complex_dtype=cnp.csingle):
    psi = cnp.zeros(grid.shape[:-1], dtype=cnp.csingle)
    for mus, sigmas, ps in zip(muss, sigmass, pss):
        psi+=np.prod(1/(sigmas*np.sqrt(2*np.pi))**0.5*cnp.exp(-(grid-mus)**2/(4*sigmas**2))*cnp.exp(1j*(ps*grid)/hbar), axis=-1, dtype=cnp.csingle)
    return psi/np.sqrt(len(muss))


