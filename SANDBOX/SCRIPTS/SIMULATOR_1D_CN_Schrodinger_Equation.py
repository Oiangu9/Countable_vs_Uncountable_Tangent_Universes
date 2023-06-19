use_cuda_numpy = False

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import importlib
import sys

if use_cuda_numpy:
    import cupy as cnp
    import cupyx.scipy as csp
    import cupyx.scipy.sparse.linalg as clg

else:
    import numpy as cnp
    import scipy as csp
    import scipy.sparse.linalg as clg


# ROUTINES ###########################################
#Discretized Hamiltonian as a Sparse Matrix
def cuda_get_discrete_U_L(Nx, dx, dt, mx, hbar):
    # it is faster to prepare the diagonals with cpu and then transfer them to gpu
    # since the implementation is very sequential as it is
    # all arguments expected to be in cpu
    # but returned sparse matrix is in gpu
    main_diagonal = 1+1j*dt*0.5*hbar*(1/dx**2/mx)*cnp.ones(Nx, dtype=complex_dtype)
    for i in range(Nx):
        main_diagonal[i] += 1j*dt*V_i[i]/2/hbar
    x_diagonals = -1j*dt*hbar/(4*mx*dx**2)*cnp.ones(Nx-1, dtype=complex_dtype)
    return csp.sparse.diags( diagonals=
        (main_diagonal, x_diagonals, x_diagonals,),
        offsets=np.array([0, 1, -1]), dtype=complex_dtype,format='csc')

if __name__ == "__main__":
    # SIMULATION PARAMETERS ##################################
    args = sys.argv # The path to the settings file should be given
    #print("Input arguments", sys.argv)
    assert len(args)==5, "Paths or experiment name not given correctly!"
    ID_string = str(args[1])
    path_to_settings = str(args[2])
    path_to_psi_and_potential=str(args[3])
    outputs_directory = str(args[4])
    ''' Expected arguments:
    - ID for the output directroy naming
    - path to the settings file for the simulation
    - name of the .py file defining the psi0(x,y) and potential(x,y) functions (do not include the .py extension in the string)
    - path to outputs directory
    '''

    with open(f"{path_to_settings}", "r") as f:
        #K = int(f.readline().split("K ")[1])
        #A = int(f.readline().split("A ")[1])
        numIts = int(f.readline().split("numIts_SE ")[1])
        dt = float(f.readline().split("dt_SE ")[1])
        outputEvery = int(f.readline().split("outputEvery_SE ")[1])
        Ns = [int(x) for x in f.readline().split("Ns_SE_and_Newt_pdf ")[1].split('[')[1].split(']')[0].split(',')]
        xlowers = [float(x) for x in f.readline().split("xlowers ")[1].split('[')[1].split(']')[0].split(',')]
        xuppers = [float(x) for x in f.readline().split("xuppers ")[1].split('[')[1].split(']')[0].split(',')]
        numDofUniv = int(f.readline().split("numDofUniv ")[1])
        numDofPartic = int(f.readline().split("numDofPartic ")[1])
        numTrajs = int(f.readline().split("numTrajs ")[1])
        hbar = float(f.readline().split("hbar ")[1])
        ms = [float(x) for x in f.readline().split("ms ")[1].split('[')[1].split(']')[0].split(',')]
        complex_dtype = eval(f.readline().split("complex_dtype ")[1])
        real_dtype =  eval(f.readline().split("real_dtype ")[1])
        try:
            K_coulomb = float(f.readline().split("K_coulomb ")[1])
            qs = [float(x) for x in f.readline().split("qs ")[1].split('[')[1].split(']')[0].split(',')]
        except:
            K_coulomb = 1
            qs=[1]*numDofPartic
    Nx = Ns[0]
    # Time grid
    ts = np.array([dt*j for j in range(numIts)])

    # Increments to be used per dimension
    dxs = [(xuppers[j]-xlowers[j])/(Ns[j]-1) for j in range(1)] # (dx, dy)
    dx = dxs[0]

    #Create coordinates at which the solution will be calculated
    nodes = [cnp.linspace(xlowers[j], xuppers[j], Ns[j]) for j in range(1)] # (xs, ys)
    xs = nodes[0]

    # SIMULATION SCENRAIO AND INITIAL STATE #################################
    try:
        sys.path.append(''.join(path_to_psi_and_potential.split('/SETTINGS_')[:-1]))
        module=importlib.import_module('SETTINGS_'+path_to_psi_and_potential.split('/SETTINGS_')[-1])
        psi0 = getattr(module, 'psi0')
        chosenV = getattr(module, 'chosenV')

    except:
        raise AssertionError ("psi0 and chosenV functions not correctly defined in the given file!")
    # PREPARE ARRAYS FOR SIMULATION #########################################
    grid=cnp.array(cnp.meshgrid(*nodes)).T #[Nx, 1]
    V_i = chosenV(grid)
    psi = cnp.zeros(Nx, dtype=complex_dtype)
    psi = psi0(grid).flatten('F')
    # propagator ######################
    U_L = cuda_get_discrete_U_L(*Ns, *dxs, dt, *ms, hbar)
    U_R = U_L.conj()
    UL_LUdec = clg.splu( U_L )

    # initialize Bohmian trajectories
    # first get the pdf
    #psi0 = cnp.asnumpy(psi)
    pdf0 = (psi.conj()*psi).real
    pdf0 = pdf0/pdf0.sum() # normalize strictly

    # sample randomly
    i_s = cnp.random.choice( pdf0.shape[0],
                replace=True, size=(numTrajs), p=pdf0 ) #[numTrajs] indices

    # avoiding edges - probability density is zero there
    i_s[i_s==0] = 1
    i_s[i_s==Ns[0]-1] = Ns[0]-2

    trajs = cnp.zeros((numTrajs, 2)) #[numTrajs, 4 -2posit2momt]
    for tr, i in enumerate(i_s):
        trajs[tr, 0] = nodes[0][i]

    cdxs = cnp.array(dxs)[cnp.newaxis,:]
    cNs = cnp.array(Ns)[cnp.newaxis, :]
    cxlowers = cnp.array(xlowers)[cnp.newaxis, :]
    cxuppers = cnp.array(xuppers)[cnp.newaxis, :]
    cms = cnp.array(ms)[cnp.newaxis, :]

    # SANITY ##############################
    # Chosen Potential and Trajectories Plot for intial time
    if use_cuda_numpy:
        V_ij=cnp.asnumpy(V_ij)
        trajs_numpy = cnp.asnumpy(trajs)
    else:
        trajs_numpy = trajs
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    if use_cuda_numpy:
        V_i=cnp.asnumpy(V_i)
        trajs_numpy = cnp.asnumpy(trajs)
    else:
        trajs_numpy = trajs

    ax.plot(xs, V_i, label="Potential Energy", color=u'#ff7f0e')
    ax.plot(trajs_numpy[:,0], [np.mean(V_i)]*len(trajs_numpy),'o',
            c="black", markersize=1, label="Initial Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Potential Energy Field\nand Initial Trajectories")
    #ax.scatter(trajs[:,0], trajs[:,1], c="black", s=2,alpha=1)
    os.makedirs(f"./{outputs_directory}/SE_1D/{ID_string}/figs/", exist_ok=True)
    image=f"./{outputs_directory}/SE_1D/{ID_string}/figs/energy_potential.png"
    plt.legend()
    plt.savefig(image, dpi=120, bbox_inches='tight')

    # Create Output Directory ######################3
    os.makedirs(f"{outputs_directory}/SE_1D/{ID_string}/pdf/", exist_ok=True)
    os.makedirs(f"{outputs_directory}/SE_1D/{ID_string}/trajs/", exist_ok=True)
    os.makedirs(f"{outputs_directory}/SE_1D/{ID_string}/moms/", exist_ok=True)

    # Time Iteration LOOP #####################################3
    ############################################################
    ############################################################

    dpsi_dx = cnp.zeros(Ns, dtype=complex_dtype)
    p = cnp.zeros(dpsi_dx.shape+(1, ), dtype=real_dtype) #[ Nx, 1]

    for it, t in enumerate(ts):
        # BOHMIAN MOMENTUM FIELD #####################################
        # first the gradient of the wavefunction at each point
        # X
        # boundaries with simple Euler formula O(dx)
        dpsi_dx[0] = (psi[1]-psi[0])/dx
        dpsi_dx[-1] = (psi[-1]-psi[-2])/dx
        # next boundaries with centered difference O(dx**2)
        dpsi_dx[1] = (psi[2]-psi[0])/(2*dx)
        dpsi_dx[-2] = (psi[-1]-psi[-3])/(2*dx)
        # rest with O(dx**4) centered difference
        dpsi_dx[2:-2] = (-psi[4:]+8*psi[3:-1]-8*psi[1:-3]+psi[:-4])/(12*dx)

        # px, py, pz
        p[:,0] = hbar*(dpsi_dx/psi).imag

        # MOMENTUM ON TRAJS ##################################################
        # if no trajectory surpasses below the node 0 or J-1 at any time,
        # traj will always be among (0,J-1) and traj//dxs will be among [0,J-1]
        trajs_idxs = (((trajs[:,:numDofUniv]-cxlowers)//cdxs).T).astype(cnp.uint) # [1, numTrajs] the closest index from below along each axis
        # relative distance to the closest index from below point along each dimension
        # the closer, the bigger its weight should be for the trajectory propagation
        ratx_down = ((trajs[:,0]-xs[ trajs_idxs[0] ])/(xs[ trajs_idxs[0]+1 ]-xs[ trajs_idxs[0] ]))[:,cnp.newaxis]
        # Interpolate momentum
        trajs[:,numDofUniv:] = ratx_down* p[ trajs_idxs[0]+1 ] +\
            (1-ratx_down)* p[ trajs_idxs[0], ]

        # Before moving the trajectories, we save the state (with the positions and velocities of this time)
        # OUTPUT #####################################################
        if it%outputEvery == 0:
            # compute the magnitude squared of the wavefunction
            pdf = (psi.conj()*psi).real
            # Approximate the norm
            #print(f"   Iteration {it} Approx.Norm = {pdf.sum()*dx:.4}")
            if use_cuda_numpy:
                pdf = cnp.asnumpy(pdf)
                trajs_numpy = cnp.asnumpy(trajs)
                p_numpy = cnp.asnumpy(p)
            else:
                trajs_numpy = trajs
                p_numpy = p
            np.save(f"{outputs_directory}/SE_1D/{ID_string}/pdf/pdf_it_{it}.npy",
                    pdf, allow_pickle=True) #[Nx]
            np.save(f"{outputs_directory}/SE_1D/{ID_string}/trajs/trajs_it_{it}.npy",
                    trajs_numpy, allow_pickle=True) #[numTrajs, 2]
            np.save(f"{outputs_directory}/SE_1D/{ID_string}/moms/momentum_field_it_{it}.npy",
                    p_numpy, allow_pickle=True) #[Nx, 1]
        # NEXT TIME ITERATION POSITIONS ##############################################
        # Evolve trajectories using the interpolated momentum field
        trajs[:,:numDofUniv] = trajs[:,:numDofUniv] + dt*trajs[:,numDofUniv:]/cms #[numTrajs, 4]

        # Those trajectories that get out of bounds should bounce back by the amount they got out
        patience = 2 # max three bounces allowed
        # Those trajectories that get out of bounds should bounce back by the amount they got out
        while(cnp.any(trajs[:,:numDofUniv]>=cxuppers) or cnp.any(trajs[:,:numDofUniv]<cxlowers)):
            trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]>=cxuppers, cxuppers-(trajs[:,:numDofUniv]-cxuppers)-1e-10 ,trajs[:,:numDofUniv] )
            trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]<cxlowers, cxlowers+(cxlowers-trajs[:,:numDofUniv]) ,trajs[:,:numDofUniv] )
            patience-=1
            if patience==0:
                trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]>=cxuppers, cxuppers-1e-10,trajs[:,:numDofUniv] )
                trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]< cxlowers, cxlowers,trajs[:,:numDofUniv] )
                break
        # NEXT PSI ####################################################
        # compute the next time iteration's wavefunction
        U_Rpsi_prev = U_R@psi # this is the vector b in Ax=b
        psi = UL_LUdec.solve(U_Rpsi_prev)


    # Free Space ######
    if use_cuda_numpy:
        cnp.get_default_memory_pool().free_all_blocks()
        cnp.get_default_pinned_memory_pool().free_all_blocks()
        nodes = [cnp.asnumpy(nodel) for nodel in nodes]

    # Generate max 5 frames as a sanity check, equispaced ##################
    number_of_outputs = int(numIts//outputEvery)
    n_frames = min(5, number_of_outputs)
    skip=int(number_of_outputs//n_frames)

    image_paths = []
    dpi = 100

    fig = plt.figure( figsize=(14,7))
    for it, t in zip( np.arange(len(ts))[::outputEvery][::skip], ts[::outputEvery][::skip]):

        fig.clf()
        pdf = np.load(f"{outputs_directory}/SE_1D/{ID_string}/pdf/pdf_it_{it}.npy")
        trajs = np.load(f"{outputs_directory}/SE_1D/{ID_string}/trajs/trajs_it_{it}.npy")

        # PDF + POTENTIAL + TRAJECTORIES ##############################################
        ax = fig.add_subplot(111)
        ax.plot(xs, pdf, label="Probablity Density")
        ax.set_xlabel("x")
        ax.set_ylabel("pdf")
        ax.set_title(f"Trajectories and Probability Density\n it={it} t={t:.4}")
        plt.legend()
        ax2=ax.twinx()
        ax2.plot(xs, V_i, label="Potential", color=u'#ff7f0e')
        ax2.plot(trajs[:,0], [np.mean(V_i)]*numTrajs, 'o',color='black',
                markersize=1,label="Bohmian Trajectories")
        ax2.set_ylabel("Potential")
        image=f"{outputs_directory}/SE_1D/{ID_string}/figs/it_{it}.png"
        plt.legend()
        plt.savefig(image, dpi=dpi)
        image_paths.append(image)

    import imageio
    fps=0.7
    images_for_animation = [ imageio.v2.imread(image_path) for image_path in image_paths]
    imageio.mimsave(f"{outputs_directory}/SE_1D/{ID_string}/CN_SE.gif", images_for_animation, duration=fps**-1*len(images_for_animation))
