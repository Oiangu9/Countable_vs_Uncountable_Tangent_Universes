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
def cuda_get_discrete_U_L(Nx, Ny, Nz, dx, dy, dz, dt, mx, my, mz, hbar):
    main_diagonal = 1+1j*dt*0.5*hbar*(1/dx**2/mx+1/dy**2/my+1/dz**2/mz)*cnp.ones(Nx*Ny*Nz, dtype=complex_dtype)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                main_diagonal[k*Nx*Ny+j*Nx+i] += 1j*dt*V_ijk[i,j,k]/2/hbar
    x_diagonals = -1j*dt*hbar/(4*mx*dx**2)*cnp.ones(Nx*Ny*Nz-1, dtype=complex_dtype)
    y_diagonals = -1j*dt*hbar/(4*my*dy**2)*cnp.ones(Nx*Ny*Nz-Nx, dtype=complex_dtype)
    z_diagonals = -1j*dt*hbar/(4*mz*dz**2)*cnp.ones(Nx*Ny*Nz-Nx*Ny, dtype=complex_dtype)
    # There are some zeros we need to place in these diagonals
    for j in range(Ny*Nz-1):
        x_diagonals[(j+1)*Nx-1] = 0
    for k in range(Nz-1):
        y_diagonals[ (k+1)*Nx*Ny-Nx:(k+1)*Nx*Ny ] = 0

    return csp.sparse.diags( diagonals=
        (main_diagonal, x_diagonals, x_diagonals,y_diagonals, y_diagonals, z_diagonals, z_diagonals),
        offsets=cnp.array([0, 1, -1, Nx, -Nx, Nx*Ny, -Nx*Ny]), dtype=complex_dtype,format='csc')

if __name__ == "__main__":
    # SIMULATION PARAMETERS ##################################
    args = sys.argv # The path to the settings file should be given
    ##print("Input arguments", sys.argv)
    assert len(args)==5, "Paths or experiment name not given correctly!"
    ID_string = str(args[1])
    path_to_settings = str(args[2])
    path_to_psi_and_potential=str(args[3])
    outputs_directory = str(args[4])
    ''' Expected arguments:
    - ID for the output directroy naming
    - path to the settings file for the simulation
    - name of the .py file defining the psi0(x,y) and potential(x,y) functions the "task-finished" flag will be generated there in a FLAGS subdirectory
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
    Nx,Ny, Nz = Ns
    # Time grid
    ts = np.array([dt*j for j in range(numIts)])

    # Increments to be used per dimension
    dxs = [(xuppers[j]-xlowers[j])/(Ns[j]-1) for j in range(3)] # (dx, dy, dz)
    dx,dy,dz = dxs

    #Create coordinates at which the solution will be calculated
    nodes = [cnp.linspace(xlowers[j], xuppers[j], Ns[j]) for j in range(3)] # (xs, ys)
    xs,ys,zs = nodes

    # SIMULATION SCENRAIO AND INITIAL STATE #################################
    try:
        sys.path.append(''.join(path_to_psi_and_potential.split('/SETTINGS_')[:-1]))
        module=importlib.import_module('SETTINGS_'+path_to_psi_and_potential.split('/SETTINGS_')[-1])
        psi0 = getattr(module, 'psi0')
        chosenV = getattr(module, 'chosenV')

    except:
        raise AssertionError ("psi0 and chosenV functions not correctly defined in the given file!")
    # PREPARE ARRAYS FOR SIMULATION #########################################
    grid=cnp.moveaxis(cnp.array(np.meshgrid(*nodes)).swapaxes(1,2),0,-1) #[Nx,Ny,Nz,3]
    V_ijk = chosenV(grid)

    psi = cnp.zeros(Nx*Ny*Nz, dtype=complex_dtype)
    psi = psi0(grid).flatten('F')

    # initialize Bohmian trajectories
    # first get the pdf
    #psi0 = cnp.asnumpy(psi)
    pdf0 = (psi.conj()*psi).real
    pdf0 = pdf0/pdf0.sum() # normalize strictly

    # sample randomly
    initial_trajs_idx = cnp.random.choice( pdf0.shape[0],
                replace=True, size=(numTrajs), p=pdf0 ) #[numTrajs] indices
    # need to convert them to positions
    k_s = initial_trajs_idx//(Nx*Ny)
    rest = initial_trajs_idx%(Nx*Ny)
    j_s = rest//Nx
    i_s = rest%Nx

    # avoiding edges - probability density is zero there
    i_s[i_s==0] = 1
    i_s[i_s==Ns[0]-1] = Ns[0]-2
    j_s[j_s==0] = 1
    j_s[j_s==Ns[1]-1] = Ns[1]-2
    k_s[k_s==0] = 1
    k_s[k_s==Ns[2]-1] = Ns[2]-2

    trajs = cnp.zeros((numTrajs, 6), dtype=real_dtype) #[numTrajs, 6=3posit+3momtum]
    for tr, (i, j, k) in enumerate(zip(i_s,j_s,k_s)):
        trajs[tr, 0] = nodes[0][i]
        trajs[tr, 1] = nodes[1][j]
        trajs[tr, 2] = nodes[2][k]

    cdxs = cnp.array(dxs)[cnp.newaxis,:]
    cNs = cnp.array(Ns)[cnp.newaxis, :]
    cxlowers = cnp.array(xlowers)[cnp.newaxis, :]
    cxuppers = cnp.array(xuppers)[cnp.newaxis, :]
    cms = cnp.array(ms)[cnp.newaxis, :]

    try:
        # SANITY ##############################
        # Chosen Potential and Trajectories Plot for initial time
        if use_cuda_numpy:
            V_ijk=cnp.asnumpy(V_ijk)
            trajs_numpy = cnp.asnumpy(trajs)
            grid = cnp.asnumpy(grid)
        else:
            trajs_numpy = trajs
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        maxim = V_ijk.max()
        minim = V_ijk.min()
        zati=3
        level_surface = grid[ V_ijk>maxim/zati, :]
        colormap = ax.scatter3D(level_surface[:,0], level_surface[:,1], level_surface[:,2],
                            c=V_ijk[V_ijk>maxim/zati],
                cmap='hot_r', s=2, alpha=V_ijk[V_ijk>maxim/zati]/maxim ) #, antialiased=True)
        ax.set_xlim((xlowers[0], xuppers[1]))
        ax.set_ylim((xlowers[1], xuppers[1]))
        ax.set_zlim((xlowers[2], xuppers[2]))

        fig.colorbar(colormap, fraction=0.04, location='left')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Potential energy>{maxim/3:.3}")
        #ax.view_init(elev=30, azim=30)
        ax.scatter(trajs_numpy[:,0], trajs_numpy[:,1], trajs_numpy[:,2], c="black", s=3,
                    alpha=1)
        os.makedirs(f"{outputs_directory}/SE_3D/{ID_string}/figs/", exist_ok=True)
        image=f"{outputs_directory}/SE_3D/{ID_string}/figs/energy_potential.png"
        plt.savefig(image, dpi=150)
    except:
        os.makedirs(f"{outputs_directory}/SE_3D/{ID_string}/figs/", exist_ok=True)

    # Propagator #################################################
    U_L = cuda_get_discrete_U_L(*Ns, *dxs, dt, *ms, hbar)
    U_R = U_L.conj()
    UL_LUdec = clg.splu( U_L )
    # Create Output Directory ######################3
    os.makedirs(f"{outputs_directory}/SE_3D/{ID_string}/pdf/", exist_ok=True)
    os.makedirs(f"{outputs_directory}/SE_3D/{ID_string}/trajs/", exist_ok=True)
    os.makedirs(f"{outputs_directory}/SE_3D/{ID_string}/moms/", exist_ok=True)

    # Time Iteration LOOP #####################################3
    ############################################################
    ############################################################
    dpsi_dx = cnp.zeros(Ns, dtype=complex_dtype)
    dpsi_dy = cnp.zeros(Ns, dtype=complex_dtype)
    dpsi_dz = cnp.zeros(Ns, dtype=complex_dtype)
    p = cnp.zeros(dpsi_dx.shape+(3, ), dtype=real_dtype) #[3, Nx, Ny, Nz]

    for it, t in enumerate(ts):
        psi_tensor = psi.reshape(Ns[::-1]).swapaxes(0,-1)+1e-10 #[Nx,Ny,Nz]

        # BOHMIAN MOMENTUM FIELD #####################################
        # first the gradient of the wavefunction at each point
        # X
        # boundaries with simple Euler formula O(dx)
        dpsi_dx[0, :, :] = (psi_tensor[1, :, :]-psi_tensor[0, :, :])/dx
        dpsi_dx[-1, :, :] = (psi_tensor[-1, :, :]-psi_tensor[-2, :, :])/dx
        # next boundaries with centered difference O(dx**2)
        dpsi_dx[1,:,:] = (psi_tensor[2,:,:]-psi_tensor[0,:,:])/(2*dx)
        dpsi_dx[-2,:,:] = (psi_tensor[-1,:,:]-psi_tensor[-3,:,:])/(2*dx)
        # rest with O(dx**4) centered difference
        dpsi_dx[2:-2,:,:] = (-psi_tensor[4:,:,:]+8*psi_tensor[3:-1,:,:]-8*psi_tensor[1:-3,:,:]+psi_tensor[:-4,:,:])/(12*dx)

        # DY
        # boundaries with simple Euler formula O(dx)
        dpsi_dy[:, 0, :] = (psi_tensor[:,1, :]-psi_tensor[:,0, :])/dy
        dpsi_dy[:,-1, :] = (psi_tensor[:,-1, :]-psi_tensor[:,-2, :])/dy
        # next boundaries with centered difference O(dx**2)
        dpsi_dy[:,1,:] = (psi_tensor[:,2,:]-psi_tensor[:,0,:])/(2*dy)
        dpsi_dy[:,-2,:] = (psi_tensor[:,-1,:]-psi_tensor[:,-3,:])/(2*dy)
        # rest with O(dx**4) centered difference
        dpsi_dy[:,2:-2,:] = (-psi_tensor[:,4:,:]+8*psi_tensor[:,3:-1,:]-8*psi_tensor[:,1:-3,:]+psi_tensor[:,:-4,:])/(12*dy)

        # DZ
        # boundaries with simple Euler formula O(dx)
        dpsi_dz[:,:,0] = (psi_tensor[:,:,1]-psi_tensor[:,:,0])/dz
        dpsi_dz[:,:,-1] = (psi_tensor[:,:,-1]-psi_tensor[:,:,-2])/dz
        # next boundaries with centered difference O(dx**2)
        dpsi_dz[:,:,1] = (psi_tensor[:,:,2]-psi_tensor[:,:,0])/(2*dz)
        dpsi_dz[:,:,-2] = (psi_tensor[:,:,-1]-psi_tensor[:,:,-3])/(2*dz)
        # rest with O(dx**4) centered difference
        dpsi_dz[:,:,2:-2] = (-psi_tensor[:,:,4:]+8*psi_tensor[:,:,3:-1]-8*psi_tensor[:,:,1:-3]+psi_tensor[:,:,:-4])/(12*dz)

        # px, py, pz
        p[:,:,:,0] = hbar*(dpsi_dx/psi_tensor).imag
        p[:,:,:,1] = hbar*(dpsi_dy/psi_tensor).imag
        p[:,:,:,2] = hbar*(dpsi_dz/psi_tensor).imag

        # TRAJ MOMENTA ##################################################
        # if no trajectory surpasses below the node 0 or J-1 at any time,
        # traj will always be among (0,J-1) and traj//dxs will be among [0,J-1]
        trajs_idxs = (((trajs[:,:3]-cxlowers)//cdxs).T).astype(cnp.uint) # [3, numTrajs] the closest index from below along each axis
        # relative distance to the closest index from below point along each dimension
        # the closer, the bigger its weight should be for the trajectory propagation
        ratx_down = ((trajs[:,0]-xs[ trajs_idxs[0] ])/(xs[ trajs_idxs[0]+1 ]-xs[ trajs_idxs[0] ]))[:,cnp.newaxis]
        raty_down = ((trajs[:,1]-ys[ trajs_idxs[1] ])/(ys[ trajs_idxs[1]+1 ]-ys[ trajs_idxs[1] ]))[:,cnp.newaxis]
        ratz_down = ((trajs[:,2]-zs[ trajs_idxs[2] ])/(zs[ trajs_idxs[2]+1 ]-zs[ trajs_idxs[2] ]))[:,cnp.newaxis]
        # Get the interpolated momenta
        trajs[:, 3:] = ratx_down*raty_down*ratz_down* p[ trajs_idxs[0]+1, trajs_idxs[1]+1, trajs_idxs[2]+1 ] +\
            (1-ratx_down)*raty_down*ratz_down* p[ trajs_idxs[0], trajs_idxs[1]+1, trajs_idxs[2]+1 ] +\
            ratx_down*(1-raty_down)*ratz_down* p[ trajs_idxs[0]+1, trajs_idxs[1], trajs_idxs[2]+1 ] +\
            ratx_down*raty_down*(1-ratz_down)* p[ trajs_idxs[0]+1, trajs_idxs[1]+1, trajs_idxs[2] ] +\
            (1-ratx_down)*(1-raty_down)*ratz_down* p[ trajs_idxs[0], trajs_idxs[1], trajs_idxs[2]+1 ] +\
            ratx_down*(1-raty_down)*(1-ratz_down)* p[ trajs_idxs[0]+1, trajs_idxs[1], trajs_idxs[2] ] +\
            (1-ratx_down)*raty_down*(1-ratz_down)* p[ trajs_idxs[0], trajs_idxs[1]+1, trajs_idxs[2] ] +\
            (1-ratx_down)*(1-raty_down)*(1-ratz_down)* p[ trajs_idxs[0], trajs_idxs[1], trajs_idxs[2] ]

        # OUTPUT #####################################################
        if it%outputEvery == 0:
            #print(f"\n > It {it}/{numIts}")
            # compute the magnitude squared of the wavefunction
            pdf = (psi_tensor.conj()*psi_tensor).real
            # Approximate the norm
            #print(f"   Iteration {it} Approx.Norm = {pdf.sum()*dx*dy*dz:.4}")
            if use_cuda_numpy:
                pdf = cnp.asnumpy(pdf)
                trajs_numpy = cnp.asnumpy(trajs)
                p_numpy = cnp.asnumpy(p)
            else:
                trajs_numpy = trajs
                p_numpy = p

            np.save(f"{outputs_directory}/SE_3D/{ID_string}/pdf/pdf_it_{it}.npy",
                    pdf, allow_pickle=True) #[Nx,Ny,Nz]
            np.save(f"{outputs_directory}/SE_3D/{ID_string}/trajs/trajs_it_{it}.npy",
                    trajs_numpy, allow_pickle=True) #[numTrajs, 6]
            np.save(f"{outputs_directory}/SE_3D/{ID_string}/moms/momentum_field_it_{it}.npy",
                    p_numpy, allow_pickle=True) #[Nx,Ny,Nz, 6]


        # Evolve trajectories using the interpolated velocity field
        trajs[:, :3] = trajs[:,:3] + dt*trajs[:,3:]/cms #[numTrajs, 3]

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
        xs,ys,zs = nodes

    # Generate max 5 frames as a sanity check, equispaced ##################
    number_of_outputs = int(numIts//outputEvery)
    n_frames = min(5, number_of_outputs)
    skip=int(number_of_outputs//n_frames)

    image_paths = []
    dpi = 100
    # Generate Santiy check frames jic
    every=1 # Only take one data point every this number in each axis to plot
    grid=np.array(np.meshgrid(*nodes)).swapaxes(1,2)[:,::every, ::every, ::every] #[3,Nx,Ny,Nz]
    #print(f"Using a mesh in the plot of {grid.shape}")
    fig = plt.figure( figsize=(21,7))

    plt.style.use('dark_background')

    for it, t in zip( np.arange(len(ts))[::outputEvery][::skip], ts[::outputEvery][::skip]):
        #print(f"\n > It {it}/{numIts}")
        fig.clf()
        pdf = np.load(f"{outputs_directory}/SE_3D/{ID_string}/pdf/pdf_it_{it}.npy")
        trajs = np.load(f"{outputs_directory}/SE_3D/{ID_string}/trajs/trajs_it_{it}.npy")

        ax = fig.add_subplot(131, projection='3d')
        # PDF TRANSP ############################################################
        maxim = pdf.max()
        minim = pdf.min()
        level_surface = grid[:, pdf>maxim/3]
        colormap = ax.scatter(*level_surface, c=pdf[pdf>maxim/3],
                cmap='hot_r', s=2, alpha=pdf[pdf>maxim/3]/maxim ) #, antialiased=True)
        ax.set_xlim((xlowers[0], xuppers[1]))
        ax.set_ylim((xlowers[1], xuppers[1]))
        ax.set_zlim((xlowers[2], xuppers[2]))
        #ax.view_init(elev=-30, azim=30)

        fig.colorbar(colormap, fraction=0.04, location='left')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Probability Density>{maxim/3:.3}\n it={it} t={t:.4}")

        # PDF SURFACE LEVELS #####################################################
        ax = fig.add_subplot(132, projection='3d')
        level_surface = grid[:,pdf>maxim/3]
        ax.set_zlim(xlowers[-1], xuppers[-1])
        for k,z in enumerate(zs):
            cset = ax.contour(grid[0,:,:,0], grid[1,:,:,0], pdf[:,:,k],
                        7, zdir='z', offset=z, cmap='hot_r', vmin=minim, vmax=maxim,
                             alpha=pdf[:,:,k].max()/maxim)
        ax.set_title(f"Probability Density\n it={it} t={t:.4}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(colormap, fraction=0.04, location='bottom', ax=ax)
        #ax.view_init(elev=-30, azim=30)

        # PDF + TRAJECTORIES ##############################################
        ax = fig.add_subplot(133, projection='3d')
        #colormap = ax.scatter3D(*grid, c=pdf[ ::every, ::every, ::every],
        #               cmap='hot_r', s=0.1, alpha=0.4 ) #, antialiased=True)
        cset = ax.contour(grid[0,:,:,0], grid[1,:,:,0], pdf.sum(axis=2), 7, zdir='z', offset=xlowers[2], cmap='hot_r')
        cset = ax.contour(grid[0,:,0,:], pdf.sum(axis=1), grid[2,:,0,:], 7, zdir='y', offset=xuppers[1], cmap='hot_r')
        cset = ax.contour(pdf.sum(axis=0), grid[1,0,:,:], grid[2,0,:,:], 7, zdir='x', offset=xlowers[0], cmap='hot_r')
        #fig.colorbar(cset, fraction=0.04, location='left')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Trajectories and Projected Probability Density Contours\n it={it} t={t:.4}")
        ax.scatter(trajs[:,0], trajs[:,1], trajs[:,2], c="white", s=3,
                    alpha=0.7)
        #ax.view_init(elev=-30, azim=30)

        image=f"{outputs_directory}/SE_3D/{ID_string}/figs/it_{it}.png"
        plt.savefig(image, dpi=dpi)
        image_paths.append(image)


    import imageio
    fps=0.7
    images_for_animation = [ imageio.v2.imread(image_path) for image_path in image_paths]
    imageio.mimsave(f"{outputs_directory}/SE_3D/{ID_string}/CN_SE.gif", images_for_animation, duration=fps**-1*len(images_for_animation))
