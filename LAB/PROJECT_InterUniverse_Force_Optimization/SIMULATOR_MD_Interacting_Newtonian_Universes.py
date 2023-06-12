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
    import cupyx.scipy.linalg as clg

else:
    import numpy as cnp
    import scipy as csp
    import scipy.sparse.linalg as clg


# ROUTINES ###########################################
def get_gradient_vector_field_3D(potential_field, real_dtype, dx, dy, dz):
    # Assumes potential_field is [Nx, Ny, Nz]
    gradient = cnp.zeros(potential_field.shape+(3,), dtype=real_dtype)
    # X
    # boundaries with simple Euler formula O(dx)
    gradient[ 0, :, :, 0] = (potential_field[1, :, :]-potential_field[0, :, :])/dx
    gradient[ -1, :, :, 0] = (potential_field[-1, :, :]-potential_field[-2, :, :])/dx
    # next boundaries with centered difference O(dx**2)
    gradient[ 1,:,:, 0] = (potential_field[2,:,:]-potential_field[0,:,:])/(2*dx)
    gradient[ -2,:,:, 0] = (potential_field[-1,:,:]-potential_field[-3,:,:])/(2*dx)
    # rest with O(dx**4) centered difference
    gradient[ 2:-2,:,:, 0] = (-potential_field[4:,:,:]+8*potential_field[3:-1,:,:]-8*potential_field[1:-3,:,:]+potential_field[:-4,:,:])/(12*dx)

    # DY
    # boundaries with simple Euler formula O(dx)
    gradient[ :, 0, :, 1] = (potential_field[:,1, :]-potential_field[:,0, :])/dx
    gradient[ :,-1, :, 1] = (potential_field[:,-1, :]-potential_field[:,-2, :])/dx
    # next boundaries with centered difference O(dx**2)
    gradient[ :,1,:, 1] = (potential_field[:,2,:]-potential_field[:,0,:])/(2*dx)
    gradient[ :,-2,:, 1] = (potential_field[:,-1,:]-potential_field[:,-3,:])/(2*dx)
    # rest with O(dx**4) centered difference
    gradient[ :,2:-2,:, 1] = (-potential_field[:,4:,:]+8*potential_field[:,3:-1,:]-8*potential_field[:,1:-3,:]+potential_field[:,:-4,:])/(12*dx)

    # DZ
    # boundaries with simple Euler formula O(dx)
    gradient[ :,:,0, 2] = (potential_field[:,:,1]-potential_field[:,:,0])/dx
    gradient[ :,:,-1, 2] = (potential_field[:,:,-1]-potential_field[:,:,-2])/dx
    # next boundaries with centered difference O(dx**2)
    gradient[ :,:,1, 2] = (potential_field[:,:,2]-potential_field[:,:,0])/(2*dx)
    gradient[ :,:,-2, 2] = (potential_field[:,:,-1]-potential_field[:,:,-3])/(2*dx)
    # rest with O(dx**4) centered difference
    gradient[ :,:,2:-2, 2] = (-potential_field[:,:,4:]+8*potential_field[:,:,3:-1]-8*potential_field[:,:,1:-3]+potential_field[:,:,:-4])/(12*dx)
    return gradient



def get_gradient_vector_field_2D(potential_field, real_dtype, dx, dy):
    # Assumes potential_field is [Nx, Ny]
    gradient = cnp.zeros(potential_field.shape+(2,), dtype=real_dtype)
    # X
    # boundaries with simple Euler formula O(dx)
    gradient[0, :, 0] = (potential_field[1, :]-potential_field[0, :])/dx
    gradient[-1, :, 0] = (potential_field[-1, :]-potential_field[-2, :])/dx
    # next boundaries with centered difference O(dx**2)
    gradient[1,:, 0] = (potential_field[2,:]-potential_field[0,:])/(2*dx)
    gradient[-2,:, 0] = (potential_field[-1,:]-potential_field[-3,:])/(2*dx)
    # rest with O(dx**4) centered difference
    gradient[2:-2,:, 0] = (-potential_field[4:,:]+8*potential_field[3:-1,:]-8*potential_field[1:-3,:]+potential_field[:-4,:])/(12*dx)

    # DY
    # boundaries with simple Euler formula O(dx)
    gradient[:, 0, 1] = (potential_field[:,1]-potential_field[:,0])/dy
    gradient[:,-1, 1] = (potential_field[:,-1]-potential_field[:,-2])/dy
    # next boundaries with centered difference O(dx**2)
    gradient[:,1, 1] = (potential_field[:,2]-potential_field[:,0])/(2*dy)
    gradient[:,-2, 1] = (potential_field[:,-1]-potential_field[:,-3])/(2*dy)
    # rest with O(dx**4) centered difference
    gradient[:,2:-2, 1] = (-potential_field[:,4:]+8*potential_field[:,3:-1]-8*potential_field[:,1:-3]+potential_field[:,:-4])/(12*dy)
    return gradient

def get_gradient_vector_field_1D(potential_field, real_dtype, dx):
    # Assumes potential_field is [Nx]
    gradient = cnp.zeros(potential_field.shape+(1,), dtype=real_dtype)
    # X
    # boundaries with simple Euler formula O(dx)
    gradient[0,0] = (potential_field[1]-potential_field[0])/dx
    gradient[-1,0] = (potential_field[-1]-potential_field[-2])/dx
    # next boundaries with centered difference O(dx**2)
    gradient[1,0] = (potential_field[2]-potential_field[0])/(2*dx)
    gradient[-2,0] = (potential_field[-1]-potential_field[-3])/(2*dx)
    # rest with O(dx**4) centered difference
    gradient[2:-2,0] = (-potential_field[4:]+8*potential_field[3:-1]-8*potential_field[1:-3]+potential_field[:-4])/(12*dx)
    return gradient #[Nx,1]



def interpolate_traj_force_from_force_field_3D(trajs, force_field, xs, ys, zs):
    # if no trajectory surpasses below the node 0 or J-1 at any time,
    # traj will always be among (0,J-1) and traj//dxs will be among [0,J-1]
    trajs_idxs = (((trajs[:,:3]-xlowers)//dxs).T).astype(cnp.uint) # [3, numTrajs] the closest index from below along each axis
    # relative distance to the closest index from below point along each dimension
    # the closer, the bigger its weight should be for the trajectory propagation
    ratx_down = ((trajs[:,0]-xs[ trajs_idxs[0] ])/(xs[ trajs_idxs[0]+1 ]-xs[ trajs_idxs[0] ]))[:,cnp.newaxis]
    raty_down = ((trajs[:,1]-ys[ trajs_idxs[1] ])/(ys[ trajs_idxs[1]+1 ]-ys[ trajs_idxs[1] ]))[:,cnp.newaxis]
    ratz_down = ((trajs[:,2]-zs[ trajs_idxs[2] ])/(zs[ trajs_idxs[2]+1 ]-zs[ trajs_idxs[2] ]))[:,cnp.newaxis]
    # Get the interpolated force
    return ratx_down*raty_down*ratz_down* force_field[ trajs_idxs[0]+1, trajs_idxs[1]+1, trajs_idxs[2]+1 ] +\
        (1-ratx_down)*raty_down*ratz_down* force_field[ trajs_idxs[0], trajs_idxs[1]+1, trajs_idxs[2]+1 ] +\
        ratx_down*(1-raty_down)*ratz_down* force_field[ trajs_idxs[0]+1, trajs_idxs[1], trajs_idxs[2]+1 ] +\
        ratx_down*raty_down*(1-ratz_down)* force_field[ trajs_idxs[0]+1, trajs_idxs[1]+1, trajs_idxs[2] ] +\
        (1-ratx_down)*(1-raty_down)*ratz_down* force_field[ trajs_idxs[0], trajs_idxs[1], trajs_idxs[2]+1 ] +\
        ratx_down*(1-raty_down)*(1-ratz_down)* force_field[ trajs_idxs[0]+1, trajs_idxs[1], trajs_idxs[2] ] +\
        (1-ratx_down)*raty_down*(1-ratz_down)* force_field[ trajs_idxs[0], trajs_idxs[1]+1, trajs_idxs[2] ] +\
        (1-ratx_down)*(1-raty_down)*(1-ratz_down)* force_field[ trajs_idxs[0], trajs_idxs[1], trajs_idxs[2] ]

def interpolate_traj_force_from_force_field_2D(trajs, force_field, xs, ys):
    # if no trajectory surpasses below the node 0 or J-1 at any time,
    # traj will always be among (0,J-1) and traj//dxs will be among [0,J-1]
    trajs_idxs = (((trajs[:,:2]-xlowers)//dxs).T).astype(cnp.uint) # [2, numTrajs] the closest index from below along each axis
    # relative distance to the closest index from below point along each dimension
    # the closer, the bigger its weight should be for the trajectory propagation
    ratx_down = ((trajs[:,0]-xs[ trajs_idxs[0] ])/(xs[ trajs_idxs[0]+1 ]-xs[ trajs_idxs[0] ]))[:,cnp.newaxis]
    raty_down = ((trajs[:,1]-ys[ trajs_idxs[1] ])/(ys[ trajs_idxs[1]+1 ]-ys[ trajs_idxs[1] ]))[:,cnp.newaxis]
    # Interpolate momentum
    return ratx_down*raty_down* force_field[ trajs_idxs[0]+1, trajs_idxs[1]+1 ] +\
        (1-ratx_down)*raty_down* force_field[ trajs_idxs[0], trajs_idxs[1]+1] +\
        ratx_down*(1-raty_down)* force_field[ trajs_idxs[0]+1, trajs_idxs[1] ] +\
        (1-ratx_down)*(1-raty_down)* force_field[ trajs_idxs[0], trajs_idxs[1]] #[numTrajs,2]

def interpolate_traj_force_from_force_field_1D(trajs, force_field, xs):
    # if no trajectory surpasses below the node 0 or J-1 at any time,
    # traj will always be among (0,J-1) and traj//dxs will be among [0,J-1]
    trajs_idxs = (((trajs-xlowers)//dxs).T).astype(cnp.uint) # [1, numTrajs] the closest index from below along each axis
    # relative distance to the closest index from below point along each dimension
    # the closer, the bigger its weight should be for the trajectory propagation
    ratx_down = ((trajs[:,0]-xs[ trajs_idxs[0] ])/(xs[ trajs_idxs[0]+1 ]-xs[ trajs_idxs[0] ]))[:,cnp.newaxis]
    # Interpolate momentum
    return ratx_down* force_field[ trajs_idxs[0]+1 ] +\
        (1-ratx_down)* force_field[ trajs_idxs[0] ]  #[numTrajs,1]


def estimate_pdf_from_trajs(trajs, grid, sigma, real_dtype, norm):
    pdf = cnp.zeros(grid.shape[:-1], dtype=real_dtype)
    # vectorizing is is too memory costly, it would require
    # one whole grid per trajectory! so better use a for loop
    N = 1/(np.sqrt((2*np.pi)**trajs.shape[-1])*sigma**trajs.shape[-1]*trajs.shape[0])
    for traj in trajs:
        pdf+=N*cnp.exp(
            np.sum( -(grid-traj)**2/(2*sigma)**2, -1)) # for arbitrary
    return norm*pdf/cnp.sum(pdf) #[N1,N2,...]

def estimate_pdf_over_trajs(trajs, sigma, real_dtype):
    pdf = cnp.zeros(trajs.shape[0], dtype=real_dtype)
    N = 1/(np.sqrt((2*np.pi)**trajs.shape[-1])*sigma**trajs.shape[-1]*trajs.shape[0])
    for traj in trajs:
        pdf += N*cnp.exp(
            np.sum(-(trajs-traj)**2/(2*sigma)**2, -1) )

    return pdf #[numTrajs]

if __name__ == "__main__":
    # SIMULATION PARAMETERS ##################################
    args = sys.argv # The path to the settings file should be given
    ##print("Input arguments", sys.argv)
    assert len(args)==8, "Paths or experiment name not given correctly!"
    ID_string = str(args[1]) #
    path_to_settings = str(args[2])
    path_to_psi_and_potential=str(args[3])
    outputs_directory = str(args[4])
    A = float(args[5])
    K = float(args[6])
    reference_traj_npy_file = str(args[7])
    ''' Expected arguments:
    - ID of the execution, for the directory names
    - path to the settings file for the simulation
    - name of the .py file defining the psi0(x,y) and potential(x,y) functions the "task-finished" flag will be generated there in a FLAGS subdirectory
    - path to outputs directory
    - A of the interuniverse potential  as a string
    - K as a string
    - reference_traj_npy_file
    '''
    # USING SE DATA AS REFERENCE! ###########################################
    with open(f"{path_to_settings}", "r") as f:
        numIts_SE = int(f.readline().split("numIts_SE ")[1])
        dt_SE = float(f.readline().split("dt_SE ")[1])
        outputEvery_SE = int(f.readline().split("outputEvery_SE ")[1])
        Ns_pdf = [int(x) for x in f.readline().split("Ns_SE_and_Newt_pdf ")[1].split('[')[1].split(']')[0].split(',')]
        xlowers = [float(x) for x in f.readline().split("xlowers ")[1].split('[')[1].split(']')[0].split(',')]
        xuppers = [float(x) for x in f.readline().split("xuppers ")[1].split('[')[1].split(']')[0].split(',')]
        numDofUniv = int(f.readline().split("numDofUniv ")[1])
        numDofPartic = int(f.readline().split("numDofPartic ")[1])
        numTrajs = int(f.readline().split("numTrajs ")[1])
        hbar = float(f.readline().split("hbar ")[1])
        ms = [float(x) for x in f.readline().split("ms ")[1].split('[')[1].split(']')[0].split(',')]
        complex_dtype = eval(f.readline().split("complex_dtype ")[1])
        real_dtype =  eval(f.readline().split("real_dtype ")[1])
        K_coulomb = float(f.readline().split("K_coulomb ")[1])
        qs = [float(x) for x in f.readline().split("qs ")[1].split('[')[1].split(']')[0].split(',')]
        dt = float(f.readline().split("dt_Newt ")[1])
        Ns_potential = [int(x) for x in f.readline().split("Ns_Newt_pot ")[1].split('[')[1].split(']')[0].split(',')]
        use_Coulomb_potential = bool(int(f.readline().split("use_Coulomb_potential_Newt ")[1]))
        use_scenario_potential = bool(int(f.readline().split("use_scenario_potential_Newt ")[1]))
        sigma = float(f.readline().split("sigma_pdf_estimation ")[1])

    # Tricks to match the output frames - DO NOT EDIT THIS!
    # Adapt the chosen dt to the closest one such that it is possible to match frames
    # That is, make them commensurate
    if dt_SE>dt:
        dt=int(dt_SE//dt)**-1*dt_SE
    else: # we need at least the same rate of outputs to compare one to one
        dt=dt_SE

    ##print(f"Using dt={dt:.3}; dt_SE={dt_SE:.3} -> dt = {dt/dt_SE:.3}*dt_SE")
    use_external_forces_and_pdf_grid=True
    # Match the output number
    outputEvery = int(dt_SE//dt)
    # now match exactly the outputed frames
    outputEvery = outputEvery_SE*outputEvery

    # Total number of iterations
    numIts = int(numIts_SE*(dt_SE//dt))

    # Time grid
    ts = np.array([dt*j for j in range(numIts)])
    output_ts = ts[::outputEvery]
    ts_SE = np.array([dt_SE*j for j in range(numIts_SE)])
    output_ts_SE=ts_SE[::outputEvery_SE]

    # SANITY CHECK for output time iteration matching! ###########
    fig=plt.figure(figsize=(12, 2))
    ax=fig.add_subplot(111)
    ax.vlines(ts[:2*outputEvery_SE*int(dt_SE//dt)], 0, 1,  label="Compute Newtonian time frames", color="red", linewidth=1)
    ax.vlines(ts[:2*outputEvery_SE*int(dt_SE//dt):outputEvery], 1, 2,  label="Output Newtonian time frames", color="black", linewidth=1)
    ax.vlines(ts_SE[:2*outputEvery_SE], -1, 0,  label="Calculated SE time frames", color="blue", linewidth=1)
    ax.vlines(ts_SE[:2*outputEvery_SE:outputEvery_SE], -1, -2,  label="Output SE time frames", color="violet", linewidth=1)
    ax.legend()
    ax.set_xlabel("t")
    os.makedirs(f"{outputs_directory}/MIW/{ID_string}/figs/", exist_ok=True)
    plt.savefig(f"{outputs_directory}/MIW/{ID_string}/figs/compared_output_times.png", dpi=120)

    # Additional initializations #####
    # Increments to be used per dimension
    dxs = [(xuppers[j]-xlowers[j])/(Ns_potential[j]-1) for j in range(numDofUniv)] # (dx, dy, dz)
    dxs = cnp.array(dxs)[cnp.newaxis,:]

    dxs_pdf = [(xuppers[j]-xlowers[j])/(Ns_pdf[j]-1) for j in range(numDofUniv)] # (dx, dy, dz)
    dxs_pdf = cnp.array(dxs_pdf)[cnp.newaxis,:]

    #Create coordinates at which the solution will be calculated
    nodes_pdf = [cnp.linspace(xlowers[j], xuppers[j], Ns_pdf[j]) for j in range(numDofUniv)] # (xs, ys, zs)
    nodes_potential = [cnp.linspace(xlowers[j], xuppers[j], Ns_potential[j]) for j in range(numDofUniv)] # (xs, ys, zs)
    # Create a node mesh
    if numDofUniv==2 or numDofUniv==3:
        grid_pdf=cnp.moveaxis(cnp.array(cnp.meshgrid(*nodes_pdf)).swapaxes(1,2), 0,-1) #[Nx,Ny,Nz,3]
        grid_potential=cnp.moveaxis(cnp.array(cnp.meshgrid(*nodes_potential)).swapaxes(1,2), 0,-1) #[Nx,Ny,Nz,3]
    else:
        grid_pdf = nodes_pdf[0][:, np.newaxis] #[Nx, 1]
        grid_potential= nodes_potential[0][:, np.newaxis] #[Nx, 1]

    xlowers = cnp.array(xlowers)[cnp.newaxis, :]
    xuppers = cnp.array(xuppers)[cnp.newaxis, :]
    ms = np.array(ms)[np.newaxis,:]

    # SCENARIO DATA DEFINITON ###############################################
    # Initialize trajectories and their velocities as done in the reference!
    # but wait until they have been generated
    while not os.path.isfile(reference_traj_npy_file):
        time.sleep(30) # it should not take long
    # yes, this is a big shit in the implementation xD
    trajs = cnp.load(reference_traj_npy_file)
    # [numTrajs, 2*dof_Universe]


    # Extra-Universe forces
    try:
        sys.path.append(''.join(path_to_psi_and_potential.split('/SETTINGS_')[:-1]))
        module=importlib.import_module('SETTINGS_'+path_to_psi_and_potential.split('/SETTINGS_')[-1])
        psi0 = getattr(module, 'psi0')
        chosenV = getattr(module, 'chosenV')

    except:
        raise AssertionError ("psi0 and chosenV functions not correctly defined in the given file!")

    if use_scenario_potential:
        if numDofUniv==3:
            potential_field = chosenV(grid_potential) #[Nx,Ny,Nz]
            external_force_field = -get_gradient_vector_field_3D(
                                potential_field, real_dtype,dxs[0,0],dxs[0,1],dxs[0,2]) #[Nx,Ny,Nz, 3]
            external_force = lambda trajs: interpolate_traj_force_from_force_field_3D(
                            trajs, force_field=external_force_field, xs=nodes_potential[0], ys=nodes_potential[1], zs=nodes_potential[2])
            try:
                # PLOT
                fig = plt.figure(figsize=(10,5))
                ax = fig.add_subplot(121, projection='3d')
                maxim = potential_field.max()
                minim = potential_field.min()
                zati=3
                level_surface = grid[:, potential_field>maxim/zati]
                colormap = ax.scatter(*level_surface, c=potential_field[potential_field>maxim/zati],
                        cmap='hot_r', s=2, alpha=potential_field[potential_field>maxim/zati]/maxim ) #, antialiased=True)
                ax.set_xlim((xlowers[0,0], xuppers[0,1]))
                ax.set_ylim((xlowers[0,1], xuppers[0,1]))
                ax.set_zlim((xlowers[0,2], xuppers[0,2]))

                fig.colorbar(colormap, fraction=0.04, location='left')
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_title(f"Potential energy>{maxim/3:.3}")
                ax.view_init(elev=70, azim=30)


                imagep=f"{outputs_directory}/MIW/{ID_string}/figs/energy_potential.png"
                plt.savefig(imagep, dpi=160)
            except:
                pass

        elif numDofUniv==2:
            potential_field = chosenV(grid_potential) #[Nx,Ny]
            external_force_field = -get_gradient_vector_field_2D(
                                potential_field, real_dtype,dxs[0,0],dxs[0,1]) #[ Nx,Ny, 2]
            external_force = lambda trajs: interpolate_traj_force_from_force_field_2D(
                                trajs, force_field=external_force_field, xs=nodes_potential[0], ys=nodes_potential[1])
            every=1 # Only take one data point every this number in each axis to plot potential

            # PLOT
            fig = plt.figure(figsize=(10,8))
            # Potential energy
            ax = fig.add_subplot(221)
            colormap = ax.imshow(potential_field.T,
                         extent=[xlowers[0,0], xuppers[0,0], xlowers[0,1], xuppers[0,1]]
                                 , origin='lower',cmap='hot_r')
            #plt.axis(aspect='image');
            fig.colorbar(colormap, fraction=0.04, location='left')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Potential Energy Field and initial trajs")
            ax.scatter(trajs[:,0], trajs[:,1], c="black", s=2,alpha=1)
            # Force field
            ax = fig.add_subplot(222)
            fcolormap=ax.quiver(grid_potential[:,:,0].flatten(), grid_potential[:,:,1].flatten(),
                  external_force_field[:,:,0].flatten(), external_force_field[:,:,1].flatten(),
                     np.linalg.norm(external_force_field, axis=-1).flatten(), cmap='hot_r')
            fig.colorbar(fcolormap, fraction=0.04, location='right')
            ax.set_aspect('equal')
            #ax.quiver(grid_potential[:,:,:,0], grid_potential[:,:,:,1], grid_potential[:,:,:,2],
            #      external_force_field[:,:,:,0], external_force_field[:,:,:,1], external_force_field[:,:,:,2], length=0.2, normalize=True)
            ax.set_title("Force Field")
            ax = fig.add_subplot(223)
            forces_mag=np.linalg.norm(external_force_field, axis=-1)
            colormap = ax.imshow(forces_mag.T,
                         extent=[xlowers[0,0], xuppers[0,0], xlowers[0,1], xuppers[0,1]]
                                 , origin='lower',cmap='hot_r')
            fig.colorbar(colormap, fraction=0.04, location='left')
            ax.set_title("Force magnitudes")
            ax = fig.add_subplot(224)
            fcolormap = ax.imshow(forces_mag.T,
                         extent=[xlowers[0,0], xuppers[0,0], xlowers[0,1], xuppers[0,1]]
                                 , origin='lower',cmap='hot_r')
            #plt.axis(aspect='image');
            #fig.colorbar(colormap, fraction=0.04, location='right')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Potential Energy Field, \ninitial trajs, initial forces")
            ax.scatter(trajs[:,0], trajs[:,1], c="black", s=2,alpha=1)
            forces_on_trajs = external_force(trajs)
            fnorm = np.linalg.norm(forces_on_trajs, axis=-1)+1e-8
            colormap = ax.quiver(trajs[:,0], trajs[:,1],
                forces_on_trajs[:,0]/fnorm, forces_on_trajs[:,1]/fnorm,
                     fnorm, cmap='hot_r',
                        clim=colormap.get_clim(), scale=1,
                                units='inches')
            fig.colorbar(fcolormap, fraction=0.04, location='right', label="force mag")

            os.makedirs(f"{outputs_directory}/MIW/{ID_string}/figs/", exist_ok=True)
            imagep=f"{outputs_directory}/MIW/{ID_string}/figs/energy_potential.png"
            plt.savefig(imagep, dpi=160)


        else:
            potential_field = chosenV(grid_potential) #[Nx]
            external_force_field = -get_gradient_vector_field_1D(
                                potential_field, real_dtype,dxs[0]) #[ Nx, 1]
            external_force = lambda trajs: interpolate_traj_force_from_force_field_1D(
                                trajs, force_field=external_force_field, xs=nodes_potential[0])
            fig = plt.figure(figsize=(14,5))
            ax = fig.add_subplot(121)
            ax.plot(grid_potential[:,0], potential_field, label="Potential Energy", color=u'#ff7f0e')
            ax.plot(trajs[:,0], [np.mean(potential_field)]*numTrajs,'o',
                    c="black", markersize=1, label="Initial Trajectories")
            ax.set_xlabel("x")
            ax.set_ylabel("Potential Energy")
            ax.set_title(f"Potential Energy Field\nand Initial Trajectories")
            ax = fig.add_subplot(122)
            ax.plot(grid_potential[:,0], external_force_field[:,0], label="Force")
            ax.plot(trajs[:,0], [np.mean(external_force_field)]*numTrajs, 'o', markersize=1,
                    label="Trajectories", color='black')
            ax.set_title("Force field")
            os.makedirs(f"{outputs_directory}/MIW/{ID_string}/figs/", exist_ok=True)
            imagep=f"{outputs_directory}/MIW/{ID_string}/figs/energy_potential.png"
            plt.savefig(imagep, dpi=120, bbox_inches='tight')


        scenario_forces = [
                external_force
            ] # external forces
    else:
        def chosenV(grid, real_dtype=cnp.single):
            return cnp.zeros(grid.shape[:-1], dtype=real_dtype)
        potential_field = chosenV(grid_pdf) #[Nx] for plotting purposes
        scenario_forces = []
    # Inter-particle Intra-Universe Forces
    def coulomb_force(trajs, qs, numDofPartic, K):
        numPartics = trajs.shape[1]//numDofPartic
        trajsPerParticl = trajs.reshape(trajs.shape[0],
                      numPartics,
                      numDofPartic) #[numTrajs, numPartcls, dimPerParticl]
        relativeVecsPerParticle=(trajsPerParticl[:,:,:,cnp.newaxis]-trajsPerParticl[:,cnp.newaxis,:,:].swapaxes(-2,-1)).swapaxes(-2,-1)
        # [numTrajs, numPartcls, numPartcls, dimPerParticl]
        # (k,i,j) is x^part_i_xik-x^part_j_xik vector [dimPerPartcl]
        relative_distances = cnp.linalg.norm(relativeVecsPerParticle, axis=-1, keepdims=True)
        # [numTrajs, numPartcls, numPartcls, 1]
        # to avoid self-interaction, we make the relative distance with themselves infinite
        relative_distances[:,cnp.arange(numPartics), cnp.arange(numPartics)]=cnp.inf
        if cnp.min(relative_distances)==0: # should not happen but...
            relative_distances=cnp.where(relative_distances, relative_distances, cnp.inf)
        # Compute the forces of all particles with all the rest
        all_forces = K*relativeVecsPerParticle/relative_distances**(2+1)
        all_forces = all_forces*qs[cnp.newaxis,:,cnp.newaxis,cnp.newaxis]*qs[cnp.newaxis,cnp.newaxis,:,cnp.newaxis]
        # [numTrajs, numPartcls, numPartcls, dimPerParticl]
        # +1 to make them unit vectors from i-th to j-th
        # (k,i,j) is the force at Universe k from particle j on i-th
        # we sum all the inter-particle forces on i-th particle of Universe k
        total=all_forces.sum(axis=2) # sum over j-> # [numTrajs, numPartcls, dimPerParticl]
        return total.reshape(total.shape[0],-1) #[numTrajs, dofPerUniverse]

    coulomb = lambda trajs : coulomb_force(trajs,
                                    qs=qs, numDofPartic=numDofPartic, K=K_coulomb)
    if use_Coulomb_potential:
        inter_particle_forces_within_universe = [
            coulomb
        ] # classical forces
    else:
        inter_particle_forces_within_universe = []


    # Inter-Universe Forces ########################
    def inter_universe_force(trajs, A, K):
        relative_vecs = (trajs[:,:,None]-trajs[:,:,None].T).swapaxes(1,2)
        # relative_vecs is [numTrajs, numTrajs, dofUniv] s.t. (i,j) is the x_xi1-x_xi2 vec
        # that is, the vector from i-th to j-th
        relative_distances = cnp.linalg.norm(relative_vecs, axis=-1, keepdims=True) #[numTrajs, numTrajs, 1]
        # to avoid self-interaction, we make the relative distance with themselves infinite
        relative_distances[cnp.arange(len(trajs)), cnp.arange(len(trajs))]=cnp.inf
        if cnp.min(relative_distances)==0: # should not happen but...
            relative_distances=cnp.where(relative_distances, relative_distances, cnp.inf)
        all_forces=A*relative_vecs/relative_distances**(K+1) # [numTrajs, numTrajs, dofUniv]
        # +1 to make them unit vectors from i-th to j-th
        # (i,j) is the force of trajectory j on i-th
        # we sum all the inter-universe forces on i-th
        return all_forces.sum(axis=1) # sum over j
    A = A/numTrajs
    quantum_force = lambda trajs : inter_universe_force(trajs, A=A, K=K)

    inter_universe_forces = [
            quantum_force
        ] # origin of quantum

    # Simulator run ##############################
    # For the Verlet algorithm, saving at least 3 time iterations is necessary
    positions = cnp.zeros(( 3, trajs.shape[0], trajs.shape[1]//2), dtype=real_dtype)
    #[3 (current, -1,-2), numTrajs, dof_Univ]

    # for the first time iteration we will estimate the previous position of the
    # particle using a simple Euler rule given the initial velocities
    positions[:] = trajs[:, :numDofUniv]-dt*trajs[:, numDofUniv:]/ms
    positions[0,:,:] = trajs[:, :numDofUniv] # newest time in position 0, then 1, then 2

    # shorthand
    dt2_masses = dt**2/(ms)

    # a Nx3 array (matrix) where we will save the forces in each time
    forces = cnp.zeros(positions.shape, dtype=real_dtype)
    # Output Directories
    os.makedirs(f"{outputs_directory}/MIW/{ID_string}/pdf/", exist_ok=True)
    os.makedirs(f"{outputs_directory}/MIW/{ID_string}/trajs/", exist_ok=True)

    # Run Iterations
    for it, t in enumerate(ts):
        # Step 1, compute the total force on each dof
        for force in inter_universe_forces:
            forces = force(trajs[:,:numDofUniv])
        for force in scenario_forces:
            forces += force(trajs[:,:numDofUniv])
        for force in inter_particle_forces_within_universe:
            forces += force(trajs[:,:numDofUniv])

        # OUTPUT of current state
        if it%outputEvery==0:
            #print(f"\n > It {it}/{numIts}")
            #print("Average Abs value Force ",np.abs(forces).mean())
            #trajs_numpy = cnp.asnumpy(trajs)
            trajs_numpy = trajs
            np.save(f"{outputs_directory}/MIW/{ID_string}/trajs/trajs_it_{it}.npy",
                    trajs_numpy, allow_pickle=True) #[numTrajs, numDofUniv]
            if use_external_forces_and_pdf_grid: # then we assume config space grid is available
                # Estimate pdf
                pdf = estimate_pdf_from_trajs(trajs[:,:numDofUniv], grid_pdf, sigma=sigma, real_dtype=real_dtype, norm=1/dxs_pdf.prod())
                #pdf = cnp.asnumpy(pdf)
                np.save(f"{outputs_directory}/MIW/{ID_string}/pdf/pdf_it_{it}.npy",
                    pdf, allow_pickle=True) #[Nx,Ny,Nz]

        # Step 2, compute the position of the particles in the next time
        # first move all the positions one slot onward in time
        positions[1:, :, :] = positions[:-1,:,:]
        # copy in the slots from 1 to 3 the ones that were in 0 to 2

        positions[0,:,:] = 2*positions[1,:,:] - positions[2,:,:] + forces*dt2_masses
        # Copy results in trajectory array
        trajs[:, :numDofUniv] = positions[0,:,:]

        # Step 3, compute the momentum in the next time for plotting purposes
        trajs[:, numDofUniv:] = trajs[:,numDofUniv:]+forces*dt

        # Those trajectories that get out of bounds should bounce back by the amount they got out
        patience = 2 # max three bounces allowed
        # Those trajectories that get out of bounds should bounce back by the amount they got out
        while(cnp.any(trajs[:,:numDofUniv]>=xuppers) or cnp.any(trajs[:,:numDofUniv]<xlowers)):
            trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]>=xuppers, xuppers-(trajs[:,:numDofUniv]-xuppers)-1e-10 ,trajs[:,:numDofUniv] )
            trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]<xlowers, xlowers+(xlowers-trajs[:,:numDofUniv]) ,trajs[:,:numDofUniv] )
            patience-=1
            if patience==0:
                trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]>=xuppers, xuppers-1e-10,trajs[:,:numDofUniv] )
                trajs[:,:numDofUniv] = cnp.where( trajs[:,:numDofUniv]< xlowers, xlowers,trajs[:,:numDofUniv] )
                break

    # Generate Santiy check frames jic
    # Generate max 5 frames as a sanity check, equispaced ##################
    number_of_outputs = int(numIts//outputEvery)
    n_frames = min(5, number_of_outputs)
    skip=int(number_of_outputs//n_frames)

    image_paths = []
    dpi=100
    # Each particle will have its own marker type in the physical space
    marker_per_particle = ['o', '^', 's', 'p', 'd', '*']
    if numDofPartic==3 and numDofUniv<4:
        every=1 # Only take one data point every this number in each axis to plot
        grid=np.array(np.meshgrid(*nodes_pdf)).swapaxes(1,2)[:,::every, ::every, ::every] #[3,Nx,Ny,Nz]
        #print(f"Using a mesh in the plot of {grid.shape}")
        fig = plt.figure( figsize=(14,7))
        #plt.style.use('dark_background')

        for it, t in zip( np.arange(len(ts))[::outputEvery][::skip], ts[::outputEvery][::skip]):
            #print(f"\n > It {it}/{numIts}")
            fig.clf()
            pdf = np.load(f"{outputs_directory}/MIW/{ID_string}/pdf/pdf_it_{it}.npy")
            trajs = np.load(f"{outputs_directory}/MIW/{ID_string}/trajs/trajs_it_{it}.npy")

            ax = fig.add_subplot(121, projection='3d')
            # PDF TRANSP ############################################################
            maxim = pdf.max()
            minim = pdf.min()
            level_surface = grid[:, pdf>maxim/3]
            colormap = ax.scatter(*level_surface, c=pdf[pdf>maxim/3],
                    cmap='hot_r', s=2, alpha=pdf[pdf>maxim/3]/maxim ) #, antialiased=True)
            ax.set_xlim((xlowers[0,0], xuppers[0,1]))
            ax.set_ylim((xlowers[0,1], xuppers[0,1]))
            ax.set_zlim((xlowers[0,2], xuppers[0,2]))

            fig.colorbar(colormap, fraction=0.04, location='left')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Probability Density>{maxim/3:.3}\n it={it} t={t:.4}")

            # PDF + TRAJECTORIES ##############################################
            ax = fig.add_subplot(122, projection='3d')
            #colormap = ax.scatter3D(*grid, c=pdf[ ::every, ::every, ::every],
            #               cmap='hot_r', s=0.1, alpha=0.4 ) #, antialiased=True)
            cset = ax.contour(grid[0,:,:,0], grid[1,:,:,0], pdf.sum(axis=2), 7, zdir='z', offset=xlowers[0,2], cmap='hot_r')
            cset = ax.contour(grid[0,:,0,:], pdf.sum(axis=1), grid[2,:,0,:], 7, zdir='y', offset=xuppers[0,1], cmap='hot_r')
            cset = ax.contour(pdf.sum(axis=0), grid[1,0,:,:], grid[2,0,:,:], 7, zdir='x', offset=xlowers[0,0], cmap='hot_r')
            #fig.colorbar(cset, fraction=0.04, location='left')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Trajectories and Projected Estimated pdf Contours\nColor==Universe; Shape==Particle\nit={it} t={t:.4}")

            # The color will indicate Universe and marker will indicate particle
            for particle in range(numDofUniv//numDofPartic):
                ax.scatter(trajs[:,particle*numDofPartic],
                        trajs[:,particle*numDofPartic+1],
                        trajs[:,particle*numDofPartic+2],
                       c='black', marker=marker_per_particle[particle],
                       s=3, alpha=1)
            fig.suptitle(f"InterUniverse K={K:.3} A={A:.3}\nNumber of Universes={numTrajs}")
            image=f"{outputs_directory}/MIW/{ID_string}/figs/it_{it}.png"
            plt.savefig(image, dpi=dpi)
            image_paths.append(image)

    if numDofPartic==2 and numDofUniv<3:
        every=1 # Only take one data point every this number in each axis to plot
        grid=np.array(np.meshgrid(*nodes_pdf)).swapaxes(-2,-1)[:,::every, ::every] #[2,Nx::ev,Ny]
        #print(f"Using a mesh in the plot of {grid.shape}")
        fig = plt.figure( figsize=(14,7))

        for it, t in zip( np.arange(len(ts))[::outputEvery][::skip], ts[::outputEvery][::skip]):
            ##print(f"\n > It {it}/{numIts}")
            fig.clf()
            pdf = np.load(f"{outputs_directory}/MIW/{ID_string}/pdf/pdf_it_{it}.npy")
            trajs = np.load(f"{outputs_directory}/MIW/{ID_string}/trajs/trajs_it_{it}.npy")
            ax = fig.add_subplot(121, projection='3d')
            maxim = pdf.max()
            minim = pdf.min()
            # PDF ############################################################
            colormap = ax.plot_surface(grid[0], grid[1], pdf, rcount=50, ccount=50,
                    cmap='RdYlBu')
            fig.colorbar(colormap, fraction=0.04, location='left')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("Probability Density")
            ax.set_title(f"Probability Density it={it} t={t:.4}")
            cset = ax.contour(grid[0], grid[1], pdf, 7, zdir='x',
                              offset=xlowers[0,0], cmap='RdYlBu', vmax=maxim, vmin=minim)
            cset = ax.contour(grid[0], grid[1], pdf, 7, zdir='y',
                              offset=xuppers[0,1], cmap='RdYlBu', vmax=maxim, vmin=minim)

            # PDF + TRAJECTORIES ##############################################
            ax = fig.add_subplot(122)
            colormap = ax.imshow(pdf.T,
                 extent=[xlowers[0,0], xuppers[0,0], xlowers[0,1], xuppers[0,1]]
                                 , origin='lower',cmap='RdYlBu')
            #plt.axis(aspect='image');
            fig.colorbar(colormap, fraction=0.04, location='right')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Trajectories and Probability Density\n it={it} t={t:.4}")
            # The color will indicate Universe and marker will indicate particle
            for particle in range(numDofUniv//numDofPartic):
                ax.scatter(trajs[:,particle*numDofPartic],
                        trajs[:,particle*numDofPartic+1],
                       c='black', marker=marker_per_particle[particle],
                       s=3, alpha=0.8)
            fig.suptitle(f"InterUniverse K={K:.3} A={A:.3}\nNumber of Universes={numTrajs}")
            image=f"{outputs_directory}/MIW/{ID_string}/figs/it_{it}.png"
            plt.savefig(image, dpi=dpi)
            image_paths.append(image)


    if numDofPartic==1 and numDofUniv<3:
        os.makedirs(f"{outputs_directory}/MIW/{ID_string}/figs/", exist_ok=True)
        image_paths = []
        dpi = 100
        fig = plt.figure( figsize=(14,7))

        for it, t in enumerate(ts):
            if it%outputEvery==0:
                ##print(f"\n > It {it}/{numIts}")

                fig.clf()
                pdf = np.load(f"{outputs_directory}/MIW/{ID_string}/pdf/pdf_it_{it}.npy")
                trajs = np.load(f"{outputs_directory}/MIW/{ID_string}/trajs/trajs_it_{it}.npy")

                # PDF + POTENTIAL + TRAJECTORIES ##############################################
                ax = fig.add_subplot(111)
                ax.plot(grid_pdf[:,0], pdf, label="Probablity Density")
                ax.set_xlabel("x")
                ax.set_ylabel("pdf")
                ax.set_title(f"Trajectories and Probability Density\n it={it} t={t:.4}")
                plt.legend()
                ax2=ax.twinx()
                ax2.plot(grid_potential[:,0], potential_field, label="Potential", color=u'#ff7f0e')
                ax2.plot(trajs[:,0], [np.mean(potential_field)]*numTrajs, 'o',color='black',
                        markersize=1,label="Bohmian Trajectories")
                ax2.set_ylabel("Potential")
                image=f"{outputs_directory}/MIW/{ID_string}/figs/it_{it}.png"
                plt.legend()
                plt.savefig(image, dpi=dpi)
                image_paths.append(image)
    # Generate gif
    import imageio
    fps=7
    images_for_animation = [ imageio.v2.imread(image_path) for image_path in image_paths]
    imageio.mimsave(f"{outputs_directory}/MIW/{ID_string}/MIW_Verlet.gif", images_for_animation, duration=fps**-1*len(images_for_animation))

    # Free Space ######
    if use_cuda_numpy:
        cnp.get_default_memory_pool().free_all_blocks()
        cnp.get_default_pinned_memory_pool().free_all_blocks()
