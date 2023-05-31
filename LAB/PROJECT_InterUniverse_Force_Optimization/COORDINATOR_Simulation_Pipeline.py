import numpy as np
import os
import multiprocessing as mp
import time
import sys

if __name__ == "__main__":
    # SIMULATION PARAMETERS ##################################
    args = sys.argv # The path to the settings file should be given
    print("Input arguments", sys.argv)
    assert len(args)==6, "Paths or experiment name not given correctly!"
    exp_name = args[1]
    path_to_settings = args[2]
    path_to_psi_and_potential=args[3]
    path_to_grid = args[4]
    num_workers = int(args[5])
    ''' Expected arguments:
    - Experiment name
    - path to the settings file for the simulation
    - name of the .py file defining the psi0(x,y) and potential(x,y) functions
    - path to the file with the simulation grid parameters
    - number of workers to use in parallel maximum
    '''
    with open(f"{path_to_grid}", "r") as f:
        numK = int(f.readline().split("numK ")[1])
        numA = int(f.readline().split("numA ")[1])
        Krange = [float(x) for x in f.readline().split("Krange ")[1].split('[')[1].split(']')[0].split(',')]
        Arange = [float(x) for x in f.readline().split("Arange ")[1].split('[')[1].split(']')[0].split(',')]
        dim = int(f.readline().split("dim ")[1])
    Ks_to_try = np.linspace(Krange[0], Krange[1], numK)
    As_to_try = np.linspace(Arange[0], Arange[1], numA)
    # Crear el arbol de directorios putser es interesante primero, luego
    # ein fitxero bat de estado nun apunte eingo deuen zeintzuk ya egin diezen
    # existitzen bazan leidu bakarrik
    # ta karpeta bat que sea de flags en plan files que van generando los chicos a la que acaban
    os.makedirs(f"./OUTPUTS/{exp_name}/", exist_ok=True)
    if not os.path.isfile(f"./OUTPUTS/{exp_name}/STATE.txt"):
        with open(f"./OUTPUTS/{exp_name}/STATE.txt", 'w') as f:
            f.write("DoneSimulations []")
        done = []
    else:
        with open(f"./OUTPUTS/{exp_name}/STATE.txt", 'r') as f:
            done = [ID for ID in f.readline().split("DoneSimulations ")[1].split('[')[1].split(']')[0].split(',')]
    # generate the IDs of the processes to do
    to_do = []
    arguments_for_workers = []
    if not f"Reference_SE_Simulation" in done:
        to_do.append(f"Reference_SE_Simulation")
        arguments_for_workers.append(("Reference_SE_Simulation", f"SIMULATOR_{dim}D_CN_Schrodinger_Equation.py",
            f"Reference_SE_Simulation {path_to_settings} {path_to_psi_and_potential} {f'./OUTPUTS/{exp_name}/'}"))
        reference_traj_npy_file = f"./OUTPUTS/{exp_name}/SE_2D/Reference_SE_Simulation/trajs/trajs_it_0.npy"
    for K in Ks_to_try:
        for A in As_to_try:
            ID = f"K_{K:.4}_A_{A:.4}"
            if ID not in done:
                to_do.append(ID)
                arguments_for_workers.append((ID, f"SIMULATOR_MD_Interacting_Newtonian_Universes.py",
                f"{ID} {path_to_settings} {path_to_psi_and_potential} {f'./OUTPUTS/{exp_name}/'} {A} {K} {reference_traj_npy_file}"))

    def run_simulator_and_return_ID(ID, path_to_simulator_py, arguments ):
        out=os.system(f"python {path_to_simulator_py} {arguments}")
        return ID

    # We initiate a pool of num_workers processes
    pool = mp.Pool(num_workers)
    result_tickets = []
    for arguments_for_worker in arguments_for_workers:
        print(arguments_for_worker)
        result_tickets.append( pool.apply_async( run_simulator_and_return_ID, arguments_for_worker ) )

    while len(done)!=len(to_do):
        for result_ticket, ID in zip(result_tickets, to_do):
            if ID not in done and result_ticket.ready():
                print(f" Done ID {ID}!")
                done.append( ID )
                with open(f"./OUTPUTS/{exp_name}/STATE.txt", 'w') as f:
                    f.write(f"DoneSimulations {done}")
            print('done',done, 'todo',to_do)
        # wait a little more (3 mins)
        time.sleep(10)


    for result_ticket in result_tickets:
        result_ticket.wait()
    pool.close()
    pool.join()
    # we remove the state file now that it has finished
    os.remove(f"./OUTPUTS/{exp_name}/STATE.txt")
