import os
import multiprocessing as mp
import time
import sys


if __name__ == "__main__":
    # SIMULATION PARAMETERS ##################################
    args = sys.argv # The path to the settings file should be given
    print("Input arguments", sys.argv)
    assert len(args)==3, "Simulation queue file path required and number of parallel coordinators"
    queue_file = str(args[1])
    num_coordinators = int(args[2])
    arguments_for_coordinators = []
    with open(queue_file, 'r') as f:
        tasks = f.readlines()
    for task in tasks:
        arguments_for_coordinators.append(
            task.replace(" ", "").split('(')[-1].split(')')[0].split(',') )

    def run_coordinator_and_return(arguments):
        print("barrutik", arguments[0], arguments[1], arguments[2], arguments[3])
        os.system(
            "python COORDINATOR_Simulation_Pipeline.py "+arguments[0]+" "+arguments[1]+" "+\
                arguments[2] + " "+ arguments[3]+" "+arguments[4]
            )
        return arguments[0] #ID
    run_coordinator_and_return(arguments_for_coordinators[0])
    # We initiate a pool of num_workers processes-> Pythoin does not allow pools to generate children pools!!! So we have to do this with processes
    #run_coordinator_and_return(arguments_for_coordinators[3])
    wait= 180 # number of seconds to wait till re-check if ready
    t=0
    processes = []
    for arguments_for_coordinatork in arguments_for_coordinators:
        while len(processes)>=num_coordinators:
            did=[]
            for k,process in enumerate(processes):
                if not process.is_alive():
                    process.join()
                    did.append(k)
                    print(f">>>> t={t}s Done one!")
            for d in did:
                processes.pop(d)
            t+=wait
            time.sleep(wait)
        processes.append( mp.Process(target=run_coordinator_and_return,
                args=(arguments_for_coordinatork,)) )
        processes[-1].start()

    '''
    pool = Pool(num_coordinators)
    result_tickets = []
    IDs=[]
    for arguments_for_coordinatork in arguments_for_coordinators:
        #print(arguments_for_worker)
        result_tickets.append( pool.apply_async( run_coordinator_and_return, arguments_for_coordinatork ) )
        IDs.append(arguments_for_coordinatork[0])

    wait= 180 # number of seconds to wait till re-check if ready
    t=0
    done=[]
    while len(arguments_for_coordinators)!=len(done):
        for result_ticket, ID in zip(result_tickets, IDs):
            if ID not in done and result_ticket.ready():
                print(f">>>> t={t}s Done ID {ID}!")
                done.append( ID )
        #print(f"t={t}s")
        #print('done',done, 'todo',to_do)
        # wait a little more (3 mins)
        t+=wait
        time.sleep(wait)

    for result_ticket in result_tickets:
        result_ticket.wait()
    pool.close()
    pool.join()
    '''
    # we remove the state file now that it has finished and place a done flag
    with open(f".//FINISHED.txt", 'w') as f:
        f.write(f"{len(arguments_for_coordinators)} coordinators called and done in t={t} s={t/60:.3} min={t/3600:.3} h\n")
