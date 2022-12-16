from tqdm import tqdm
from multiprocessing import Process


### Multiprocessing the subjects ###
procs = []
for i in tqdm(range(len(files))):
    p = Process(target=connectome_msmt_csd, args=(config, files[i], acronym))
    p.start()
    procs.append(p)

    while len(procs)%num_threads == 0 and len(procs) > 0:
        for p in procs:
            # wait for 10 seconds to wait process termination
            p.join(timeout=10)
            # when a process is done, remove it from processes queue
            if not p.is_alive():
                procs.remove(p)
                
    # Final chunk could be shorter than num_threads, so it's handled waiting for its completion 
    #       (join without arguments wait for the end of the process)
    if i == len(files) - 1:
        for p in procs:
            p.join()