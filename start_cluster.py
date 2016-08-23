import subprocess, os, argparse
import cluster_tools
import numpy as np
'''
Start a cluster of dask python instances.
'''

# 1. Start dask scheduler
from tornado.ioloop import IOLoop
loop = IOLoop()

from threading import Thread
t = Thread(target=loop.start)
t.daemon=True
t.start()

from distributed import Scheduler
s = Scheduler(loop=loop)
port = 8786
s.start(port)
print 'Started scheduler at >>> %s:%i'%(s.ip, port)


# 2. Create workers
parser = argparse.ArgumentParser()
parser.add_argument("num_workers", help="How many workers do you need?",
                    type=int)
parser.add_argument("--memory", help="How many GB ram?", default=4, type=int)
parser.add_argument("--walltime", help="Walltime (h)?", default=5, type=int)

args = parser.parse_args()

cwd = os.getcwd()

commands = cluster_tools.get_worker_templates(args.walltime, args.memory, cwd, s.ip, port)
np.random.shuffle(commands)
print commands[0][1]

num_workers = 0
torque_ids = []

for max_jobs, command in commands:
    with open('cluster_worker.sh', 'w') as cw:
        cw.write(command)

    print 'Submitting %i worker to TORQUE.'%max_jobs,
    r = subprocess.Popen('qsub -N nwilming_cluster -t 0-%i cluster_worker.sh'%max_jobs, 
            shell=True, stdout=subprocess.PIPE)
    torque_id = r.stdout.readline().split('.')[0]
    torque_ids.append(torque_id)
    print 'Workers submitted. QSUB ID is' , torque_id
    num_workers += max_jobs
    if max_jobs >= args.num_workers:
        break


import time
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print 'Cancelling all workers:', 
    for torque_id in torque_ids:
        subprocess.Popen('qdel %s'%torque_id, shell=True)
        print torque_id
    print '' 
    time.sleep(10)
    import sys
    sys.exit()
