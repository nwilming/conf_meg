'''
Start a cluster of dask python instances.
'''

import subprocess
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

# Now the scheduler is running.
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("num_workers", help="How many workers do you need?",
                    type=int)
parser.add_argument("--memory", help="How many GB ram?", default=4, type=int)
parser.add_argument("--walltime", help="Walltime (h)?", default=5, type=int)

args = parser.parse_args()

cwd = os.getcwd()
command = '''
#!/bin/sh
# walltime: defines maximum lifetime of a job
# nodes/ppn: how many nodes (usually 1)? how many cores?

   #PBS -q batch
   #PBS -l walltime=%i:00:00
   #PBS -l nodes=1:ppn=1
   #PBS -l mem=%igb

# -- run in the current working (submission) directory --
cd %s

chmod g=wx $PBS_JOBNAME

# FILE TO EXECUTE
dask-worker %s:%i
'''%(args.walltime, args.memory, cwd, s.ip, port)

with open('cluster_worker.sh', 'w') as cw:
    cw.write(command)

print 'Submitting %i worker to TOEQUE'%args.num_workers
subprocess.system('qsub -N nwilming_cluster -t 0-%i cluster_worker.sh'%args.num_workers)

import time
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print 'Interrupt'
    import sys
    sys.exit()
