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


print 'Started scheduler at >>> %s:%i'%(s.ip, port)

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
   #PBS -o cluster/$PBS_JOBID.o
   #PBS -e cluster/$PBS_JOBID.e

# -- run in the current working (submission) directory --
cd %s

chmod g=wx $PBS_JOBNAME

export PYTHONPATH=/home/nwilming/

# FILE TO EXECUTE
dask-worker %s:%i 1> cluster/$PBS_JOBID.out 2> cluster/$PBS_JOBID.err
'''%(args.walltime, args.memory, cwd, s.ip, port)

print command 

with open('cluster_worker.sh', 'w') as cw:
    cw.write(command)

print 'Submitting %i worker to TOEQUE'%args.num_workers
r = subprocess.Popen('qsub -N nwilming_cluster -t 0-%i cluster_worker.sh'%args.num_workers, shell=True, stdout=subprocess.PIPE)
torque_id = r.stdout.readline().split('.')[0]
print 'Workers submitted. QSUB ID is' , torque_id

import time
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print 'Cancelling all workers'
    subprocess.Popen('qdel %s'%torque_id, shell=True)
    time.sleep(5)
    import sys
    sys.exit()
