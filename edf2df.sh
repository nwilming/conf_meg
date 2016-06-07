#!/bin/sh
# embedded options to qsub - start with #PBS
# walltime: defines maximum lifetime of a job
# nodes/ppn: how many nodes (usually 1)? how many cores?

  #PBS -q batch
  #PBS -l walltime=6:00:00
  #PBS -l nodes=1:ppn=1
  #PBS -l mem=1gb
  #PBS -o jobs/$PBS_JOBID.o
  #PBS -e jobs/$PBS_JOBID.e

# -- run in the current working (submission) directory --
cd $PBS_O_WORKDIR

chmod g=wx $PBS_JOBNAME

# FILE TO EXECUTE
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nwilming/EyeLinkDisplaySoftware/lib/

python load_edfs.py $PBS_ARRAYID 1,2>jobs/$PBS_JOBID.o
#python load_edfs.py 8 1,2>jobs/$PBS_JOBID.o

