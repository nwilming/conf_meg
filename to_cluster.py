'''
Run a simple script on one cluster node!
'''
import argparse
import os


# 2. Create workers
parser = argparse.ArgumentParser()
parser.add_argument("script", help='Which python script?')
parser.add_argument("--cores", help="How many cores do you need?",
                    type=int, default=1)
parser.add_argument("--memory", help="How many GB ram?", default=12, type=int)
parser.add_argument("--walltime", help="Walltime (h)?", default=15, type=int)

args = parser.parse_args()


command = '''
#!/bin/sh
# walltime: defines maximum lifetime of a job
# nodes/ppn: how many nodes (usually 1)? how many cores?

#PBS -q batch
#PBS -l walltime=%i:00:00
#PBS -l nodes=1:ppn=%s
#PBS -l mem=%igb
# -- run in the current working (submission) directory --
cd %s

# FILE TO EXECUTE
python %s 1> cluster/$PBS_JOBID.out 2> cluster/$PBS_JOBID.err
'''%(args.walltime, args.cores, args.memory, os.getcwd(), args.script)

print command
tmp = file('to_cluster.sh', 'w')
tmp.write(command)

os.system('qsub to_cluster.sh')
