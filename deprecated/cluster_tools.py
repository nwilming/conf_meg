import numpy as np
import subprocess

def get_info():
    info = {}
    p = subprocess.Popen('qnodes', stdout=subprocess.PIPE, shell=True)
    cnode = None
    for line in iter(p.stdout.readline, ''):
        if line.startswith('node'):
            cnode = line.strip()
            info[cnode] = {}
        if 'status=down' in line:
            del info[cnode]
        if 'physmem' in line:
            for part in line.split(','):
                if 'physmem' in part or 'ncpus' in part:
                    begin, end = part.split('=')
                    if end.endswith('kb'):
                        end = round(int(end[:-2])/1024./1024.,2)
                    else:
                        end = int(end)
                    info[cnode][begin] = end
    return info


def get_num_nodes(node, info, gb):
    mpw = info[node]['physmem']/float(info[node]['ncpus'])
    nodes_per_task = np.ceil(gb/mpw)
    return int(nodes_per_task)

def get_worker_templates(walltime, memory, cwd, ip, port):
    # build memory string
    info = get_info()
    commands = []
    for node in list(info.keys()):
        try:
            npw = get_num_nodes(node, info, memory)
            mem_str = '%s:ppn=%i'%(node, npw)
            max_jobs = info[node]['ncpus']/npw
            print(max_jobs)
            command = '''
    #!/bin/sh
    # walltime: defines maximum lifetime of a job
    # nodes/ppn: how many nodes (usually 1)? how many cores?

       #PBS -q batch
       #PBS -l walltime=%i:00:00
       #PBS -l nodes=%s
       #PBS -l mem=%igb
    # -- run in the current working (submission) directory --
    cd %s

    chmod g=wx $PBS_JOBNAME

    export PYTHONPATH=/home/nwilming/

    # FILE TO EXECUTE
    dask-worker --nthreads=1 %s:%i 1> cluster/$PBS_JOBID.out 2> cluster/$PBS_JOBID.err
            '''%(walltime, mem_str, memory, cwd, ip, port)
            commands.append((max_jobs, command))
        except KeyError:
            pass
    return commands

