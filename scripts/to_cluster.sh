
    #!/bin/bash
    # walltime: defines maximum lifetime of a job
    # nodes/ppn: how many nodes (usually 1)? how many cores?

    #PBS -q batch
    #PBS -l walltime=15:00:00
    #PBS -l nodes=1:ppn=1
    #PBS -l mem=20gb
    #PBS -N PMISS
    
    cd /mnt/homes/home024/nwilming/conf_analysis  
    mkdir -p cluster
    chmod a+rwx cluster
    
    #### set journal & error options
    #PBS -o /mnt/homes/home024/nwilming/conf_analysis/cluster/$PBS_JOBID.o
    #PBS -e /mnt/homes/home024/nwilming/conf_analysis/cluster/$PBS_JOBID.e


    # FILE TO EXECUTE
    python cluster_do_preprocessing.pySXNdBv/job_56.py 1> /mnt/homes/home024/nwilming/conf_analysis/cluster/$PBS_JOBID.out 2> /mnt/homes/home024/nwilming/conf_analysis/cluster/$PBS_JOBID.err
    