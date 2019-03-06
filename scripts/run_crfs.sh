#!/usr/bin/env python
"""
Compile and run a stan program with cmdstan
"""
import argparse
import os
from os.path import join
from subprocess import run


def expand(filename):     
    return os.path.normpath(os.path.join(os.environ['PWD'], filename))

if __name__ == "__main__":
    home = os.environ["HOME"]
    standir = join(home, "cmdstan-2.18.1")

    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Which stan file to run?")
    parser.add_argument("--data", help="Which data file?")
    parser.add_argument("--out", help="Where to save samples?")
    parser.add_argument(
        "--refresh", help="Refresh each ith iter.", default=50, type=int
    )
    parser.add_argument("--init", help="Initfile to use", default="none")
    args = parser.parse_args()

    args.script = expand(args.script)
    print('Running stan program:', args.script)

    args.data = expand(args.data)
    print('Using datafile:', args.data)

    buildcmd = "cd %s; make %s" % (standir, args.script)
    #print(buildcmd)
    out = run(buildcmd, shell=True)
    print(out.returncode)
    if not (out.returncode == 0):
        raise RuntimeError('Buld did not complete')

    #print(out)    
    samplecmd = "screen {command} sample data file={data} output file={output} refresh={refresh}".format(
        data=args.data, output=args.out, refresh=args.refresh, command=args.script
    )
    print(samplecmd)
    if args.init is not "none":
        args.init = expand(args.init)
        print('Using initfile:', args.init)
        samplecmd += " init={init}".format(init=args.init)

    print(samplecmd)
    run(samplecmd, shell=True)

    # home=/home/student/n/nwilming
    # cd /home/student/n/nwilming/cmdstan-2.18.1
    # make /home/student/n/nwilming/conf_analysis/crfs_hierarchical_nosmp

    # cd $home

    # $home/conf_analysis/crfs_hierarchical_nosmp sample\
    #    data file=$home/fake_vfc.rdump\
    #    output file=/net/store/users/nwilming/fake_all_subs_out_nomu.csv\
    #    refresh=5\


# init=$home/fake_all_vfc_init_nom.rdump
"""

    parser.add_argument("--nodes", help="Specifiy a list of nodes or the number of cores in PBS syntax",
                        default='1:ppn=1', type=str)
    parser.add_argument("--array", help="Map scripts function to workers?",
                        default=False, action="store_true")
    parser.add_argument("--filter", dest='filter',
                        help="Only run an array job in filter is in job parameters",
                        default=None)
    parser.add_argument("--nosubmit", dest='submit',
                        help="Do not submit to torque", default=True, action="store_false")

    parser.add_argument(
        "--name", help="Give a name to this job. Defaults to executed command.")
    parser.add_argument("--redo-older", dest='older',
                        help="Redo all tasks where results are older than this Ymd date (e.g. 20160825). Only effective for array jobs.",
                        type=str, default='now')
    parser.add_argument('-D', action='append', default=[], 
        help="Extra arguments to be passed to the executing function. E.g. -Dfoo=10 means foo=10")
    parser.add_argument("--pyenv", default='none',
                        help="Activate a different conda environment",
                        type=str, dest='env')
"""
