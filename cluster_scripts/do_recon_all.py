import sys
sys.path.append('/home/nwilming/')
import glob


def execute(x):
    subjid, filename = x
    print('Starting task:', x)
    from subprocess import call
    cmd = 'recon-all -subjid S%02i -all'%(subjid)
    print cmd
    call(cmd, shell=True)


def list_tasks(older_than='now', filter=None):
    for f in [13]: #range(1,16):
    	#subjid = f.split('/')[-1].replace('.nii','')
    	yield (f, None)


if __name__ == '__main__':
	for x in list_tasks():
		print x
