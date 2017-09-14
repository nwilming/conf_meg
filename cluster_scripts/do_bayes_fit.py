import sys
sys.path.append('/home/nwilming/')
from conf_analysis.behavior import individual_sample_model as ism


def execute(x):
    print('Starting task:', x)
    snum, var = x
    d = ism.get_data()
    d = d.query('snum==%i' % snum)
    ism.get_all_subs(d, models=[var])


def list_tasks(older_than='now'):
    for snum in range(1, 16):
        for var in [True, False]:
            yield (snum, var)
