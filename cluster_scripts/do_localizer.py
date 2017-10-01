import sys
sys.path.append('/home/nwilming/')
import pickle
import mne, locale
locale.setlocale(locale.LC_ALL, "en_US")


def list_tasks(older_than='now', filter=None):
    x = range(1, 16)
    for sub in x:
        yield sub


def execute(x):
    import matplotlib
    matplotlib.use('agg')
    from conf_analysis.meg import localizer
    print('Starting task:', x)
    localizer.select_by_gamma(x)
    print('Ended task:', x)
