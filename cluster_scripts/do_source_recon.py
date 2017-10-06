import sys
sys.path.append('/home/nwilming/')
import glob
from pymeg import tfr
from conf_analysis.meg import source_recon as sr, preprocessing, dics


def make_filters(subjid):
    print('Starting task:', subjid)
    params = tfr.params_from_json(
        '/home/nwilming/conf_analysis/required/all_tfr150_parameters.json')

    epochs, meta = preprocessing.get_epochs_for_subject(subjid, 'stimulus')
    epochs.times = epochs.times - 0.75
    epochs = epochs.apply_baseline((-0.25, 0))

    tfrepochs = dics.get_tfr(subjid, n_blocks=1)
    F, t_smooth, f_smooth = tfr.get_smoothing(60, **params)
    filters = dics.make_dics_filter(epochs, tfrepochs.freqs, F,
                                    tfrepochs.times, f_smooth,
                                    t_smooth, subjid,
                                    n_jobs=12)
    import cPickle
    filename = '/home/nwilming/conf_meg/S%i-%3.2iHz-dics-filters.pickle' % (
        subjid, F)
    cPickle.dump(
        filters, open(filename, 'w'))


def apply_filter(subjid):
    params = tfr.params_from_json(
        '/home/nwilming/conf_analysis/required/all_tfr150_parameters.json')
    F, t_smooth, f_smooth = tfr.get_smoothing(60, **params)
    tfrepochs = dics.get_tfr(subjid)
    import cPickle
    filters = cPickle.load(
        open('/home/nwilming/conf_meg/S%i-%3.2iHz-dics-filters.pickle' %
             (subjid, F)))
    filename = '/home/nwilming/conf_meg/S%i-%3.2iHz-power.memmap' % (subjid, F)
    power = dics.apply_dics_filter(filters, F, tfrepochs, filename, n_jobs=12)


def execute(x):
    subjid, task = x
    if (task is None) or (task == 'make'):
        make_filters(subjid)
    if (task is None) or (task == 'apply'):
        apply_filter(subjid)


def list_tasks(older_than='now', filter=None):
    for f in range(1, 2):
        print f, filter
        yield f, filter


if __name__ == '__main__':
    for x in list_tasks():
        print x
