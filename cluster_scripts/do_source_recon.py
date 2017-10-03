    import sys
    sys.path.append('/home/nwilming/')
    import glob


def execute(x):
    subjid = x
    print('Starting task:', x)
    from conf_analysis.meg import source_recon as sr, preprocessing, dics
    from pymeg import tfr

    params = tfr.params_from_json(
        '/home/nwilming/conf_analysis/required/all_tfr150_parameters.json')

    epochs, meta = preprocessing.get_epochs_for_subject(x, 'stimulus')
    epochs.times = epochs.times - 0.75
    epochs = epochs.apply_baseline((-0.25, 0))
    
    tfrepochs = dics.get_tfr(x)
    F, t_smooth, f_smooth = tfr.get_smoothing(60, **params)

    forward, bem, source, trans, source_pow = dics.dics(epochs,
                                                        tfrepochs,
                                                        F,
                                                        tfrepochs.times,
                                                        f_smooth,
                                                        t_smooth,
                                                        x)
    import cPickle
    import h5py
    fname = '/home/nwilming/conf_data/S%02i-src-power-params.pickle' % x
    cPickle.dump({'forward': forward, 'bem': bem, 'source': source,
                  'trans': trans}, open(fname, 'w'))
    fname = '/home/nwilming/conf_data/S%02i-src-power-%fHx-params.hdf5' % (
        x, F)
    hdf5 = h5py.File(fname, 'w')
    dset = hdf5.create_dataset('source_pow', data=source_pow)
    hdf5.close()


def list_tasks(older_than='now', filter=None):
    for f in range(1, 2):
        yield f


if __name__ == '__main__':
    for x in list_tasks():
        print x
