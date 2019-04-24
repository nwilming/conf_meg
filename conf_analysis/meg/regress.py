'''
Evaluate a linear regression model per subject and average across subjects
'''



def compute_regression_model(subject, epoch, area, hemi, model):
    from pymeg import aggregate_sr as asr
    from conf_analysis.meg import preprocessing
    import patsy

    filenames = '/home/nwilming/conf_meg/sr_labeled/aggs/S%i_*%s*.hdf'
    agg = hdf2agg(filenames, hemi=hemi, cluster=area, freq=None)
    meta = None # Get metadata

    # Make design matrix
    y, X = patsy.dmatrices('hash~%s'%model)
    # Each trial and time point becomes one target
