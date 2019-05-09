
from conf_analysis.meg import decoding_analysis as da


def get_motor_prediction(subject, latency, cluster='JWG_M1'):
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(
                filenames, hemi="Lateralized", cluster=cluster
            )
    meta = da.augment_meta(da.preprocessing.get_meta_for_subject(subject, "stimulus"))        
    midc_decoder(
        meta, data, cluster, latency=0, splitmc=True, target_col="response", predict=True
    )        
    return scores


def submit_cross_area_decoding():
    from pymeg import parallel

    pmap = partial(
        parallel.pmap,
        email=None,
        tasks=1,
        nodes=1,
        memory=60,
        ssh_to=None,
        walltime="72:00:00",  # walltime='72:00:00',
        cluster="SLURM",
        env="py36",
    )

    for subject in range(1, 16):
        for odd in [True, False]:
            pmap(run_cross_area_decoding, [(subject, [odd,])], name="XA" + str(subject))


def run_cross_area_decoding(subject, odd_partition=[True, False], ntasks=15):
    set_n_threads(1)
    from multiprocessing import Pool
    from glob import glob
    from pymeg import atlas_glasser, aggregate_sr as asr
    import os
    from os.path import join
    from glob import glob

    low_level_areas = [
        "vfcPrimary",
        "vfcEarly",
        "vfcV3ab",
        "vfcIPS01",
        "vfcIPS23",
        "vfcLO",
        "vfcTO",
        "vfcVO",
        "vfcPHC",
        "JWG_IPS_PCeS",
        "vfcFEF",
        "HCPMMP1_dlpfc",
        "HCPMMP1_insular_front_opercular",
        "HCPMMP1_frontal_inferior",
        "HCPMMP1_premotor",
        "JWG_M1",
    ]
    high_level_areas = [
        "JWG_IPS_PCeS",
        "vfcFEF",
        "HCPMMP1_dlpfc",
        "HCPMMP1_premotor",
        "JWG_M1",
    ]
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))

    args = []
    filename = outpath + "/cross_area_S%i-decoding_%s.hdf" % (subject, str(odd_partition))
    print("Save target:", filename)
    cnt = 0
    add_meta = {"subject": subject, "low_hemi": "Averaged", "high_hemi": "Lateralized"}
    for lla, hla in product(low_level_areas, high_level_areas):
        for odd in odd_partition:
            low_level_data = asr.delayed_agg(filenames, hemi="Averaged", cluster=lla)
            high_level_data = asr.delayed_agg(
                filenames, hemi="Lateralized", cluster=hla
            )
            args.append(
                (
                    subject,
                    low_level_data,
                    lla,
                    high_level_data,
                    hla,
                    0.18,
                    "response",
                    "cross_dcd_S%i-"%subject,
                    "%s" % cnt,
                    add_meta,
                    odd,
                )
            )
            cnt += 1
    print("There are %i decoding tasks for subject %i" % (len(args), subject))
    scratch = os.environ["TMPDIR"]
    try:
        with Pool(ntasks, maxtasksperchild=1) as p:
            scores = p.starmap(motor_decoder, args)  # , chunksize=ntasks)
    finally:
        # Do this to find out why memory error occurs
        print('Saving data')
        try:
            cmd = join(scratch, "*.hdf")
            os.system("cp %s /nfs/nwilming/MEG/scratch/"%join(scratch, '*.hdf'))
        except Exception:
            pass

        # Now collect all data from scratch
        save_filenames = join(scratch, "cross_dcd" + "*.hdf")
        scores = [pd.read_hdf(x) for x in glob(save_filenames)]
        scores = pd.concat(scores)
        # scores.loc[:, 'subject'] = subject
        scores.to_hdf(filename, "decoding")
        return scores


def motor_decoder(
    subject,
    low_level_data,
    low_level_area,
    motor_data,
    motor_area,
    low_level_peak,
    target_col="response",
    save_filename=None,
    save_prefix=None,
    add_meta={},
    odd_times=None,
):
    """
    This signal predicts motor activity from a set of time points in early visual cortex.

    The idea is to train to decoders simultaneously:

    1) Train a motor decoder and predict log(probability) of a choice. Do this 
       on the training set.
    2) Train a decoder from early visual cortex data that predicts the 
       log(probability) of a choice. Train on training set, evaluate on test.
    """
    meta = augment_meta(preprocessing.get_meta_for_subject(subject, "stimulus"))
    meta = meta.set_index("hash")
    low_level_data = low_level_data()
    motor_data = motor_data()

    # Iterate over all high level time points:
    times = motor_data.columns.get_level_values("time").values
    dt = np.diff(times)[0]
    t_idx = (-0.4 < times) & (times < 0.1)

    cnt = 0
    time_points = times[t_idx]
    if odd_times is not None:
        if odd_times:
            time_points = time_points[:-1:2]
        else:
            time_points = time_points[1::2]
    for high_level_latency in time_points:
        print("High level latency:", high_level_latency)
        target_time_point = times[np.argmin(abs(times - high_level_latency))]
        md = prep_motor_data(motor_area, motor_data, target_time_point)

        # Buld target vector
        motor_target = (meta.loc[md.index, target_col]).astype(int)
        all_scores = []
        for low_level_latency in np.arange(-0.1, 0.2 + dt, dt):
            lld_data = prep_low_level_data(
                low_level_area, low_level_data, low_level_peak, low_level_latency
            )
            # low_level_meta = meta.loc[lld_data.index, :]
            assert all(lld_data.index == motor_target.index)
            scores = chained_categorize(motor_target, md, lld_data)
            scores.loc[:, "low_level_latency"] = low_level_latency
            all_scores.append(scores)
        save_all_scores(
            all_scores,
            add_meta,
            save_filename,
            save_prefix,
            cnt,
            high_level_latency,
            low_level_peak,
            low_level_area,
            motor_area,
        )
        cnt += 1
        del all_scores
        del md
        del motor_target
    del motor_data
    del low_level_data


def prep_motor_data(motor_area, motor_data, target_time_point):
    md = motor_data.loc[:, target_time_point]
    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(motor_area):
        x = pd.pivot_table(
            md.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        X.append(x)
    return pd.concat(X, 1)


def prep_low_level_data(
    low_level_area, low_level_data, low_level_peak, low_level_latency
):
    lld = []
    low_times = low_level_data.columns.get_level_values("time").values
    for s in np.arange(0, 1, 0.1) + low_level_peak + low_level_latency:
        # Turn data into (trial X Frequency)
        low_target_time_point = low_times[np.argmin(abs(low_times - s))]
        for a in ensure_iter(low_level_area):
            x = pd.pivot_table(
                low_level_data.query('cluster=="%s"' % a),
                index="trial",
                columns="freq",
                values=low_target_time_point,
            )
            lld.append(x)
    return pd.concat(lld, 1)


def save_all_scores(
    all_scores,
    add_meta,
    save_filename,
    save_prefix,
    cnt,
    high_level_latency,
    low_level_peak,
    low_level_area,
    motor_area,
):
    all_scores = pd.concat(all_scores)
    all_scores.loc[:, "high_level_latency"] = high_level_latency
    all_scores.loc[:, "low_level_peak"] = low_level_peak
    all_scores.loc[:, "low_level_area"] = low_level_area
    all_scores.loc[:, "high_level_area"] = motor_area
    for key, value in add_meta.items():
        all_scores.loc[:, key] = value
    scratch = os.environ["TMPDIR"]
    sf = join(scratch, save_filename + save_prefix + "_%i.hdf" % cnt)
    print("Saving to ", sf)
    all_scores.to_hdf(sf, "df")


def chained_categorize(target_a, data_a, data_b):
    """
    Trains a classifier to predict target_a from data_a. Then
    predicts log(pA) with that for training set and trains a 
    second classifier to predict log(pA) with data_b.

    target_a, data_a and data_b all need to have the same index.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score, mean_squared_error
    from sklearn.decomposition import PCA
    from sklearn.utils import shuffle
    from imblearn.pipeline import Pipeline

    if not (
        all(target_a.index.values == data_a.index.values)
        and all(data_a.index.values == data_b.index.values)
    ):
        raise RuntimeError("Target and data not aligned with same index.")

    target_a = target_a.values
    data_a = data_a.values
    data_b = data_b.values
    # Determine prediction target:
    # y_type = type_of_target(target_a)

    metrics = ["roc_auc", "accuracy"]
    classifier_a = svm.SVC(kernel="linear", probability=True)
    classifier_a = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Upsampler", RandomOverSampler(sampling_strategy="minority")),
            ("SVM", classifier_a),
        ]
    )

    classifier_b = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver="full")),
            ("linear_regression", LinearRegression()),
        ]
    )
    classifier_b_baseline = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver="full")),
            ("linear_regression", LinearRegression()),
        ]
    )

    scores = []
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(data_a, target_a):
        train_data_a = data_a[train_index]
        train_target_a = target_a[train_index]
        clf_a = classifier_a.fit(train_data_a, train_target_a)

        train_data_b = data_b[train_index]
        target_b = clf_a.predict_log_proba(train_data_a)[:, 0]
        clf_b = classifier_b.fit(train_data_b, target_b)
        clf_b_baseline = classifier_b_baseline.fit(train_data_b, shuffle(target_b))
        classifier_a_roc = roc_auc_score(
            target_a[test_index], clf_a.predict_log_proba(data_a[test_index])[:, 0]
        )

        clf_a_test_predicted = clf_a.predict_log_proba(data_a[test_index])[:, 0]
        clf_b_test_predicted = clf_b.predict(data_b[test_index])
        clf_b_baseline_test_predicted = clf_b_baseline.predict(data_b[test_index])

        classifier_b_msqerr = mean_squared_error(
            clf_a_test_predicted, clf_b_test_predicted
        )
        classifier_b_shuffled_msqerr = mean_squared_error(
            clf_a_test_predicted, clf_b_baseline_test_predicted
        )

        classifier_b_corr = np.corrcoef(clf_a_test_predicted, clf_b_test_predicted)[
            0, 1
        ]
        classifier_b_shuffled_corr = np.corrcoef(
            clf_a_test_predicted, clf_b_baseline_test_predicted
        )[0, 1]

        scores.append(
            {
                "classifier_a_roc": classifier_a_roc,
                "classifier_b_msqerr": classifier_b_msqerr,
                "classifier_b_shuffled_msqerr": classifier_b_shuffled_msqerr,
                "classifier_b_corr": classifier_b_corr,
                "classifier_b_shuffled_corr": classifier_b_shuffled_corr,
                # "classifier_b_weights": clf_b.steps[-1][1].coef_,
            }
        )

    return pd.DataFrame(scores)
