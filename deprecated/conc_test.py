import mne
# Generate some random data
data = np.random.randn(5, 1000)*10e-12

sfreq = 100.
# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'MEG3', 'MEG4', 'MEG5'],
    ch_types=['grad']*5,
    sfreq=sfreq)

# Create 5 raw structures with annotations at 1 and 2 seconds into recording. Each
# raw has 10s of data
raws = []
for fs in [1000, 100, 12, 2001, 171]:
    ants = mne.annotations.Annotations(onset=[1., 2], duration=[.5, .5], description='x')
    r = mne.io.RawArray(data.copy(), info, first_samp=fs)
    r.annotations = ants
    r.annotations.onset += fs/sfreq
    raws.append(r)

conc = mne.concatenate_raws([r.copy() for r in raws])

# Correct annotations should be visible at 1,2, 11, 12, 21, 22, 31, 32, 41, and 42s
# when calling conc.plot().
corr_ants = mne.annotations.Annotations(
            onset=[1., 2, 11, 12, 21, 22, 31, 32, 41, 42], duration=[.5]*10, description='x')
corr_ants.onset += raws[0].first_samp/sfreq

assert all(conc.annotations.onset == corr_ants.onset)

def combine_annotations(annotations, first_samples, last_samples, sfreq):
    durations = [(1+l-f)/sfreq for f, l in zip(first_samples, last_samples)]
    offsets = cumsum([0] + durations[:-1])

    onsets = [(ann.onset-(fs/sfreq))+offset
                        for ann, fs, offset in zip(annotations, first_samples, offsets)]
    onsets = np.concatenate(onsets) + (first_samples[0]/sfreq)
    return mne.annotations.Annotations(onset=onsets,
        duration=np.concatenate([ann.duration for ann in annotations]),
        description=np.concatenate([ann.description for ann in annotations]))

d = combine_annotations([r.annotations for r in raws],
                        [r.first_samp for r in raws],
                        [r.last_samp for r in raws], sfreq)

assert all(d.onset == corr_ants.onset)

m = mne.concatenate_raws([r.copy() for r in raws])
m.annotations = d
