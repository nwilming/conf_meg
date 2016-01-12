function preproc(subject, session)
dataset = get_raw_filenames(subject, session);
[data, cfg] = prepare_dataset(dataset, subject, session, 'trialdef');
[data, artifacts] = reject_artifacts(dataset, data);
data = downsample(data, samplerows);
savepath = get_filenames(subject, session, 'confidence');
save(savepath)
end