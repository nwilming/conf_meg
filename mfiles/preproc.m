function preproc(subject, session)
dataset = get_raw_filenames(subject, session);
[data, cfg] = prepare_dataset(dataset, subject, session, 'trialdef');
artifacts = reject_artifacts(dataset, data);
cfg.artfctdef = artifacts;

% Compute the time period of interest: Between ref onset and decision. 
event_fields = {'start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',...
    'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',...
    'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',...
    'response_v', 'feedback', 'feedback_v', 'end'};

trial_onset = cfg.trl(:, 1);
ref_onset = cfg.trl(:, find(strcmp('ref_onset', event_fields)) + 3);
response = cfg.trl(:, find(strcmp('response', event_fields)) + 3);

cfg.artfctdef.crittoilim = [ref_onset, response]/1200;
cfg.artfctdef.reject = 'complete';

datat = ft_redefinetrial(cfg, data);
clean = ft_rejectartifact(cfg, datat);

samplerows = find(clean.trialinfo(1,:)>255);
clean = downsample(clean, samplerows);
savepath = get_filenames(subject, session, 'confidence');
clear data datat
save(savepath)
end