function [data, cfg] = prepare_dataset(dataset, subject, session, trialfun)
% read in the dataset as a continuous segment
cfg                         = [];
cfg.dataset                 =  dataset;
cfg.continuous              = 'yes'; % read in the data
cfg.precision               = 'single'; % for speed and memory issues
cfg.channel                 = channel_names(dataset);
cfg.sj                      = subject;
cfg.session                 = session;
cfg.trl =  [204276      863968      209354];
% add a highpass filter to get rid of cars
% 0.1 Hz, see Acunzo et al
% if filtering higher than that, can't look at onset of ERPs!
cfg.hpfilttype              = 'fir';
cfg.hpfiltord               = 6;
cfg.hpfilter                = 'yes';
cfg.hpfreq                  = 0.1;

data = ft_preprocessing(cfg);

fprintf('\nNow epoching\n')

cfg = rmfield(cfg, 'trl');
cfg.trialfun                = trialfun;


cfg.trialdef.pre        = 0; % before the fixation trigger, to have enough length for padding (and avoid edge artefacts)
cfg.trialdef.post       = 1; % after feedback
cfg 					= ft_definetrial(cfg); % define all trials
data 					= ft_preprocessing(cfg, data);

data = ft_selectdata(cfg, data);
