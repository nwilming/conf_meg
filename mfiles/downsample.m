function [data] = downsample(data, samplerows, varargin)
fs = default_arguments(varargin, 'fs', 1200);
rfs = default_arguments(varargin, 'rfs', 500);
detrend = default_arguments(varargin, 'detrend', 'yes');

cfg             = [];
cfg.resamplefs  = rfs;
cfg.fsample     = fs;
cfg.detrend     = detrend;

data.trialinfo(:,samplerows) = round(data.trialinfo(:,samplerows) * (cfg.resamplefs/cfg.fsample));
data        = ft_resampledata(cfg,data);