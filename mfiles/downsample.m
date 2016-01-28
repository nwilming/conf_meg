function [data] = downsample(cfg, data, samplerows, varargin)
fs = default_arguments(varargin, 'fs', 1200);
rfs = default_arguments(varargin, 'rfs', 500);
detrend = default_arguments(varargin, 'detrend', 'no');

cfgs             = [];
cfgs.resamplefs  = rfs;
cfgs.fsample     = fs;
cfgs.detrend     = detrend;

data.trialinfo(:,samplerows) = round(data.trialinfo(:,samplerows) * (cfg.resamplefs/cfg.fsample));
data        = ft_resampledata(cfgs, data);

cfg.trl(:, 1:3) = round(cfg.trl(:,1:3) * (cfg.resamplefs/cfg.fsample));
cfg.trl(:, samplerows + 3) = round(cfg.trl(:,samplerows+3) * (cfg.resamplefs/cfg.fsample));


