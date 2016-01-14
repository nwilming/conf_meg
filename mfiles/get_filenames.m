function dataset = get_filenames(subject, session, varargin)

path = default_arguments(varargin, 'path', '/home/nwilming/conf_data/conf_meg/');

path = fullfile(path, sprintf('S%i', subject));
mkdir(path);

type = default_arguments(varargin, 'type', 'confidence');

dataset = fullfile(path, sprintf('%02i_%02i_%s.mat', subject, session, type));
