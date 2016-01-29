function dataset = get_raw_filenames(subject, session, varargin)

path = default_arguments(varargin, 'path', '/home/nwilming/conf_data/conf_meg/');

type = default_arguments(varargin, 'type', 'conf');

sub_info(1).conf = {'s01-01_Confidence_20151208_02.ds',...
                    's01-02_Confidence_20151210_02.ds',...
                    's01-03_Confidence_20151216_02.ds',...
                    's01-04_Confidence_20151217_02.ds'};
sub_info(1).rest = {'s01-01_Confidence_20151208_01.ds',...
                    's01-02_Confidence_20151210_01.ds',...
                    's01-03_Confidence_20151216_01.ds',...
                    's01-04_Confidence_20151217_01.ds'};
                
sub_info(2).conf = {'s02-01_Confidence_20151208_02.ds',...
                    's02-02_Confidence_20151210_02.ds',...
                    's02-03_Confidence_20151211_02.ds',...
                    's02-4_Confidence_20151215_02.ds'};
sub_info(2).rest = {'s02-01_Confidence_20151208_01.ds',...
                    's02-02_Confidence_20151210_01.ds',...
                    's02-03_Confidence_20151211_01.ds',...
                    's02-4_Confidence_20151215_01.ds'};

sub_info(3).conf = {'s03-01_Confidence_20151208_01.ds',...
                    's03-02_Confidence_20151215_02.ds',...
                    's03-03_Confidence_20151216_02.ds',...
                    's03-04_Confidence_20151217_02.ds'}; 
sub_info(3).rest = {'',...
                    's03-02_Confidence_20151215_01.ds',...
                    's03-03_Confidence_20151216_01.ds',...
                    's03-04_Confidence_20151217_01.ds'};
                    
sub_info(4).conf = {'s04-01_Confidence_20151210_02.ds',...
                    's04-02_Confidence_20151211_02.ds',...
                    's04-03_Confidence_20151215_02.ds',...
                    's04-04_Confidence_20151217_02.ds'};
sub_info(4).rest = {'s04-01_Confidence_20151210_01.ds',...
                    's04-02_Confidence_20151211_01.ds',...
                    's04-03_Confidence_20151215_01.ds',...
                    's04-04_Confidence_20151217_01.ds'};            

sub_info(5).conf = {'s05-01_Confidence_20151210_02.ds',...
                    's05-02_Confidence_20151211_02.ds',...
                    's05-03_Confidence_20151216_02.ds',...
                    's05-04_Confidence_20151217_02.ds'};          
sub_info(5).rest = {'s05-01_Confidence_20151210_01.ds',...
                    's05-02_Confidence_20151211_01.ds',...
                    's05-03_Confidence_20151216_01.ds',...
                    's05-04_Confidence_20151217_01.ds'};                  
                
sub_info(6).conf = {'s06-01_Confidence_20151215_02.ds',...
                    's06-02_Confidence_20151216_02.ds',...
                    's06-03_Confidence_20151217_02.ds',...
                    's06-04_Confidence_20151218_02.ds'};
sub_info(6).rest = {'s06-01_Confidence_20151215_01.ds',...
                    's06-02_Confidence_20151216_01.ds',...
                    's06-03_Confidence_20151217_01.ds',...
                    's06-04_Confidence_20151218_01.ds'};                

dataset = fullfile(path, sub_info(subject).(type)(session));
dataset = dataset{1};
