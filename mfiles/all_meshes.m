subs = {'S04', 'S05', 'S06', 'S07',...
    'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15'}

make_mesh('S03', '/home/nwilming/fs_subject_dir')

%for ii = 1:14
%    make_mesh(subs{ii}, '/home/nwilming/fs_subject_dir');
%end