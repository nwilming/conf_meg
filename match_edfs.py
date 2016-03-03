'''
Some of the edfs were not copied from the host PC correctly to stimulus PC. Need
to find corrupt ones and match these to those from the host PC.
'''
import load_edfs as le
import edfread
import locale


locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
corrupt_files = {}

for sub in ['S07', 'S08', 'S09', 'S10', 'S11', 'S12']:
        edf, mf, sub = le.listfiles('/Users/nwilming/u/conf_data/'+sub)
        for _, e in edf.iteritems():
                try:
                    preamble = edfread.read_preamble(e)
                except IOError:
                    preamble = edfread.read_preamble(e, 4)
                    rec_time = preamble.split('\n')[0].replace('** DATE: ', '')
                    corrupt_files[rec_time] = e
                #print preamble

backup_dir = '/Users/nwilming/Desktop/edfs2/'
import glob
backup_files = glob.glob(backup_dir + '*.edf')
backups = {}
for e in backup_files:
    preamble = edfread.read_preamble(e, 4)
    rec_time = preamble.split('\n')[0].replace('** DATE: ', '')
    backups[rec_time] = e

for t, name in corrupt_files.iteritems():
    try:
        print t, name, '->', backups[t]
        print 'cp ', backups[t], name
    except:
        print 'No match!'

import cPickle
cPickle.dump(backups, open('backup_time_stamps.pickle', 'w'))
