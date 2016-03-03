import load_edfs as le
from pylab import *
import time
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

print 'sb2fname = {'
for subject in ['S%02i'%i for i in range(1, 15)]:
    edf, mf, s = le.listfiles('/Users/nwilming/u/conf_data/%s'%subject)
    edf = dict((k, edf[k]) for k in sorted(edf))
    mf = dict((k, mf[k]) for k in sorted(mf))
    d = {}

    # For each day list files sorted by time
    unique_days = unique([int('%s%s%s'%(t.tm_year, t.tm_mon, t.tm_mday)) for t in edf.keys()])

    days2sessions = dict((d, i+1) for i, d in enumerate(sorted(unique_days)))

    for ei, mi in zip(argsort([time.mktime(a) for a in edf.keys()]), argsort([time.mktime(b) for b in mf.keys()])):
        e = edf.keys()[ei]
        day = int('%s%s%s'%(e.tm_year, e.tm_mon, e.tm_mday))
        session = days2sessions[day]
        if not session in d.keys():
            d[session] = [(edf.values()[ei], mf.values()[mi])]
        else:
            d[session].append((edf.values()[ei], mf.values()[mi]))

    print "\t'%s':{"%(subject)
    for session, files in d.iteritems():
        print '\t\t%i:{'%session
        for block, f in enumerate(files):
            print '\t\t\t%i:%s,'%(block, f)
        print '\t\t},'
    print '\t},'
print '}'

print '\n'
print '''
fnames2sb = {}
for sub, d in sb2fname.iteritems():
    fnames2sb = dict((v[0].split('/')[-1], (session, block, v[1])) for session, bvs in d.iteritems() for block, v in bvs.iteritems())
'''
