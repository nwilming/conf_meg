def gethist(s):
    fs = glob.glob('S%i/*.*def'%s)
    results = {}
    edges = linspace(0, 1000, 1001)
    for t in ['muscle', 'jumps']:
        d = np.concatenate([cPickle.load(open(f))[t] for f in fs])

        results[t] = histogramm(d, edges)[0]
    d = np.concatenate([cPickle.load(open(f))['cars'][0] for f in fs])
    results['cars'] = histogramm(d, edges)[0]
    d = np.concatenate([cPickle.load(open(f))['cars'][1] for f in fs])
    results['cars_dt'] = histogramm(d, edges)[0]
    return results
