def pxSmaller(samples, sigma_vals):
    x = linspace(0, 1, 10000)
    return log(norm.cdf(0.5, samples, sigma_vals))

def pxLarger(samples, sigma_vals):
    x = linspace(0, 1, 10000)
    return log(1-norm.cdf(0.5, samples, sigma_vals))

assert(pxLarger([0.5], [0.1]) == log(0.5))
assert(pxLarger([0.6], [0.1]) > pxSmaller([0.6], [0.1]))
assert(exp(pxLarger([0.6], [0.1])) == (1-(exp(pxSmaller([0.6], [0.1])))))

def LPR(samples, sigma):
    return pxLarger(samples, sigma).sum(0)/(pxSmaller(samples, sigma).sum(0))

def LPD(samples, sigma):
    return  (pxLarger(samples, sigma).sum(0))-pxSmaller(samples, sigma).sum(0)


def approx_density(samples):
    Xm, Ys = meshgrid(linspace(0.3, 0.7, 250), linspace(0.002, 0.25, 100))
    e = array([norm.pdf(s, Xm, Ys) for s in samples])
    e = prod(e/(e.sum(1).sum(1)[:,newaxis, newaxis]), 0)
    return Xm, Ys, e/e.sum()

def L_approx_density(samples):
    Xm, Ys = meshgrid(linspace(0.3, 0.7, 250), linspace(0.002, 0.25, 100))
    e = array([norm.pdf(s, Xm, Ys) for s in samples])
    e = sum(log(e/(e.sum(1).sum(1)[:,newaxis, newaxis])), 0)
    return Xm, Ys, e

def pH1(samples):
    Xm, Ys, e = approx_density(samples)
    idx = argmin(abs(Xm[0,:]-0.5))
    return e[:, idx:].sum()

def get_field(N=100, var=[0.05, 0.1, 0.15], t=0.075):
    results = []
    for v in var:
        for n in range(N):
            samples = randn(10)*v + 0.5 + t
            results.append({'nv':v, 'var':samples.std(), 'mean':samples.mean(), 'pH1':pH1(samples)})
            samples = randn(10)*v + 0.5 - t
            results.append({'nv':v, 'var':samples.std(), 'mean':samples.mean(), 'pH1':pH1(samples)})
    return pd.DataFrame(results)

def viz_lp():
    Xm, Ys = meshgrid(linspace(0.3, 0.7, 100), linspace(0.001, 0.25, 100))
    lpr = LPD(Xm.ravel()[newaxis,:], Ys.ravel()[newaxis, :])
    contourf(Xm, Ys, lpr.reshape(Xm.shape), linspace(-20, 20, 51))

def hit_vs_false(t=0.075, N=1000, crits=[-1, -2, -3], scale=0.5):
    hits = []
    fas = []
    vars = [0.4, 0.45, 0.5]
    scalev = lambda x: (x-mean(vars))*scale + mean(vars)
    for var, crit in zip(vars, crits):
        samples = randn(10, N)*var + t + 0.5
        #samples[:, samples.mean(0)>0.5]
        lpr = LPD(samples, scalev(var)) - crit
        hit = sum(lpr>0)/float(N)
        samples = randn(10, N)*var - t + 0.5
        #samples[:, samples.mean(0)<0.5]
        lpr = LPD(samples, scalev(var)) - crit
        fa = sum(lpr>0)/float(N)
        hits.append(hit)
        fas.append(fa)
    return hits, fas
