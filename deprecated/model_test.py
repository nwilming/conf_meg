from pylab import *
from scipy.stats import norm, t
from scipy.special import gammaln
import pandas as pd
import pylab as pyplot

class NG(object):
    '''
    A normal-gamma distribution
    '''
    def __init__(self, mu=None, alpha=None, beta=None, std=None, precision=None):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        assert precision is not None
        self.precision = precision
        if self.alpha < 0 or self.beta < 0 or self.precision < 0:
            raise ValueError('lambda, alpha and beta need to be > 0')
            #self.std = (1./precision)**.5

    def __call__(self, MU, SIGMA):
        PREC = SIGMA #s2p(SIGMA)
        Zng = ((gamma(self.alpha))/(self.beta**self.alpha)) * (2*pi/self.precision)**.5
        Za = PREC**(self.alpha-0.5)
        Zc = (-PREC/2) * (self.precision*((MU-self.mu)**2) + 2*self.beta)
        return (1./Zng)*Za*exp(Zc)

def s2p(sigma):
    '''
    Convert standard deviation to precisions
    '''
    return 1/(sigma**2)

def PNG(prior_mean=None, prior_precision=None, palpha=None, pbeta=None):
    '''
    Returns the posterior distribution for infering mean and variance of a normal
    distribution.
    Roughly speaking prior_mean and prio_std control what mean to expect and what
    std this mean has. palpha and pbeta control the prior for the variance.
    '''

    def posterior(samples):
        sample_mu, s = mean(samples), std(samples)**2.
        n = float(len(samples))
        mu = (prior_precision*prior_mean + n*sample_mu)/(prior_precision+n)
        precision = prior_precision+n
        alpha = palpha +n/2.
        beta = (pbeta
                    + 0.5*
                    (n*s +
                    ((prior_precision*n*((sample_mu-prior_mean)**2))/(prior_precision+n))))
        return NG(mu=mu, alpha=alpha, beta=beta, precision=precision)
    return posterior

def PMu(prior_mean=None, prior_precision=None, palpha=None, pbeta=None):
    '''
    Returns a marginal distribution for finding P(mu|D).
    '''
    def marginal(samples, cdf=False):
        sample_mu, s = mean(samples), std(samples)**2.
        n = float(len(samples))
        mu = (prior_precision*prior_mean + n*sample_mu)/(prior_precision+n)
        precision = prior_precision+n
        alpha = palpha +n/2.
        beta = (pbeta
                    + 0.5*(sum((samples-sample_mu)**2)
                    + ((prior_precision*n*((sample_mu-prior_mean)**2))
                                /(2*(prior_precision+n)))))
        tmu = mu
        tsigma = beta/(alpha*precision)
        if cdf:
            return lambda x: t.cdf(x, 2*alpha, tmu, tsigma**.5)
        return lambda x: t.pdf(x, 2*alpha, tmu, tsigma**.5)
    return marginal

def PestH1(**kw):
    marginal = PMu(**kw)
    def PH1(samples, cutoff=0):
        return 1-marginal(samples, cdf=True)(cutoff)
    return PH1


def PMuVec(prior):
    '''
    Returns a marginal distribution for finding P(mu|D).
    '''
    def marginal(sample_mu, sample_prec, n, cdf=False):
        sample_var = p2s(sample_prec)
        n = float(n)
        mu = (prior.precision*prior.mu + n*sample_mu)/(prior.precision+n)
        precision = prior.precision+n
        alpha = prior.alpha +n/2.
        beta = (prior.beta
                    + 0.5*(n*sample_var)
                    + ((prior.precision*n*((sample_mu-prior.mu)**2))
                                /(2*(prior.precision+n))))
        tmu = mu
        tsigma = beta/(alpha*precision)
        if cdf:
            return lambda x: t.cdf(x, 2*alpha, tmu, tsigma)
        return lambda x: t.pdf(x, 2*alpha, tmu, tsigma)
    return marginal

def PestH1Vec(prior):
    marginal = PMuVec(prior)
    def PH1(mean, s, N, cutoff=0):
        return 1-marginal(mean, s, N, cdf=True)(cutoff)
    return PH1

def classify(samples, pH1, bias):
    mean = samples.mean(0)
    s = samples.std(0)
    p = pH1(mean+bias, s, len(samples))
    return p>0.5


def hvsf(N, classify, t):
    hits = []
    fas = []
    vars = [0.05, 2., 4.95]
    for var in vars:
        samples = randn(10, N)*var + t
        hit = sum(classify(samples))/float(N)
        samples = randn(10, N)*var - t
        fa = sum(classify(samples))/float(N)
        hits.append(hit)
        fas.append(fa)
    return hits, fas

# Compute FA and hit rate as function of sample_variance
def foo(variances, prior, t=1., side=1):
    samples_2nd = np.concatenate(
        [randn(10, 100000)*v+t*side for v in variances]
        )
    sm = samples_2nd.mean(0)
    ss = samples_2nd.std(0)
    print ss.shape, mean(ss)
    p = PestH1Vec(NG(**prior))
    p2nd = lambda x,s: p(x, s, 10.)
    phit = p2nd(sm, s2p(ss))
    return sm, ss, phit

def bar(x,y,c, ea, eb):
    out = nan*empty((len(ea)-1, len(eb)-1))
    for i, (xl, xh) in enumerate(zip(ea[:-1], ea[1:])):
        idx = (xl<x) & (x<=xh)
        for j, (yl, yh) in enumerate(zip(eb[:-1], eb[1:])):
            idy = (yl<y) & (y<=yh)
            out[i, j] = nanmean(c[idx&idy])
    return out.T

def centers(edges):
    return [low+(high-low)/2. for low, high in zip(edges[:-1], edges[1:])]

def p(prior):
    ng = NG(**prior)
    print "Mode of prior: %2.2f, %2.2fP / %2.2fS"%(ng.mu, (ng.alpha-.5)/ng.beta, p2s((ng.alpha-.5)/ng.beta))
    M,P = meshgrid(linspace(-2, 2, 151), linspace(s2p(6.5), s2p(0.15), 1051))
    clf()
    subplot(2,3,1)
    contourf(M, p2s(P), NG(**prior)(M,P), 51)
    axhline(p2s((ng.alpha-.5)/ng.beta), color='k')
    pyplot.locator_params(nbins=4)

    ylabel('Sample STD')
    xlabel('Sample Mean')
    title('Prior believe')
    subplot(2,3,2)

    title(r'$log(P(\mu>0|Stimulus))$')
    pH1 =  PestH1Vec(ng)(M,P, 10)
    contourf(M, p2s(P), pH1, 11)
    contour(M, p2s(P), pH1, [.5], colors='k')
    pyplot.locator_params(nbins=4)

    xlabel('Sample Mean')
    yticks([])

    subplot(2,3,3)
    title(r'$log \frac{P(\mu>0|Stimulus)}{P(\mu<0|Stimulus)}$')
    dP = log(PestH1Vec(ng)(M,P, 10)) - log((1-PestH1Vec(ng)(M,P, 10)))
    CS = contourf(M, p2s(P), dP, 11)
    contour(M, p2s(P), dP, [0.0], colors='k')
    pyplot.locator_params(nbins=4)

    xlabel('Sample Mean')
    yticks([])

    #axvline(0.075, color='k')
    '''
    subplot(2,3,4)
    em, es = linspace(-5.5, 5.5, 51), linspace(0, 5.5, 151)
    sm, ss, phit = foo([.1, .1, .1], prior, side=-1)
    out = bar(sm, ss, phit, em, es)
    print out.shape
    print phit
    contourf(centers(em), centers(es), out, linspace(0, 1, 11), cmap=winter())
    contour(M, p2s(P), PestH1Vec(ng)(M,P, 10), [0.5], colors='k')
    fa = []
    for i, (yl, yh) in enumerate(zip(es[:-1], es[1:])):
        idy = (yl<ss) & (ss<=yh)
        fa.append(sum(phit[idy]>0.5)/float(sum(idy)))

    subplot(2,3,5)
    sm, ss, phit = foo([.1, .1, .1], prior, side=1)
    out = bar(sm, ss, phit, em, es)
    print phit
    contourf(centers(em), centers(es), out, linspace(0, 1, 11), cmap=winter())
    contour(M, p2s(P), PestH1Vec(ng)(M,P, 10), [0.5], colors='k')
    subplot(2,3,6)
    hit = []
    for i, (yl, yh) in enumerate(zip(es[:-1], es[1:])):
        idy = (yl<ss) & (ss<=yh)
        hit.append(sum(phit[idy]>0.5)/float(sum(idy)))
    plot(centers(es), hit, label='hits')
    plot(centers(es), fa, label='fa')
    legend()
    #xlim([.75, 1.75])
    xlabel('STD')
    '''
    return prior

def p2s(prec):
    return 1./(prec**.5)

def gamma_params(mode=10., sd=10.):
    '''
    Converst mode and sd to shape and rate of a gamma distribution.
    '''
    var = pow(sd, 2)
    rate = (mode + pow(pow(mode, 2) + 4*var, 0.5))/(2 * var)
    shape = 1+mode*rate
    return shape, rate


def pbymode(mx, my, vx, vy, dict=True):
    '''
    Calculate parameters of NG based on mode. The mode is given by:
        mx = mode of the mean of the gaussian
        my = mode of the precision of the gaussian
        vx = variance of mean of the gaussian
        vy = variance of the variance of the gaussian

    Does not work for all combinations of inputs. No idea why.
    '''
    alpha = 0.5*(my**2 + my*sqrt(my**2 + 2.0*vy) + vy)/vy
    lamb =  (3.0*my + sqrt(my**2 + 2.0*vy))/(vx*(4.0*my**2 - vy))
    beta =  0.5*(my + sqrt(my**2 + 2.0*vy))/vy
    mu = mx
    if dict:
        return {'mu':mu, 'precision':lamb, 'alpha':alpha, 'beta':beta}
    return mu, lamb, alpha, beta

def predict(data, prior):
    ng = NG(**prior)
    ph1 =  PestH1Vec(ng)
    mc = data.mc
    mc = (mc-mean(mc))/std(mc)
    sigma = data.stdc
    sigma = (sigma)/std(sigma)
    lpr = lambda x, y: log(ph1(x,y, 10)) - log((1-ph1(x,y, 10)))
    return lpr(mc, sigma)
