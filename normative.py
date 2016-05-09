'''
Implements a normative model that infers mean of a sample in the face of unknown
mean and variance of the generative distribution.
'''
import numpy as np
from scipy.stats import norm, gamma
from scipy.stats import t
from scipy.special import gammaln


def NG(mu, precision, mu0, kappa0, alpha0, beta0):
    '''
    PDF of a Normal-Gamma distribution.
    '''
    return norm.pdf(mu, mu0, 1./(precision*kappa0)) * gamma.pdf(precision, alpha0, scale=1./beta0)


def NGposterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mu0, k0, a0, b0 = prior
    mun = (k0*mu0 + n*xbar)/(k0+n)
    kn = k0+n
    an = a0 + (n/2.)
    bn = (b0 + 0.5 *  n*(sigma**2)
                      + (k0*n*((xbar-mu0)**2))
                                  /(2*(k0 + n)))
    return mun, kn, an, bn


def Mu_posterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution of mu for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mun, kn, an, bn = NGposterior(xbar, sigma, n, prior)
    df = 2*an
    loc = mun
    scale = bn/(an*kn)
    return df, loc, scale


def Sigma_posterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution of sigma for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mun, kn, an, bn = NGposterior(xbar, sigma, n, prior)
    return an, 1./bn


def plarger(mean, std, n, prior):
    '''
    Calculate the probability that a sample has a mean value larger than 0. This
    works by marginalizing out the standard deviation.
    '''
    df, loc, scale = Mu_posterior(mean.ravel(), std.ravel(), n, prior)
    return 1.-t(df,loc,scale).cdf(0).reshape(mean.shape)


def psmaller(mean, std, n, prior):
    '''
    Calculate the probability that a sample has a mean value smaller than 0:
        psmaller(m,s,n, prior) = 1-plarger(mean, std, n, prior)
    '''
    df, loc, scale = Mu_posterior(mean.ravel(), std.ravel(), n, prior)
    return t(df,loc,scale).cdf(0).reshape(mean.shape)


def logLPR(mean, std, prior, debug=False):
    '''
    This gives the log posterior ration. Since probabilities can
    easily become 0 or 1 I make sure to clip them to 0+eps and 1-eps.
    '''
    mean = np.asarray(mean)
    std = np.asarray(std)
    pL = plarger(mean, std, 10., prior)
    pL = np.maximum(np.minimum(pL, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    pS = psmaller(mean, std, 10., prior)
    pS = np.maximum(np.minimum(pS, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    if debug:
        print 'p<0:', pS, 'p>0:', pL
    return np.log(pL) - np.log(pS)


def get_samples(N=10, thresh=0.25, symmetric=True, vars=[0.05, 0.1, 0.15]):
    '''
    Generate a set of samples in analogy to my experiment.
    '''
    if not symmetric:
        return np.concatenate(
              [np.random.randn(N, 10)*v + thresh for v in vars])
    else:
        return np.concatenate(
              [np.random.randn(N, 10)*v + thresh for v in vars]
             +[np.random.randn(N, 10)*v - thresh for v in vars])


class IdealObserver(object):
    '''
    Represents an ideal observer with a fixed prior.
    '''

    def __init__(self, bias, conf_threshold, prior=None):
        self.bias=bias
        self.conf_threshold = conf_threshold
        self.prior = prior
        if prior is None:
            self.prior = 0, .01, .01, .01


    def __call__(self, mean, sigma, index=False):
        '''
        Return a decision (-2, -1, 1, 2) for sample with mu=mean and sigma=sigma.
        '''
        LPR = logLPR(mean, sigma, self.prior)-self.bias
        confidence = 1 + (abs(LPR) >  self.conf_threshold)
        choice = np.sign(LPR)
        choice[choice==0]+=1
        if not index:
            return choice*confidence
        else:
            index = choice*confidence+2
            index[index>1] -= 1
            return index

    def p(self, mean, sigma):
        '''
        Return multinomial probability vector for samples.
        '''
        decision = self(mean, sigma, index=True)
        ps = np.ones((4, len(decision)))*0 + 0.01
        for i, idx in enumerate(decision):
            ps[idx, i] = 1
        return ps/ps.sum(0)


def err_fct(parameters, true, data):
    '''
    Computes the likelihood of data given a set of parameters.
    '''
    obs = IdealObserver(bias=parameters[0], conf_threshold=np.log(parameters[1]), prior=parameters[2:])
    answers = np.ones((4, len(true)))*0
    for i, idx in enumerate(true):
        answers[idx, i] = 1
    ps = obs.p(data[0], data[1])
    return sum([multinomial(a,p) for a, p in zip(answers.T, ps.T)])


def multinomial(xs, ps):
   """
   Return the probability to draw counts xs from multinomial distribution with
   probabilities ps.

   Returns:
       probability: float
   """
   xs, ps = np.array(xs), np.array(ps)
   n = sum(xs)

   result = gammaln(n+1) - sum(gammaln(xs+1)) + sum(xs * np.log(ps))
   return -result


def p2s(precision):
    '''
    Convert precision to standard deviation.
    '''
    return (1./precision)**.5


def s2p(sigma):
    '''
    Convert standard deviation to precision.
    '''
    return 1./(sigma**2)
