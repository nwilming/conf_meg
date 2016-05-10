'''
Implements a normative model that infers mean of a sample in the face of unknown
mean and variance of the generative distribution.
'''
import numpy as np
from scipy.stats import norm, gamma
from scipy.stats import t
from scipy.special import gammaln
from numba import jit


def NG(mu, precision, mu0, kappa0, alpha0, beta0):
    '''
    PDF of a Normal-Gamma distribution.
    '''
    return norm.pdf(mu, mu0, 1./(precision*kappa0)) * gamma.pdf(precision, alpha0, scale=1./beta0)


@jit
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


@jit
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


def logLPR(mean, std, prior):
    '''
    This gives the log posterior ration. Since probabilities can
    easily become 0 or 1 I make sure to clip them to 0+eps and 1-eps.
    '''
    mean = np.asarray(mean)
    std = np.asarray(std)
    pL = plarger(mean, std, 10., prior)
    pL = np.maximum(np.minimum(pL, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    #pS = psmaller(mean, std, 10., prior)
    #pS = np.maximum(np.minimum(pS, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    return np.log(pL) - np.log(1-pL)


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
        self.bias=float(bias)
        self.conf_threshold = float(conf_threshold)
        self.prior = prior

        if prior is None:
            print 'Prior is None, Alarm!'
            self.prior = 0., .01, .01, .01

    def __str__(self):
        s = 'Bias: %2.1f, conf_threshold: %2.1f, prior:'%(self.bias, self.conf_threshold)
        for i in self.prior:
            s +=  '%2.4f  '%i
        return s

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


def vec_multinomial(xs, ps):
   """
   Return the probability to draw counts xs from multinomial distribution with
   probabilities ps.

   Returns:
       probability: float
   """
   xs, ps = np.array(xs), np.array(ps)
   n = sum(xs, 0)

   result = gammaln(n+1) - sum(gammaln(xs+1), 0) + sum(xs * np.log(ps), 0)
   return -result


def err_fct(parameters, true, data):
    '''
    Computes the likelihood of data given a set of parameters.
    '''
    prior = parameters[2:]
    obs = IdealObserver(bias=parameters[0], conf_threshold=np.log(parameters[1]), prior=prior)

    answers = np.ones((4, len(true)))*0
    for i, idx in enumerate(true):
        answers[idx, i] = 1

    ps = obs.p(data[0], data[1])
    val = vec_multinomial(answers, ps).sum()
    return val

def opt_err_fct(parameters, true, data):
    '''
    Approximates the likelihood of data given a set of parameters.
    '''
    prior = parameters[2:]
    prior[1:] = abs(prior[1:]) 
    obs = IdealObserver(bias=parameters[0], conf_threshold=np.log(parameters[1]), prior=prior)
    p = obs(data[0], data[1])
    idx = ((true==p)/2.)+0.25
    return -sum(np.log(idx))


def fit(decisions, mean, sigma, x0=[0, 1., 0., 1., 1., 1.], eval_only=False,
        Ns=4, rranges=None, **kwargs):
    from scipy import optimize
    #decisions = decisions+2
    #decisions[decisions>1] -= 1
    #assert all(np.unique(decisions) == np.unique([0,1,2,3]))
    data = (mean, sigma)
    err = lambda x: opt_err_fct(x, decisions, data)
    if eval_only:
        return err
    if rranges is None:
        rranges = ((-3, 3), # Bias
               (1, 5), # Conf threshold
               (-5, 5), #mean prior
               (0, 5), #kappa
               (0, 10),  #alpha
               (0, 5))
    resbrute = optimize.brute(err, rranges, full_output=True, finish=optimize.fmin, Ns=Ns)
    bias = resbrute[0][0]
    cutoff = resbrute[0][1]
    prior = resbrute[0][2:]
    return bias, cutoff, prior, resbrute

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
