
data {

    int<lower=0> N; // Number of trials
    int nfreq; // Number of frequencies    
    vector[nfreq] y[N]; // power
    real<lower=-0.5, upper=0.5> contrast[N]; // contrast
    
    
}

parameters {    
    vector[nfreq] amplitude; //p parameter        
    vector[nfreq] slope; //p parameter        
    vector[nfreq] c50; //p parameter        
    vector[nfreq] yb; //p parameter        
    cholesky_factor_corr[nfreq] Lcorr;     
    vector<lower=0>[nfreq] tau;
}

transformed parameters {}


model {
    vector[nfreq] mu[N]; // predicted power
    matrix[nfreq, nfreq] L_Sigma;

    for (n in 1:N){
        mu[n] = amplitude .* inv_logit(slope .* (contrast[n] + c50) - 0.5) + yb;
    }
    amplitude ~ normal(1, 2);
    yb ~ normal(0, 0.5);
    slope ~ normal(0, 10);
    c50 ~ normal(0, 0.1);    
    tau ~ cauchy(0, 1);
    L_Sigma = diag_pre_multiply(tau, Lcorr);
    Lcorr ~ lkj_corr_cholesky(2);    
    y ~ multi_normal_cholesky(mu, L_Sigma);
    
}

/*
%pylab
%load_ext autoreload
%autoreload 2
from imp import reload
import pandas as pd
import pystan

def get_d():
    meta = pd.read_hdf('tmp.hdf', 'meta')
    agg = pd.read_hdf('tmp.hdf', 'agg')
    X = pd.pivot_table(agg, values=0.2, index='trial', columns='freq')
    contrast = meta.loc[X.index, 'contrast_probe']
    cvals = np.stack(contrast)
    cvals[cvals<0] = 0
    cvals[cvals>1] = 1
    x = X.values[:, ::2]
    x = (x-x.mean(0)[np.newaxis,:])/x.std(0)[np.newaxis,:]    
    return {'N':x.shape[0], 'nfreq':x.shape[1], 'y':x, 
        'ones':cvals[:,0]*0+1, 'contrast':cvals[:,0]-0.5}

d = get_d()


def init(x):
    nfreq = x.shape[1]
    corr = np.corrcoef(x.T)
    def foo():
        return dict(p=np.array([0.5]*nfreq), c50=np.array([0]*nfreq), 
            Lcorr=np.linalg.cholesky(corr), tau=np.std(x, 0))
    return foo

foo = init(d['y'])
*/
