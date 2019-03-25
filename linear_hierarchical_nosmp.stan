/*
Linear model for MEG power data from visual cortex.

Reduced to ignore sample number.

Lcorr_all: Needs to be init'd with np.eye, because cholesky decomp for large (>8x8) 
covariance matrices fails. 
https://discourse.mc-stan.org/t/cholesky-factor-corr-initializing-with-invalid-values/3167

Hierarchichal structure:

CRF(Rmin, Rmax, p, c50)

Rmin ~ per subject / freq
Rmax ~ per subject / freq

(p ~ (per sample and freq ~ subject))
//(c50 ~ (per sample and freq ~ subject))
        
Lcorr ~ per frequency
*/  
functions{
     real crf(real contrast, real P, real Rmin){        
        real result;                
        result = P * contrast - Rmin;               
        return result;

     }
}
data {
    // One trial is one sample presentation
    int nsub;                           // Number of subject
    int<lower=0> N;                     // Number of trials
    int nfreq;                          // Number of frequencies     
    int subject[N];                     // Subject per trial
    vector[nfreq] y[N];                 // Observed power
    real<lower=0, upper=1> contrast[N]; // contrast        
    vector[nfreq] rmin_emp[nsub];       // Each subject has own rmin
}

parameters {    
    // Priors:    
    matrix<lower=-100, upper=100>[nsub, nfreq] Rmin;       // Response at 0% contrast    
    vector<lower=0.001>[nfreq] P[nsub];                    // Exponent            
    cholesky_factor_corr[nfreq] Lcorr_all;                 // Cholesky factor of cov matrix
    vector<lower=0>[nfreq] tau;                            // Variance of cov matrix

    // Hyperpriors:
    vector<lower=0.001>[nfreq] P_pop;                    // Population prior for P    
}

transformed parameters {
    matrix[nfreq, nfreq] L_Sigma;
    L_Sigma = diag_pre_multiply(tau, Lcorr_all);
}


model {

    for (sub in 1:nsub){
        for (f in 1:nfreq){            
            Rmin[sub, f] ~ normal(rmin_emp[sub, f], 1);
        }
    }
            
    for (f in 1:nfreq){        
        for (sub in 1:nsub){
            P[sub][f] ~ normal(P_pop[f], 1);        
        }
        P_pop[f] ~ normal(1, 15);            
    }

    
    tau ~ cauchy(0, 1);

    Lcorr_all ~ lkj_corr_cholesky(2);   
    {
        vector[nfreq] mu[N]; // predicted power     
        for (n in 1:N){
            for (f in 1:nfreq){
                mu[n, f] = crf(contrast[n], P[subject[n]][f], Rmin[subject[n]][f]);
            }
        }   

        y ~ multi_normal_cholesky(mu, L_Sigma);
    }
    
}