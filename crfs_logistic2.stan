/*
CRF model for MEG power data from visual cortex.

Lcorr_all: Needs to be init'd with np.eye, because cholesky decomp for large (>8x8) 
covariance matrices fails. 
https://discourse.mc-stan.org/t/cholesky-factor-corr-initializing-with-invalid-values/3167
    
*/
functions{
     real crf(real contrast, real Rmax, real Rmin,
              real p, real C50){        
        real result;
        vector[1] tmp;
        result = Rmax * ((contrast ^ p) / ((contrast ^ p) + (C50 ^ p))) + Rmin;       
        /*if (is_inf(result) || is_nan(result)){
            print("Con: ", contrast, " Rmax: ", Rmax, " Rmin: ", Rmin, " p: ", p, " c50: ", C50, " result: ", result);
            print(tmp[100000000]);
        }*/
        return result;

     }
}
data {
    // One trial is one sample presentation
    int<lower=0> N; // Number of trials
    int nfreq; // Number of frequencies 
    int nsamp; // Number of samples   
    vector[nfreq] y[N]; // power
    real<lower=0, upper=1> contrast[N]; // contrast
    int samples[N]; // Sample per trial
    vector[nfreq] rmax_emp;
    vector[nfreq] rmin_emp;
}

parameters {    
    vector<lower=0.001, upper=1000>[nfreq] Rmax; // Max amplitude relative to Rmin
    vector<lower=-100, upper=100>[nfreq] Rmin; // Response at 0% contrast    
    matrix<lower=0.001>[nfreq, nsamp] P; // Exponent        
    matrix<lower=0.001, upper=0.99>[nfreq, nsamp] cfifty; // 50% contrast response
    cholesky_factor_corr[nfreq] Lcorr_all;     
    vector<lower=0>[nfreq] tau;
}

transformed parameters {
    vector[nfreq] mu[N]; // predicted power
    matrix[nfreq, nfreq] L_Sigma;

    for (n in 1:N){
        for (f in 1:nfreq){
            mu[n, f] = crf(
                contrast[n], 
                Rmax[f], 
                Rmin[f], 
                P[f, samples[n]], 
                cfifty[f, samples[n]]);
        }
    }
    L_Sigma = diag_pre_multiply(tau, Lcorr_all);
}


model {


    for (f in 1:nfreq){
        Rmax[f] ~ normal(rmax_emp[f], 1);
        Rmin[f] ~ normal(rmin_emp[f], 1);
    }
    for (s in 1:nsamp){
        for (f in 1:nfreq){
            P[f, s] ~ normal(1, 15);
            cfifty[f, s] ~ normal(0.5, 0.1);    
        }
    }
    
    tau ~ cauchy(0, 1);

    Lcorr_all ~ lkj_corr_cholesky(2);    
    //print(mu[1])
    y ~ multi_normal_cholesky(mu, L_Sigma);
    
}
