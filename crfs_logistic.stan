functions{
     real crf(real contrast, real Rmax, real Rmin,
              real p, real C50){        
        //cpow = (contrast^p);
        return Rmax * ((contrast ^ p) / ((contrast ^ p) + (C50 ^ p))) + Rmin;       
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
    vector[nfreq] Rmax; // Max amplitude relative to Rmin
    vector[nfreq] Rmin; // Response at 0% contrast    
    matrix<lower=0.001>[nfreq, nsamp] P; // Exponent        
    matrix[nfreq, nsamp] cfifty; // 50% contrast response
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

    Rmax ~ normal(rmax_emp, 0.25);
    Rmin ~ normal(rmin_emp, 0.25);
    to_vector(P) ~ normal(0, 15);
    to_vector(cfifty) ~ normal(0.5, 0.1);    
    
    tau ~ cauchy(0, 1);
    
    Lcorr_all ~ lkj_corr_cholesky(2);    
    y ~ multi_normal_cholesky(mu, L_Sigma);
    
}
