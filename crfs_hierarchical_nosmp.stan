/*
CRF model for MEG power data from visual cortex.

Reduced to ignore sample number.

Lcorr_all: Needs to be init'd with np.eye, because cholesky decomp for large (>8x8) 
covariance matrices fails. 
https://discourse.mc-stan.org/t/cholesky-factor-corr-initializing-with-invalid-values/3167

Hierarchichal structure:

CRF(Rmin, Rmax, p, c50)

Rmin ~ per subject / freq
Rmax ~ per subject / freq

(p ~ (per sample and freq ~ subject))
(c50 ~ (per sample and freq ~ subject))
        
Lcorr ~ per frequency
*/  
functions{
     real crf(real contrast, real Rmax, real Rmin,
              real p, real C50){        
        real result;
        real cpow;
        cpow = contrast ^ p;        
        result = Rmax * (cpow / (cpow + (C50 ^ p))) + Rmin;       
        /*if (is_inf(result) || is_nan(result)){
            print("Con: ", contrast, " Rmax: ", Rmax, " Rmin: ", Rmin, " p: ", p, " c50: ", C50, " result: ", result);
            print(tmp[100000000]);
        }*/
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
    vector[nfreq] rmax_emp[nsub];       // Each subject has own rmax
    vector[nfreq] rmin_emp[nsub];       // Each subject has own rmin
}

parameters {    
    // Priors:
    matrix<lower=0.001, upper=1000>[nsub, nfreq] Rmax;     // Max amplitude relative to Rmin
    matrix<lower=-100, upper=100>[nsub, nfreq] Rmin;       // Response at 0% contrast    
    vector<lower=0.001>[nfreq] P[nsub];                    // Exponent        
    vector<lower=0.001, upper=0.99>[nfreq] cfifty[nsub];   // 50% contrast response
    cholesky_factor_corr[nfreq] Lcorr_all;                 // Cholesky factor of cov matrix
    vector<lower=0>[nfreq] tau;                            // Variance of cov matrix

    // Variance priors
    matrix<lower=0>[nsub, nfreq] Pstd;               // STD of P
    matrix<lower=0>[nsub, nfreq] cfiftystd;          // STD of c50
    // Hyperpriors:
    vector<lower=0.001>[nfreq] P_pop;                    // Population prior for P
    vector<lower=0.001, upper=0.99>[nfreq] cfifty_pop;   // Population prior for c50
}

transformed parameters {
    matrix[nfreq, nfreq] L_Sigma;
    L_Sigma = diag_pre_multiply(tau, Lcorr_all);
}


model {

    for (sub in 1:nsub){
        for (f in 1:nfreq){
            Rmax[sub, f] ~ normal(rmax_emp[sub, f], .25);
            Rmin[sub, f] ~ normal(rmin_emp[sub, f], .25);
        }
    }
            
    for (f in 1:nfreq){        
        for (sub in 1:nsub){
            P[sub][f] ~ normal(P_pop[f], Pstd[f]);
            cfifty[sub][f] ~ normal(cfifty_pop[f], cfiftystd[f]);    
        }
        P_pop[f] ~ normal(1, 15);
        Pstd[f] ~ student_t(1, 0, 15);
        cfifty_pop[f] ~ normal(0.5, 0.1);
        cfiftystd[f] ~ student_t(1, 0, 0.01);

        
    }

    
    tau ~ cauchy(0, 25);

    Lcorr_all ~ lkj_corr_cholesky(2);   
    {
        vector[nfreq] mu[N]; // predicted power     
        for (n in 1:N){
            for (f in 1:nfreq){
                mu[n, f] = crf(
                    contrast[n], 
                    Rmax[subject[n]][f], 
                    Rmin[subject[n]][f], 
                    P[subject[n]][f], 
                    cfifty[subject[n]][f]);
            }
        }   

        y ~ multi_normal_cholesky(mu, L_Sigma);
    }
    
}
