data {
  int<lower=1> n;
  int<lower=2> d;                 // drop 1 covariate => need at least 2
  matrix[n, d] x;
  array[n] int<lower=0, upper=1> a;
  vector[n] y;                    // observed outcome y(a)
}

parameters {
  // WRONG outcome model: ignore x[,1]
  vector[d-1] theta_free;         // learn theta[2:d] only
  real tau;
  real<lower=1e-12> sigma_sq;

  // RIGHT assignment model (kept, but Gelman imputation check won't use pi explicitly)
  vector[d] phi;
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);

  vector[d] theta;                // full theta used in likelihood/GQ
  vector[n] a_vec;

  theta[1] = 0;                   // force theta1 = 0 (WRONG)
  theta[2:d] = theta_free;

  for (i in 1:n) a_vec[i] = a[i];
}

model {
  // priors
  theta_free ~ normal(0, 1);
  tau        ~ normal(0, 2);
  sigma_sq   ~ gamma(1, 1);
  phi        ~ normal(0, 1);

  // likelihoods
  a ~ bernoulli_logit(x * phi);                 // RIGHT assignment
  y ~ normal(x * theta + a_vec * tau, sigma);   // WRONG outcome
}

generated quantities {
  // Gelman-style: impute missing counterfactuals, then compute ATE and discrepancy
  real ate_obs_imp;
  real ate_rep_imp;
  real T_obs_gelman;
  real T_rep_gelman;

  real sum_diff_obs = 0;
  real sum_diff_rep = 0;

  for (i in 1:n) {
    real mu0 = dot_product(row(x, i), theta);
    real mu1 = mu0 + tau;

    // (1) OBS: impute missing counterfactual given observed y(a)
    real y0_imp;
    real y1_imp;

    if (a[i] == 1) {
      y1_imp = y[i];
      y0_imp = normal_rng(mu0, sigma);
    } else {
      y0_imp = y[i];
      y1_imp = normal_rng(mu1, sigma);
    }
    sum_diff_obs += (y1_imp - y0_imp);

    // (2) REP: generate full potential outcomes, observe y_rep(a), then impute missing
    real y0_full = normal_rng(mu0, sigma);
    real y1_full = normal_rng(mu1, sigma);
    real y_rep_obs = (a[i] == 1) ? y1_full : y0_full;

    real y0_imp_rep;
    real y1_imp_rep;

    if (a[i] == 1) {
      y1_imp_rep = y_rep_obs;
      y0_imp_rep = normal_rng(mu0, sigma);
    } else {
      y0_imp_rep = y_rep_obs;
      y1_imp_rep = normal_rng(mu1, sigma);
    }
    sum_diff_rep += (y1_imp_rep - y0_imp_rep);
  }

  ate_obs_imp = sum_diff_obs / n;
  ate_rep_imp = sum_diff_rep / n;

  T_obs_gelman = tau * tau - 2.0 * tau * ate_obs_imp;
  T_rep_gelman = tau * tau - 2.0 * tau * ate_rep_imp;
}
