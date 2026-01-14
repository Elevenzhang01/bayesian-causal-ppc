
data {
  int<lower=1> n;
  int<lower=3> d;     // need at least 3 dims if dropping 2
  matrix[n, d] x;
  array[n] int<lower=0, upper=1> a;
  vector[n] y;
}
parameters {
  // WRONG outcome model: ignore x[,1]
  vector[d-1] theta_free;         // learn theta[2:d] only
  real tau;
  real<lower=1e-12> sigma_sq;

  // RIGHT assignment model
  vector[d] phi;
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);

  vector[d] theta;                // full theta used in likelihood/GQ
  vector[n] a_vec;

  theta[1] = 0;                   // force theta1 = 0
  theta[2:d] = theta_free;

  for (i in 1:n) a_vec[i] = a[i]; // safe int -> vector
}

model {
  // priors
  theta_free ~ normal(0, 1);
  tau        ~ normal(0, 2);
  sigma_sq   ~ gamma(1, 1);

  phi ~ normal(0, 1);

  // likelihoods
  a ~ bernoulli_logit(x * phi);                 // RIGHT assignment
  y ~ normal(x * theta + a_vec * tau, sigma);   // WRONG outcome
}

generated quantities {
  vector[n] y0_rep;
  vector[n] y1_rep;
  vector[n] pi1;
  real ate_rep;

  for (i in 1:n) {
    y0_rep[i] = normal_rng(dot_product(row(x, i), theta), sigma);
    y1_rep[i] = normal_rng(dot_product(row(x, i), theta) + tau, sigma);
    pi1[i]    = inv_logit(dot_product(row(x, i), phi));   // pi_i(1)
  }

  ate_rep = tau;
}
