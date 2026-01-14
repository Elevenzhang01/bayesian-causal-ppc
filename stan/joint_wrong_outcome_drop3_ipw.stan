data {
  int<lower=1> n;
  int<lower=4> d;     // need at least 3 dims if dropping 2
  matrix[n, d] x;
  array[n] int<lower=0, upper=1> a;
  vector[n] y;
}

parameters {
  // WRONG outcome model: ignore x[,1] and x[,2]
  vector[d-3] theta_free; 
  real tau;
  real<lower=1e-12> sigma_sq;

  // RIGHT assignment model
  vector[d] phi;
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);

  vector[d] theta;
  vector[n] a_vec;

  theta[1] = 0;
  theta[2] = 0;
  theta[3] = 0;
  theta[4:d] = theta_free;

  for (i in 1:n) a_vec[i] = a[i];
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
