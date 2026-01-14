data {
  int<lower=1> n;
  int<lower=1> d;
  matrix[n, d] x;
  array[n] int<lower=0, upper=1> a;
  vector[n] y;                  // observed outcome y(a)
}

parameters {
  // outcome model
  vector[d] theta;
  real tau;
  real<lower=1e-12> sigma_sq;

  // assignment model
  vector[d] phi;
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);
  vector[n] a_vec;
  for (i in 1:n) a_vec[i] = a[i];   // safe int -> vector
}

model {
  // priors
  theta ~ normal(0, 1);
  tau   ~ normal(0, 2);
  sigma_sq ~ gamma(1, 1);
  phi   ~ normal(0, 1);

  // likelihoods
  a ~ bernoulli_logit(x * phi);
  y ~ normal(x * theta + a_vec * tau, sigma);
}

generated quantities {
  vector[n] y0_rep;
  vector[n] y1_rep;
  vector[n] pi1;
  real ate_rep;

  for (i in 1:n) {
    y0_rep[i] = normal_rng(dot_product(row(x, i), theta), sigma);
    y1_rep[i] = normal_rng(dot_product(row(x, i), theta) + tau, sigma);
    pi1[i]    = inv_logit(dot_product(row(x, i), phi));  // pi_i(1)
  }
  ate_rep = tau;
}
