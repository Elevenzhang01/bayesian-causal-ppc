data {
  int<lower=1> n;
  int<lower=1> d;
  matrix[n, d] x;
  array[n] int<lower=0, upper=1> a;
  vector[n] y;                    // observed outcome y(a)
}

parameters {
  // RIGHT outcome model
  vector[d] theta;
  real tau;
  real<lower=1e-12> sigma_sq;     // <- IMPORTANT: avoid gamma_lpdf at 0

  // WRONG assignment model
  vector[d] phi;
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);
  vector[n] a_vec;
  for (i in 1:n) a_vec[i] = a[i];  // safe int -> vector
}

model {
  // priors
  theta ~ normal(0, 1);
  tau   ~ normal(0, 2);           
  sigma_sq ~ gamma(1, 1);

  phi ~ normal(0, 1);

  // WRONG assignment likelihood:
  // P(a=1|x) = 0.7 + 0.3 * inv_logit(x'phi)
  for (i in 1:n) {
    real eta = dot_product(row(x, i), phi);
    real pi1_tmp = 0.7 + 0.3 * inv_logit(eta);
    // clamp for numerical stability
    pi1_tmp = fmin(1 - 1e-12, fmax(1e-12, pi1_tmp));
    a[i] ~ bernoulli(pi1_tmp);
  }

  // RIGHT outcome likelihood
  y ~ normal(x * theta + a_vec * tau, sigma);
}

generated quantities {
  vector[n] y0_rep;
  vector[n] y1_rep;
  vector[n] pi1;
  real ate_rep;

  for (i in 1:n) {
    // rep potential outcomes under the (right) fitted outcome model
    y0_rep[i] = normal_rng(dot_product(row(x, i), theta), sigma);
    y1_rep[i] = normal_rng(dot_product(row(x, i), theta) + tau, sigma);

    // propensity under the (wrong) assignment model
    pi1[i] = 0.7 + 0.3 * inv_logit(dot_product(row(x, i), phi));
    pi1[i] = fmin(1 - 1e-12, fmax(1e-12, pi1[i]));
  }

  ate_rep = tau;
}
