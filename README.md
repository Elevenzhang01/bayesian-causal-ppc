# bayesian-causal-ppc

This repository reproduces the **synthetic data experiments (the “fiction” scenario)** from the empirical study in

> **Model Criticism for Bayesian Causal Inference**  
> Dustin Tran, et al.

The goal is to compare different **posterior predictive check (PPC)** strategies for Bayesian causal inference under correctly specified and misspecified models.

---

## Experiment setup (fiction scenario)

We consider a binary treatment \( a \in \{0,1\} \), covariates \( x \), and potential outcomes \( y(0), y(1) \).
Data are generated from a known data-generating process (DGP), which allows direct diagnostics under controlled misspecification.

Five joint models are considered:

### 1. Right outcome / Right assignment
- Both outcome and treatment assignment models are correctly specified.

### 2. Right outcome / Wrong assignment (DGP 2)
- Outcome model is correct.
- Treatment assignment model is misspecified relative to the true DGP.

### 3–5. Wrong outcome models (assignment correct)
Three outcome misspecification settings with increasing severity:
- Drop 1 covariate
- Drop 2 covariates
- Drop 3 covariates

---

## Posterior predictive checks (PPC)

Two PPC constructions are compared:

### IPW-based PPC
- Uses inverse probability weighting (IPW)
- Relies on marginal posterior propensities
- Discrepancy is constructed using a weighted ATE estimator

### Gelman-style imputation PPC
- Imputes missing counterfactual outcomes from the posterior predictive distribution
- Constructs discrepancies using completed potential outcomes
- Closely follows the imputation-based PPC described in the paper

---

## Repository structure

```text
bayesian-causal-ppc/
├── stan/
│   ├── joint_right_right.stan
│   ├── joint_right_outcome_wrong_assignment_dgp2.stan
│   ├── joint_wrong_outcome_drop1_gelman.stan
│   ├── joint_wrong_outcome_drop2_ipw.stan
│   └── joint_wrong_outcome_drop3_ipw.stan
└── scripts/
    └── run_all.py
