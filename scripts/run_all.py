import os
import json
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_ppc(T_rep, T_obs, out_png, title):
    lo, hi = np.quantile(T_rep, [0.05, 0.95])
    T_real = np.mean(T_obs)

    plt.figure(figsize=(7.2, 3.2))
    plt.hist(T_rep, bins=60, density=True, alpha=0.35, label="Reference (T_rep)")
    plt.axvline(T_real, linewidth=3, label="Realized (mean T_obs)")
    plt.axvline(lo, linestyle="--", linewidth=3, label="90% interval")
    plt.axvline(hi, linestyle="--", linewidth=3)
    plt.xlabel("Adjusted MSE of ATE (T)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------------
# DGPs (content matches your notebook)
# -------------------------
def make_dgp1(seed=2, n=10000, d=10):
    np.random.seed(seed)
    x = np.random.uniform(0, 1, size=(n, d))

    phi_true = 0.5 * np.array([1,-1,1,-1, 0.5,-0.5, 0,0, 0.25,-0.25])
    pi_true = 1/(1+np.exp(-(x@phi_true)))
    a = np.random.binomial(1, pi_true, size=n).astype(int)

    theta_true = np.random.normal(0, 1, size=d)
    sigma_true = 1.0
    tau_true = 2.5

    y0 = (x @ theta_true) + np.random.normal(0, sigma_true, size=n)
    y1 = (x @ theta_true + tau_true) + np.random.normal(0, sigma_true, size=n)
    y  = a * y1 + (1 - a) * y0

    return {"n": n, "d": d, "x": x, "a": a, "y": y, "y0": y0, "y1": y1}

def make_dgp2(seed=2, n=10000, d=10):
    np.random.seed(seed)
    x = np.random.uniform(0, 1, size=(n, d))

    phi_true = np.array([0.45, 0.10, 0.40, 0.05, 0.25, 0.00, 0.00, 0.00, 0.15, 0.00])
    b = 0.85
    pi_true = 1/(1+np.exp(-(b + x @ phi_true)))
    a = np.random.binomial(1, pi_true, size=n).astype(int)

    theta_true = np.random.normal(0, 1, size=d)
    sigma_true = 1.0
    tau_true = 2.5

    y0 = (x @ theta_true) + np.random.normal(0, sigma_true, size=n)
    y1 = (x @ theta_true + tau_true) + np.random.normal(0, sigma_true, size=n)
    y  = a * y1 + (1 - a) * y0

    return {"n": n, "d": d, "x": x, "a": a, "y": y, "y0": y0, "y1": y1}

# -------------------------
# IPW-marginal discrepancy (matches your notebook)
# -------------------------
def ppc_ipw_marginal(fit, a, y):
    y0_rep    = fit.stan_variable("y0_rep")    # (S, n)
    y1_rep    = fit.stan_variable("y1_rep")    # (S, n)
    tau_draws = fit.stan_variable("tau")       # (S,)
    pi1_draws = fit.stan_variable("pi1")       # (S, n)

    S = tau_draws.shape[0]
    eps = 1e-6

    pi1_bar = pi1_draws.mean(axis=0)
    pi1_bar = np.clip(pi1_bar, eps, 1 - eps)

    a_float = a.astype(float)
    ate_hat = np.mean(a_float * y / pi1_bar - (1.0 - a_float) * y / (1.0 - pi1_bar))

    T_rep = np.empty(S)
    T_obs = np.empty(S)

    for s in range(S):
        tau_s = tau_draws[s]
        ate_rep_s = np.mean(y1_rep[s] - y0_rep[s])
        T_rep[s] = tau_s**2 - 2.0 * tau_s * ate_rep_s
        T_obs[s] = tau_s**2 - 2.0 * tau_s * ate_hat

    tail_prob = np.mean(T_rep >= T_obs)
    return T_rep, T_obs, float(tail_prob)

# -------------------------
# Gelman-imputation discrepancy (already computed in Stan GQ)
# -------------------------
def ppc_gelman(fit):
    T_obs = fit.stan_variable("T_obs_gelman")  # (S,)
    T_rep = fit.stan_variable("T_rep_gelman")  # (S,)
    tail_prob = np.mean(T_rep >= T_obs)
    return T_rep, T_obs, float(tail_prob)

# -------------------------
# Run helper
# -------------------------
def run_one(model_path, data_out, seed=123, chains=4, parallel_chains=4, iter_sampling=2000, iter_warmup=2000):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(
        data=data_out,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        seed=seed
    )
    return fit

def main():
    ensure_dir("results")

    # -------------------------
    # Data
    # -------------------------
    dgp1 = make_dgp1(seed=2, n=10000, d=10)
    dgp2 = make_dgp2(seed=2, n=10000, d=10)

    # -------------------------
    # Exp1: DGP1 + Right/Right + IPW-marginal
    # -------------------------
    exp = "exp1_right_right_dgp1"
    outdir = os.path.join("results", exp); ensure_dir(outdir)
    fit = run_one("stan/joint_right_right.stan",
                  {"n": dgp1["n"], "d": dgp1["d"], "x": dgp1["x"], "a": dgp1["a"], "y": dgp1["y"]},
                  seed=101)
    T_rep, T_obs, tail = ppc_ipw_marginal(fit, dgp1["a"], dgp1["y"])
    save_json({"tail_prob": tail}, os.path.join(outdir, "summary.json"))
    plot_ppc(T_rep, T_obs, os.path.join(outdir, "ppc.png"),
             title=f"Outcome test (fiction, joint right model)  tail prob={tail:.3f}")

    # -------------------------
    # Exp2: DGP1 + Wrong outcome drop1 + Right assignment + IPW-marginal
    # -------------------------
    exp = "exp2_wrong_outcome_drop1_dgp1_ipw"
    outdir = os.path.join("results", exp); ensure_dir(outdir)
    fit = run_one("stan/joint_wrong_outcome_drop1_ipw.stan",
                  {"n": dgp1["n"], "d": dgp1["d"], "x": dgp1["x"], "a": dgp1["a"].astype(int), "y": dgp1["y"]},
                  seed=102)
    T_rep, T_obs, tail = ppc_ipw_marginal(fit, dgp1["a"], dgp1["y"])
    save_json({"tail_prob": tail}, os.path.join(outdir, "summary.json"))
    plot_ppc(T_rep, T_obs, os.path.join(outdir, "ppc.png"),
             title=f"Outcome test (fiction, wrong outcomes drop1)  tail prob={tail:.3f}")

    # -------------------------
    # Exp3: DGP1 + Wrong outcome drop3 + Right assignment + IPW-marginal
    # -------------------------
    exp = "exp3_wrong_outcome_drop3_dgp1_ipw"
    outdir = os.path.join("results", exp); ensure_dir(outdir)
    fit = run_one("stan/joint_wrong_outcome_drop3_ipw.stan",
                  {"n": dgp1["n"], "d": dgp1["d"], "x": dgp1["x"], "a": dgp1["a"].astype(int), "y": dgp1["y"]},
                  seed=103)
    T_rep, T_obs, tail = ppc_ipw_marginal(fit, dgp1["a"], dgp1["y"])
    save_json({"tail_prob": tail}, os.path.join(outdir, "summary.json"))
    plot_ppc(T_rep, T_obs, os.path.join(outdir, "ppc.png"),
             title=f"Outcome test (fiction, wrong outcomes drop3)  tail prob={tail:.3f}")

    # -------------------------
    # Exp4: DGP2 + Right outcome + Wrong assignment + IPW-marginal
    # -------------------------
    exp = "exp4_right_outcome_wrong_assign_dgp2_ipw"
    outdir = os.path.join("results", exp); ensure_dir(outdir)
    fit = run_one("stan/joint_right_outcome_wrong_assignment_dgp2.stan",
                  {"n": dgp2["n"], "d": dgp2["d"], "x": dgp2["x"], "a": dgp2["a"].astype(int), "y": dgp2["y"]},
                  seed=104)
    T_rep, T_obs, tail = ppc_ipw_marginal(fit, dgp2["a"], dgp2["y"])
    save_json({"tail_prob": tail}, os.path.join(outdir, "summary.json"))
    plot_ppc(T_rep, T_obs, os.path.join(outdir, "ppc.png"),
             title=f"Outcome test (fiction, right outcome + wrong assignment)  tail prob={tail:.3f}")

    # -------------------------
    # Exp5: DGP1 + Wrong outcome drop1 + Gelman impute PPC
    # -------------------------
    exp = "exp5_wrong_outcome_drop1_dgp1_gelman"
    outdir = os.path.join("results", exp); ensure_dir(outdir)
    fit = run_one("stan/joint_wrong_outcome_drop1_gelman.stan",
                  {"n": dgp1["n"], "d": dgp1["d"], "x": dgp1["x"], "a": dgp1["a"].astype(int), "y": dgp1["y"]},
                  seed=105)
    T_rep, T_obs, tail = ppc_gelman(fit)
    save_json({"tail_prob": tail}, os.path.join(outdir, "summary.json"))

    lo, hi = np.quantile(T_rep, [0.05, 0.95])
    T_real = T_obs.mean()

    plt.figure(figsize=(7.2, 3.2))
    plt.hist(T_rep, bins=60, density=True, alpha=0.35, label="Reference (T_rep, Gelman)")
    plt.axvline(T_real, linewidth=3, label="Realized (mean T_obs, Gelman)")
    plt.axvline(lo, linestyle="--", linewidth=3, label="90% interval")
    plt.axvline(hi, linestyle="--", linewidth=3)
    plt.xlabel("Discrepancy T (Gelman imputation)")
    plt.ylabel("Density")
    plt.title(f"Outcome test (fiction, wrong outcomes, Gelman)  tail prob={tail:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ppc.png"), dpi=200)
    plt.close()

    print("Done. Tail probs saved under results/*/summary.json")

if __name__ == "__main__":
    main()
