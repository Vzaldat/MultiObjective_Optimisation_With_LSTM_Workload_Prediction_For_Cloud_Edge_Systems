import os, json, joblib
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

from Training import WorkloadLSTM, make_windows
from PymooProblemSolver import env, solve_front, pick, forecast_next_util
 
os.makedirs("results", exist_ok=True)
STEPS = 500          # number of evaluation points spread across the test set
LOAD_FACTOR = 1.3    # peak load = LOAD_FACTOR x edge capacity (must match run_rolling)
WEIGHTS = (0.5, 0.5) # latency / energy weights used to pick from each front
 
# ---------------------------------------------------------------- load artifacts
p = json.load(open("best_params.json"))
scaler = joblib.load("scaler.pkl")
raw_dev, raw_test = np.load("raw_dev.npy"), np.load("raw_test.npy")
seq = p["seq_length"]
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WorkloadLSTM(3, p["hidden_size"], p["num_layers"], dropout=p["dropout"])
model.load_state_dict(torch.load("final_workload_lstm.pt", map_location=device))
model.to(device).eval()
 
# u_max = raw_dev[:, 0].max()
lam_peak = LOAD_FACTOR * env.edge.mu(1.0, raw_dev[:, 1].mean())
# to_lam = lambda u: max(u, 0.0) / u_max * lam_peak
u_ref = np.percentile(raw_dev[:, 0], 95)
to_lam = lambda u: max(u, 0.0) / u_ref * lam_peak   # lam_peak = 1.3 x edge capacity

# ---------------------------------------------------------------- forecaster check
def forecaster_eval():
    """LSTM vs persistence MSE on the test set (scaled units)."""
    X, y = make_windows(scaler.transform(raw_test), seq).tensors
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            preds.append(model(X[i:i + 8192].to(device)).cpu())
    pred = torch.cat(preds)
    mse_lstm = torch.mean((pred - y) ** 2).item()
    mse_persist = torch.mean((X[:, -1, 0:1] - y) ** 2).item()
    mse_mean = torch.mean((float(scaler.transform(raw_dev)[:, 0].mean()) - y) ** 2).item()
    L = X.shape[1]
    w = torch.tensor([(1 - 0.2) ** (L - 1 - k) for k in range(L)]); w = (w / w.sum()).view(1, L)
    mse_ema = torch.mean(((X[:, :, 0] * w).sum(1, keepdim=True) - y) ** 2).item()
    print(f"mean-predictor MSE: {mse_mean:.6f} | EMA MSE: {mse_ema:.6f}")
    print(f"Forecaster test MSE | LSTM: {mse_lstm:.6f} | persistence: {mse_persist:.6f}")
    return dict(mse_mean=mse_mean mse_lstm=mse_lstm, mse_persistence=mse_persist, mse_ema=mse_ema)

 
 
# ---------------------------------------------------------------- policies
def decide(policy, lam, cpi, mpki):
    """Return (x, s_e, s_c, fell_back)."""
    if policy == "edge_only":  return 0.0, 1.0, 1.0, False
    if policy == "cloud_only": return 1.0, 1.0, 1.0, False
    if policy == "static_ws":  return 0.5, 1.0, 1.0, False
    X, F = solve_front(env, lam, cpi, mpki)
    if X is None:                         # no feasible point: fall back to cloud
        return 1.0, 1.0, 1.0, True
    x, s_e, s_c = pick(X, F, WEIGHTS)
    return x, s_e, s_c, False
 
 
POLICIES = ["predictive", "reactive","ema", "oracle", "edge_only", "cloud_only", "static_ws"]
 
# ---------------------------------------------------------------- main loop
idx = np.linspace(0, len(raw_test) - seq - 2, STEPS).astype(int)
rows, fronts = [], {}
 
for n, t in enumerate(idx):
    window = raw_test[t:t + seq]
    u_true, cpi_true, mpki_true = raw_test[t + seq]
    lam_true = to_lam(u_true)
 
    lam_hat = to_lam(forecast_next_util(model, scaler, window))
    lam_last = to_lam(window[-1, 0])
    cpi_p, mpki_p = window[-1, 1], window[-1, 2]   # persistence for CPI/MPKI
    
    lam_last = to_lam(window[-1, 0])
    lam_ema = to_lam(pd.Series(window[:, 0]).ewm(alpha=0.2).mean().iloc[-1])

    inputs = {
        "predictive": (lam_hat,  cpi_p,      mpki_p),
        "reactive":   (lam_last, cpi_p,      mpki_p),
        "ema": (lam_ema, cpi_p, mpki_p),
        "oracle":     (lam_true, cpi_true,   mpki_true),
        "edge_only":  (lam_hat,  cpi_p,      mpki_p),
        "cloud_only": (lam_hat,  cpi_p,      mpki_p),
        "static_ws":  (lam_hat,  cpi_p,      mpki_p),
    }
    for pol in POLICIES:
        lam_in, c_in, m_in = inputs[pol]
        x, s_e, s_c, fb = decide(pol, lam_in, c_in, m_in)
        lat, en, g1, g2 = env.evaluate(x, s_e, s_c, lam_true, cpi_true, mpki_true)
        rows.append(dict(step=n, t=t, policy=pol, x=x, s_e=s_e, s_c=s_c,
                         latency=float(lat), energy=float(en),
                         violated=bool(max(g1, g2) > 0), fallback=fb,
                         lam_true=lam_true, lam_hat=lam_hat, lam_last=lam_last))
 
    if n % 50 == 0:   # keep a few predictive fronts for the shifting-front figure
        X, F = solve_front(env, lam_hat, cpi_p, mpki_p)
        if X is not None:
            fronts[n] = (F, lam_hat / lam_peak)
    if n % 50 == 0:
        print(f"step {n}/{STEPS}")
 
df = pd.DataFrame(rows)
df.to_csv("results/benchmark_raw.csv", index=False)
 
# ---------------------------------------------------------------- summary table
summary = (df.groupby("policy")
             .agg(mean_latency_ms=("latency", lambda s: s.mean() * 1e3),
                  median_latency_ms=("latency", lambda s: s.median() * 1e3),
                  p95_latency_ms=("latency", lambda s: s.quantile(0.95) * 1e3),
                  mean_energy_J=("energy", "mean"),
                  violation_rate=("violated", "mean"),
                  fallbacks=("fallback", "sum"))
             .reindex(POLICIES))
print("\n", summary.round(4))
summary.round(4).to_csv("results/benchmark_summary.csv")
with open("results/benchmark_summary.md", "w") as f:
    f.write(summary.round(3).to_markdown())
 
fc = forecaster_eval()
json.dump(fc, open("results/forecaster.json", "w"))
 
# ---------------------------------------------------------------- figures
# (1) policies on the latency-energy plane + violation rate
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for pol in POLICIES:
    s = summary.loc[pol]
    ax[0].scatter(s.median_latency_ms, s.mean_energy_J, s=70, label=pol)
ax[0].set_xlabel("Median latency (ms)"); ax[0].set_ylabel("Mean energy (J / interval)")
ax[0].set_title("Policies on the latency-energy plane"); ax[0].legend(fontsize=8)
ax[1].bar(POLICIES, summary.violation_rate * 100)
ax[1].set_ylabel("Constraint violations (%)"); ax[1].set_title("Violation rate on realized load")
ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("results/policy_comparison.png", dpi=200); plt.close()
 
# (2) shifting Pareto front
plt.figure(figsize=(6, 4.5))
for n, (F, load_frac) in fronts.items():
    F = F[np.argsort(F[:, 0])]
    plt.plot(F[:, 0] * 1e3, F[:, 1], "o-", ms=3, label=f"load = {load_frac:.0%} of peak")
plt.xlabel("Latency (ms)"); plt.ylabel("Energy (J / interval)")
plt.title("Pareto front shifting with predicted load"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/pareto_shift.png", dpi=200); plt.close()
 
# (3) forecast vs realized load
d = df[df.policy == "predictive"]
plt.figure(figsize=(8, 3.5))
plt.plot(d.step, d.lam_true, label="realized", lw=1)
plt.plot(d.step, d.lam_hat, label="LSTM forecast", lw=1)
plt.plot(d.step, d.lam_last, label="last observed", lw=0.8, alpha=0.6)
plt.xlabel("evaluation step"); plt.ylabel("request rate (req/s)"); plt.legend()
plt.tight_layout(); plt.savefig("results/forecast_vs_actual.png", dpi=200); plt.close()
 
print("\nSaved tables and figures to results/")

