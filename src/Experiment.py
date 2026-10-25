import json, joblib
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

from Training import WorkloadLSTM
from PymooProblemSolver import env, run_rolling, I_INSTR, P_TX, PAYLOAD_BITS, BANDWIDTH

p = json.load(open("best_params.json"))
scaler = joblib.load("scaler.pkl")
raw_dev, raw_test = np.load("raw_dev.npy"), np.load("raw_test.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WorkloadLSTM(3, p["hidden_size"], p["num_layers"], dropout=p["dropout"])
model.load_state_dict(torch.load("final_workload_lstm.pt", map_location=device))
model.to(device).eval()

# 1) Sanity sweep first: is there a real tradeoff?
u_ref = np.percentile(raw_dev[:, 0], 95)
cpi, mpki = raw_dev[:, 1].mean(), raw_dev[:, 2].mean()
lam_peak = 1.3 * env.edge.mu(1.0, cpi)
x = np.linspace(0, 1, 101)
for frac in [0.3, 0.6, 0.9]:
    lam = frac * lam_peak
    lat, en, g1, g2 = env.evaluate(x, np.full_like(x, 1.0), np.full_like(x, 1.0), lam, cpi, mpki)
    ok = (g1 <= 0) & (g2 <= 0)
    plt.plot(lat[ok] * 1e3, en[ok], label=f"load={frac:.0%} of peak")
plt.xlabel("Latency (ms)"); plt.ylabel("Energy (J)"); plt.legend(); plt.show()

# 2) Short rolling run to check timing and behavior
results, fronts = run_rolling(model, scaler, raw_dev, raw_test, env, p["seq_length"], steps=20)
print(results.describe())

# 3) Full run once that looks sane
# results, fronts = run_rolling(model, scaler, raw_dev, raw_test, env, p["seq_length"], steps=500)
# results.to_csv("results_predictive.csv", index=False)

def diagnose(env, cpi, mpki):
    tx = PAYLOAD_BITS / BANDWIDTH
    print(f"transmission energy per request: {P_TX * tx:.2e} J, tx time: {tx*1e3:.1f} ms")
    for name, t in [("edge", env.edge), ("cloud", env.cloud)]:
        mu = t.mu(1.0, cpi)
        for rho in (0.3, 0.9):
            lam = rho * mu
            W = (t.rtt + (tx if name == "cloud" else 0)) + 1 / (mu - lam)
            e_req = (t.power(lam, 1.0, cpi, mpki) - t.p_idle) / lam
            print(f"{name:5s} rho={rho:.1f} | latency {W*1e3:8.2f} ms | dyn+mem energy/req {e_req:.2e} J")

diagnose(env, raw_dev[:, 1].mean(), raw_dev[:, 2].mean())

cpi_ref = np.median(raw_dev[raw_dev[:, 1] > 0, 1])
I = 4 * 2.5e9 / (200 * cpi_ref)      # edge cores * f_max / (target mu * CPI)
print("cpi_ref =", cpi_ref, "| set I_INSTR =", f"{I:.3e}")