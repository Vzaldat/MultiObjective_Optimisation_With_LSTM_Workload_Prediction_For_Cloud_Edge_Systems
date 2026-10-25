from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import torch
from Training import WorkloadLSTM as model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from dataclasses import dataclass

I_INSTR = 3.671e+08
E_MISS = 1e-7
PAYLOAD_BITS = 2e5
BANDWIDTH = 100e6
P_TX = 0.8
RHO_MAX = 0.9
CPI_MIN = 0.1361987 * 0.25

@dataclass
class Tier:
    name: str
    cores: int
    f_max: float
    p_idle: float
    kappa: float
    alpha: float
    rtt: float = 0.0
    cpi_factor: float = 1.0
    mpki_factor: float = 1.0

    def mu(self, s, cpi):
        cpi = np.maximum(cpi, CPI_MIN)
        return self.cores * self.f_max * s / (I_INSTR * cpi * self.cpi_factor)
    
    def power(self, lam, s, cpi, mpki):
        rho = np.clip(lam / self.mu(s, cpi), 0.0, 1.0)
        p_dyn = self.kappa * (self.f_max * s / 1e9) ** self.alpha * rho
        p_mem = lam * (I_INSTR / 1000.0) * mpki * self.mpki_factor * E_MISS

        return self.p_idle + p_dyn + p_mem


class OffloadEnv:
    def __init__(self, edge, cloud, dt=1.0):
        self.edge, self.cloud, self.dt = edge, cloud, dt

    def evaluate(self, x, s_e, s_c, lam, cpi, mpki):
        lam_c, lam_e = x * lam, (1 - x) * lam
        mu_e = self.edge.mu(s_e, cpi)
        mu_c = self.cloud.mu(s_c, cpi)
        tx = PAYLOAD_BITS / BANDWIDTH

        W_e = 1.0 / np.maximum(mu_e - lam_e, 1e-6 * mu_e)
        W_c = self.cloud.rtt + tx + 1.0 / np.maximum(mu_c - lam_c, 1e-6 * mu_c)

        latency = (1 - x) * W_e + x * W_c

        energy = (self.edge.power(lam_e, s_e, cpi, mpki) + self.cloud.power(lam_c, s_c, cpi, mpki) + lam_c * P_TX * tx) * self.dt

        g_edge = lam_e - RHO_MAX * mu_e
        g_cloud = lam_c - RHO_MAX * mu_c

        return latency, energy, g_edge, g_cloud

edge = Tier("edge", cores=4, f_max=2.5e9, p_idle=5.0, kappa=4.0, alpha=2.0)

cloud = Tier("cloud", cores=64, f_max=3.5e9, p_idle=20.0, kappa=5.0, alpha=3.0, rtt=0.02, cpi_factor=0.85, mpki_factor=0.6)

env = OffloadEnv(edge, cloud)

class OffloadProblem(Problem):
    # decision vector : [x (offload fraction), s_e (edge freq scale), s_c (cloud freq scale)]

    def __init__(self, env, lam, cpi, mpki):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=2, xl=np.array([0.0, 0.4, 0.4]), xu=np.array([1.0, 1.0, 1.0]))
        self.env, self.lam, self.cpi, self.mpki = env, lam, cpi, mpki

    def _evaluate(self, X, out, *args, **kwargs):
        lat, en, g1, g2 = self.env.evaluate(X[:, 0], X[:, 1], X[:, 2], self.lam, self.cpi, self.mpki)

        out["F"] = np.column_stack([lat, en])
        out["G"] = np.column_stack([g1, g2])

def solve_front(env, lam, cpi, mpki, pop=60, gens=40, seed=1):
    res = minimize(OffloadProblem(env, lam, cpi, mpki), NSGA2(pop_size=pop), ("n_gen", gens), seed=seed, verbose=False)

    if res.X is None:
        return None, None
    return np.atleast_2d(res.X), np.atleast_2d(res.F)

def pick(X, F, w=(0.5, 0.5)):

    nF = (F - F.min(0)) / (np.ptp(F, axis=0) + 1e-12)
    return X[np.argmin(nF @ np.array(w))]


def forecast_next_util(model, scaler, raw_window):
    device = next(model.parameters()).device
    x = torch.tensor(scaler.transform(raw_window)[None], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x).item()
    return y * scaler.data_range_[0] + scaler.data_min_[0]

def run_rolling(model, scaler, raw_dev, raw_test, env, seq_length, steps=200):
    u_ref = np.percentile(raw_dev[:, 0], 95)
    lam_peak = 1.3 * env.edge.mu(1.0, raw_dev[:, 1].mean())
    records, fronts = [], {}

    for t in range(steps):
        window = raw_test[t: t + seq_length]
        u_hat = max(forecast_next_util(model, scaler, window), 0.0)

        lam_hat = u_hat / u_ref * lam_peak
        cpi, mpki = window[-1, 1], window[-1, 2]

        X, F = solve_front(env, lam_hat, cpi, mpki)
        if X is None:
            continue
        x, s_e, s_c = pick(X, F)
        fronts[t] = F

        u_true, cpi_t, mpki_t = raw_test[t + seq_length]
        lam_true = u_true / u_ref * lam_peak
        lat, en, g1, g2 = env.evaluate(x, s_e, s_c, lam_true, cpi_t, mpki_t)
        records.append(dict(t=t, lam_hat=lam_hat, lam_true=lam_true, x=x, s_e=s_e, s_c=s_c, latency=lat, energy=en, violated=bool(max(g1, g2) > 0)))

    return pd.DataFrame(records), fronts

# scaler = MinMaxScaler()
# results, fronts = run_rolling(model, scaler, raw_dev, raw_test, env, p["seq_length"])
# print(results[["x", "latency", "energy"]].describe())
# print("Constraint violations from forecast error:", results["violated"].mean())

# import matplotlib.pyplot as plt
# for t in list(fronts)[::40]:
#     F = fronts[t][np.argsort(fronts[t][:, 0])]
#     plt.plot(F[:, 0] * 1e3, F[:, 1], "o-", label=f"t={t}")
# plt.xlabel("Latency (ms)"); plt.ylabel("Energy (J / interval)"); plt.legend(); plt.show()