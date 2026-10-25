import pandas as pd
df = pd.read_csv("results/benchmark_raw.csv")

# per-policy summary of x
print(df.groupby("policy").x.agg(["mean", "median", "std", "min", "max"]).round(3))

p = df[df.policy == "predictive"]
print("mean x (per interval):", p.x.mean())
print("share of requests offloaded:", (p.x * p.lam_true).sum() / p.lam_true.sum())
print("corr(x, realized load):", p.x.corr(p.lam_true))