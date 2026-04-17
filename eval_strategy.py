"""Quick evaluator for strategy variants. Usage: uv run python eval_strategy.py <path> [--sims N]"""
import sys, json, subprocess

path = sys.argv[1]
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 500

result = subprocess.run(
    [sys.executable, "-m", "orderbook_pm_challenge", "run", path, "--simulations", str(sims), "--json"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"CRASH: {result.stderr[:500]}")
    sys.exit(1)

data = json.loads(result.stdout)
rs = [r for r in data["simulation_results"] if not r.get("failed")]
failed = len(data["simulation_results"]) - len(rs)

if not rs:
    print("CRASH: all sims failed")
    sys.exit(1)

mean = lambda xs: sum(xs) / len(xs)
edges = [r["total_edge"] for r in rs]
retails = [r["retail_edge"] for r in rs]
arbs = [r["arb_edge"] for r in rs]
fills = [r["fill_count"] for r in rs]

me = mean(edges)
mr = mean(retails)
ma = mean(arbs)
mf = mean(fills)
se = sorted(edges)

print(f"mean_edge={me:.2f} retail={mr:.2f} arb={ma:.2f} arb_ratio={abs(ma)/max(mr,0.01)*100:.1f}%")
print(f"fills={mf:.0f} P10={se[len(se)//10]:.2f} P50={se[len(se)//2]:.2f} P90={se[9*len(se)//10]:.2f}")
print(f"min={min(edges):.2f} max={max(edges):.2f} failures={failed}")
