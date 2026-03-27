import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os
import glob

# ── Load sweep data for panels 1-2 ──────────────────────────────────────────

results_dir = os.path.join(os.path.dirname(__file__), 'results')
sweep_files = sorted(glob.glob(os.path.join(results_dir, 'sweep_*.json')))
if not sweep_files:
    print("No sweep JSON found. Run: python benchmarks/benchmarks.py sweep")
    exit(1)
with open(sweep_files[-1]) as f:
    sweep = json.load(f)

# ── Data for panel 3: realistic benchmarks (AWS g7e.8xlarge — Xeon 8559C + RTX PRO 6000 Blackwell 102GB) ──

# (label, sklearn_sps, spams_sps, v0_cpu_sps, v0_blas_sps, v0_gpu_sps)
realistic = [
    ('Image patches\n256×1024, S=32',  594,  7139, 3257, 1811, 183904),
    ('Face recog.\n8064×1207, S=30',   352,  1717, 1408, 1731,  25158),
    ('Audio\n512×2048, S=64',          207,  2359,  926,  295,  28336),
]

COLORS = {
    'sklearn': '#888888',
    'spams':   '#2196F3',
    'v0_cpu':  '#FF9800',
    'v0_blas': '#795548',
    'v0_gpu':  '#E91E63',
}
LABELS = {
    'sklearn': 'sklearn',
    'spams':   'SPAMS (C++)',
    'v0_cpu':  'Ours — CPU',
    'v0_blas': 'Ours — CPU BLAS',
    'v0_gpu':  'Ours — GPU',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('OMP Benchmark Results — Xeon 8559C + RTX PRO 6000 Blackwell (AWS g7e.8xlarge)',
             fontsize=13, fontweight='bold', y=1.01)

# ── Panel 1: GPU vs SPAMS speedup heatmap (S=32, from sweep) ────────────────
ax = axes[0]
N_values = sweep['meta']['N_values']
B_values = sweep['meta']['B_values']
S = 32

mat = np.full((len(N_values), len(B_values)), np.nan)
for i, N in enumerate(N_values):
    for j, B in enumerate(B_values):
        cell = sweep['cells'].get(f"{S}_{N}_{B}")
        if cell is None or cell == 'skip':
            continue
        spams = cell.get('spams')
        gpu = cell.get('v0_gpu')
        if isinstance(spams, dict) and isinstance(gpu, dict):
            mat[i, j] = gpu['sps'] / spams['sps']

cmap = matplotlib.cm.RdYlGn.copy()
cmap.set_bad(color='#cccccc')
from matplotlib.colors import LogNorm
im = ax.imshow(mat, cmap=cmap, norm=LogNorm(vmin=0.05, vmax=10),
               aspect='auto', origin='lower')

ax.set_xticks(range(len(B_values)))
ax.set_xticklabels(B_values, fontsize=8)
ax.set_yticks(range(len(N_values)))
ax.set_yticklabels(N_values, fontsize=8)
ax.set_xlabel('B (n_samples)', fontsize=9)
ax.set_ylabel('N (n_components)', fontsize=9)
ax.set_title('GPU speedup over SPAMS (S=32)', fontsize=11, fontweight='bold')

for i in range(len(N_values)):
    for j in range(len(B_values)):
        val = mat[i, j]
        if not np.isnan(val):
            color = 'white' if val > 3 or val < 0.3 else 'black'
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color=color)
        else:
            ax.text(j, i, 'OOM', ha='center', va='center', fontsize=6, color='#999999')

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('GPU / SPAMS', fontsize=9)

# ── Panel 2: Speedup vs sklearn across sweep (S=32, selected algs) ──────────
ax = axes[1]

# For each algorithm, compute median speedup vs sklearn at each N
algs_to_plot = ['spams', 'v0_cpu', 'v0_gpu']
markers = {'spams': 's', 'v0_cpu': '^', 'v0_gpu': '*'}

for alg in algs_to_plot:
    ns, medians = [], []
    for N in N_values:
        speedups = []
        for B in B_values:
            cell = sweep['cells'].get(f"{S}_{N}_{B}")
            if cell is None or cell == 'skip':
                continue
            sk = cell.get('sklearn')
            al = cell.get(alg)
            if isinstance(sk, dict) and isinstance(al, dict):
                speedups.append(al['sps'] / sk['sps'])
        if speedups:
            ns.append(N)
            medians.append(np.median(speedups))

    ax.plot(ns, medians, color=COLORS[alg], marker=markers[alg],
            label=LABELS[alg], linewidth=2, markersize=7)

ax.axhline(1.0, color=COLORS['sklearn'], linestyle='--', linewidth=1, label='sklearn (1x)')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('N (n_components)', fontsize=10)
ax.set_ylabel('Median speedup vs sklearn', fontsize=10)
ax.set_title(f'Speedup vs N (S={S}, median across B)', fontsize=11, fontweight='bold')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.set_xticks(N_values)
ax.legend(fontsize=8.5)
ax.grid(True, which='both', alpha=0.3)

# ── Panel 3: Speedup bar chart — realistic benchmarks ───────────────────────
ax = axes[2]
bench_labels = [r[0] for r in realistic]
n = len(realistic)
x = np.arange(n)
bar_w = 0.2

bar_algs = [
    ('spams',   COLORS['spams'],  1),  # index into realistic tuple
    ('v0_cpu',  COLORS['v0_cpu'], 3),
    ('v0_gpu',  COLORS['v0_gpu'], 5),
]

for j, (key, col, idx) in enumerate(bar_algs):
    speedups = []
    for r in realistic:
        sps = r[idx]
        if sps is not None:
            speedups.append(sps / r[1])  # divide by sklearn_sps
        else:
            speedups.append(0)

    bars = ax.bar(x + (j - 1) * bar_w, speedups,
                  bar_w, label=LABELS[key], color=col, alpha=0.85)
    for bar, sp in zip(bars, speedups):
        if sp > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{sp:.1f}x', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                    'OOM', ha='center', va='bottom', fontsize=7, color='#999999')

ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(bench_labels, fontsize=9)
ax.set_ylabel('Speedup vs sklearn', fontsize=10)
ax.set_title('Speedup — realistic benchmarks', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(results_dir, 'benchmark_plot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
