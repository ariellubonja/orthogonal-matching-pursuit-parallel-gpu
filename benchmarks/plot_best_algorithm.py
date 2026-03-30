import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import glob
import sys

# ── Load sweep data ──────────────────────────────────────────────

results_dir = os.path.join(os.path.dirname(__file__), 'results')

# Prefer AWS sweep, fall back to laptop
aws_files = sorted(glob.glob(os.path.join(results_dir, 'aws_sweep_*.json')))
sweep_files = sorted(glob.glob(os.path.join(results_dir, 'sweep_*.json')))
if aws_files:
    sweep_path = aws_files[-1]
elif sweep_files:
    sweep_path = sweep_files[-1]
else:
    print("No sweep JSON found. Run: python benchmarks/benchmarks.py sweep")
    sys.exit(1)

print(f"Using: {sweep_path}")
with open(sweep_path) as f:
    sweep = json.load(f)

N_values = sweep['meta']['N_values']
B_values = sweep['meta']['B_values']
S_values = sweep['meta']['S_values']

ALGS = ['sklearn', 'spams', 'v0_cpu', 'v0_blas', 'v0_gpu']
ALG_LABELS = {
    'sklearn': 'sklearn',
    'spams':   'SPAMS',
    'v0_cpu':  'v0 CPU',
    'v0_blas': 'v0 BLAS',
    'v0_gpu':  'v0 GPU',
}
ALG_COLORS = {
    'sklearn': '#888888',
    'spams':   '#2196F3',
    'v0_cpu':  '#FF9800',
    'v0_blas': '#795548',
    'v0_gpu':  '#E91E63',
}
ALG_INDEX = {alg: i for i, alg in enumerate(ALGS)}

# ── One heatmap per S value ──────────────────────────────────────

S = 8

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

hw = sweep['meta'].get('gpu', 'Unknown GPU')
cpu = sweep['meta'].get('cpu', 'Unknown CPU')
fig.suptitle(f'Fastest Algorithm per Config (S={S})',
             fontsize=12, fontweight='bold', y=1.02)

# Build a colormap from the algorithm colors
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap([ALG_COLORS[a] for a in ALGS])
bounds = np.arange(len(ALGS) + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

mat = np.full((len(N_values), len(B_values)), np.nan)
best_names = [['' for _ in B_values] for _ in N_values]

for i, N in enumerate(N_values):
    for j, B in enumerate(B_values):
        cell = sweep['cells'].get(f"{S}_{N}_{B}")
        if cell is None or cell == 'skip':
            continue

        best_alg = None
        best_sps = -1
        for alg in ALGS:
            entry = cell.get(alg)
            if isinstance(entry, dict) and entry.get('sps', 0) > best_sps:
                best_sps = entry['sps']
                best_alg = alg

        if best_alg is not None:
            mat[i, j] = ALG_INDEX[best_alg]
            best_names[i][j] = ALG_LABELS[best_alg]

ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto', origin='lower',
          interpolation='nearest')

ax.set_xticks(range(len(B_values)))
ax.set_xticklabels(B_values, fontsize=8)
ax.set_yticks(range(len(N_values)))
ax.set_yticklabels(N_values, fontsize=8)
ax.set_xlabel('B (n_samples)', fontsize=10)
ax.set_ylabel('N (n_components)', fontsize=10)

# Annotate cells with algorithm name
for i in range(len(N_values)):
    for j in range(len(B_values)):
        val = mat[i, j]
        if np.isnan(val):
            ax.text(j, i, 'skip', ha='center', va='center',
                    fontsize=5.5, color='#999999')
        else:
            alg_name = ALGS[int(val)]
            text_color = 'white' if alg_name in ('v0_gpu', 'spams', 'v0_blas') else 'black'
            ax.text(j, i, best_names[i][j], ha='center', va='center',
                    fontsize=7, fontweight='bold', color=text_color)

# Legend
patches = [mpatches.Patch(color=ALG_COLORS[a], label=ALG_LABELS[a]) for a in ALGS]
fig.legend(handles=patches, loc='lower center', ncol=len(ALGS),
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
heatmap_dir = os.path.join(results_dir, 'heatmaps')
os.makedirs(heatmap_dir, exist_ok=True)
for ext in ['png', 'pdf']:
    out_path = os.path.join(heatmap_dir, f'best_algorithm.{ext}')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
