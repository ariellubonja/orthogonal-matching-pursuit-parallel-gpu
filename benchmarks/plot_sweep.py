import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import json
import os
import sys
import glob

COLORS = {
    'sklearn':   '#888888',
    'spams':     '#9C27B0',
    'cr_sparse': '#00BCD4',
    'naive_cpu': '#2196F3',
    'v0_cpu':    '#FF9800',
    'v0_blas':   '#795548',
    'naive_gpu': '#4CAF50',
    'v0_gpu':    '#E91E63',
}
LABELS = {
    'sklearn':   'sklearn',
    'spams':     'SPAMS',
    'cr_sparse': 'cr-sparse (JAX GPU)',
    'naive_cpu': 'Naive CPU',
    'v0_cpu':    'v0 CPU',
    'v0_blas':   'v0 BLAS',
    'naive_gpu': 'Naive GPU',
    'v0_gpu':    'v0 GPU',
}
ABBREVS = {
    'sklearn':   'sk',
    'spams':     'SP',
    'cr_sparse': 'CR',
    'naive_cpu': 'nC',
    'v0_cpu':    'v0C',
    'v0_blas':   'BL',
    'naive_gpu': 'nG',
    'v0_gpu':    'v0G',
}


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


def build_speedup_matrix(data, alg, S, baseline='sklearn'):
    """Build 2D array of speedup vs baseline. NaN for OOM/missing/invalid."""
    N_values = data['meta']['N_values']
    B_values = data['meta']['B_values']
    mat = np.full((len(N_values), len(B_values)), np.nan)

    for i, N in enumerate(N_values):
        for j, B in enumerate(B_values):
            cell = data['cells'].get(f"{S}_{N}_{B}")
            if cell is None or cell == 'skip':
                continue
            base_data = cell.get(baseline)
            alg_data = cell.get(alg)
            if (isinstance(base_data, dict) and 'time' in base_data and
                    isinstance(alg_data, dict) and 'time' in alg_data):
                mat[i, j] = base_data['time'] / alg_data['time']
    return mat


def draw_heatmap(ax, mat, N_values, B_values, title, vmin=None, vmax=None):
    cmap = matplotlib.cm.YlGnBu.copy()
    cmap.set_bad(color='#cccccc')

    if vmin is None:
        vmin = max(np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.1, 0.1)
    if vmax is None:
        vmax = max(np.nanmax(mat) if not np.all(np.isnan(mat)) else 10, 1.1)

    im = ax.imshow(mat, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                   aspect='auto', origin='lower')

    ax.set_xticks(range(len(B_values)))
    ax.set_xticklabels(B_values, fontsize=8)
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels(N_values, fontsize=8)
    ax.set_xlabel('B (n_samples)', fontsize=9)
    ax.set_ylabel('N (n_components)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')

    for i in range(len(N_values)):
        for j in range(len(B_values)):
            val = mat[i, j]
            if np.isnan(val):
                # Check if it's OOM or invalid
                cell = None  # will be handled by caller
                ax.text(j, i, '', ha='center', va='center', fontsize=6, color='#666666')
            else:
                color = 'white' if val > 5 else 'black'
                ax.text(j, i, f'{val:.1f}x', ha='center', va='center', fontsize=6, color=color)

    return im


def draw_best_algorithm_heatmap(ax, data, S, algs):
    """Color each cell by which algorithm is fastest."""
    N_values = data['meta']['N_values']
    B_values = data['meta']['B_values']

    # Assign each algorithm an integer index for coloring
    alg_indices = {alg: idx for idx, alg in enumerate(algs)}
    colors_list = [COLORS[a] for a in algs]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors_list)

    mat = np.full((len(N_values), len(B_values)), np.nan)
    labels = [['' for _ in B_values] for _ in N_values]

    for i, N in enumerate(N_values):
        for j, B in enumerate(B_values):
            cell = data['cells'].get(f"{S}_{N}_{B}")
            if cell is None or cell == 'skip':
                continue
            best_alg = None
            best_sps = -1
            for alg in algs:
                alg_data = cell.get(alg)
                if isinstance(alg_data, dict) and 'sps' in alg_data:
                    if alg_data['sps'] > best_sps:
                        best_sps = alg_data['sps']
                        best_alg = alg
            if best_alg is not None:
                mat[i, j] = alg_indices[best_alg]
                labels[i][j] = ABBREVS[best_alg]

    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=len(algs) - 0.5,
                   aspect='auto', origin='lower', interpolation='nearest')

    ax.set_xticks(range(len(B_values)))
    ax.set_xticklabels(B_values, fontsize=8)
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels(N_values, fontsize=8)
    ax.set_xlabel('B (n_samples)', fontsize=9)
    ax.set_ylabel('N (n_components)', fontsize=9)
    ax.set_title('Best algorithm', fontsize=10, fontweight='bold')

    for i in range(len(N_values)):
        for j in range(len(B_values)):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLORS[a], label=LABELS[a]) for a in algs]
    ax.legend(handles=legend_patches, fontsize=6, loc='upper left',
              framealpha=0.8)

    return im


def plot_sweep_heatmaps(data, output_dir, baseline='sklearn'):
    # All algorithms present in data (for best-algorithm panel)
    every_alg = ['sklearn', 'spams', 'cr_sparse', 'naive_cpu', 'v0_cpu', 'v0_blas', 'naive_gpu', 'v0_gpu']

    # Check which algorithms actually have data
    has_gpu = any(
        isinstance(cell.get('v0_gpu'), dict)
        for cell in data['cells'].values()
        if isinstance(cell, dict)
    )
    if not has_gpu:
        every_alg = [a for a in every_alg if 'gpu' not in a]

    has_spams = any(
        isinstance(cell.get('spams'), dict)
        for cell in data['cells'].values()
        if isinstance(cell, dict)
    )
    if not has_spams:
        every_alg = [a for a in every_alg if a != 'spams']

    has_cr_sparse = any(
        isinstance(cell.get('cr_sparse'), dict)
        for cell in data['cells'].values()
        if isinstance(cell, dict)
    )
    if not has_cr_sparse:
        every_alg = [a for a in every_alg if a != 'cr_sparse']

    # Speedup heatmaps exclude baseline (you don't plot "X vs X")
    all_algs = [a for a in every_alg if a != baseline]

    baseline_label = LABELS.get(baseline, baseline)
    n_algs = len(all_algs)
    # Layout: n_algs + 1 (best) subplots per sparsity level
    ncols = min(n_algs + 1, 4)
    nrows = (n_algs + 1 + ncols - 1) // ncols

    for S in data['meta']['S_values']:
        # Compute global vmin/vmax across all algorithms for this S
        all_vals = []
        for alg in all_algs:
            mat = build_speedup_matrix(data, alg, S, baseline=baseline)
            valid = mat[~np.isnan(mat)]
            if len(valid) > 0:
                all_vals.extend(valid.tolist())

        if not all_vals:
            continue

        vmin = max(min(all_vals) * 0.8, 0.05)
        vmax = max(all_vals) * 1.2

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[None, :]
        elif ncols == 1:
            axes = axes[:, None]

        fig.suptitle(f'Speedup vs {baseline_label} — S={S}',
                     fontsize=13, fontweight='bold', y=1.02)

        for idx, alg in enumerate(all_algs):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col]
            mat = build_speedup_matrix(data, alg, S, baseline=baseline)
            im = draw_heatmap(ax, mat, data['meta']['N_values'],
                              data['meta']['B_values'],
                              LABELS.get(alg, alg), vmin=vmin, vmax=vmax)

        # Best-algorithm heatmap in next slot
        best_idx = n_algs
        row, col = best_idx // ncols, best_idx % ncols
        ax = axes[row][col]
        draw_best_algorithm_heatmap(ax, data, S, every_alg)

        # Hide unused subplots
        for idx in range(best_idx + 1, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row][col].set_visible(False)

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=f'Speedup vs {baseline_label}')

        plt.tight_layout(rect=[0, 0, 0.90, 0.96])
        suffix = f'_vs_{baseline}' if baseline != 'sklearn' else ''
        out_path = os.path.join(output_dir, f'sweep_heatmap_S{S}{suffix}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == '__main__':
    args = sys.argv[1:]
    baseline = 'sklearn'
    json_path = None

    for arg in args:
        if arg.startswith('--baseline='):
            baseline = arg.split('=', 1)[1]
        elif not arg.startswith('--'):
            json_path = arg

    if json_path is None:
        # Find most recent sweep JSON
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        sweep_files = sorted(glob.glob(os.path.join(results_dir, 'sweep_*.json')))
        if not sweep_files:
            print("No sweep JSON files found. Run: python benchmarks/benchmarks.py sweep")
            sys.exit(1)
        json_path = sweep_files[-1]
        print(f"Using most recent sweep: {json_path}")

    data = load_sweep(json_path)
    output_dir = os.path.dirname(json_path)
    plot_sweep_heatmaps(data, output_dir, baseline=baseline)
