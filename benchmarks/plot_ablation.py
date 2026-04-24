import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import json
import os
import sys
import glob


ABLATION_LABELS = {
    'a1': 'Batching',
    'a2': 'Gram precomputation',
    'a3': 'Inverse Cholesky',
}

ABLATION_COLORS = {
    'a1': '#2196F3',
    'a2': '#4CAF50',
    'a3': '#FF9800',
}

CONFIG_LABELS = {
    'image_patches': 'Image patches\n256×1024, S=32',
    'face_recognition': 'Face recog.\n8064×1207, S=30',
    'audio': 'Audio\n512×2048, S=64',
}


def load_ablation(path):
    with open(path) as f:
        return json.load(f)


def compute_speedups(results, device='cpu'):
    """Compute speedup factor for each ablation axis on each config."""
    configs = list(results.keys())
    ablations = ['a1', 'a2', 'a3']
    speedups = {}

    for cfg in configs:
        cell = results[cfg]
        speedups[cfg] = {}
        for ab in ablations:
            if ab == 'a1':
                num = cell.get(f'a1_loop_{device}', 0)
                den = cell.get(f'a1_batched_{device}', 0)
            elif ab == 'a2':
                num = cell.get(f'a2_no_precompute_{device}', 0)
                den = cell.get(f'a2_precompute_{device}', 0)
            elif ab == 'a3':
                num = cell.get(f'a3_std_chol_{device}', 0)
                den = cell.get(f'a3_inv_chol_{device}', 0)
            speedups[cfg][ab] = num / den if den > 0 and num > 0 else 0

    return speedups


def plot_ablation_bars(data, output_dir):
    results = data['results']
    configs = list(results.keys())
    ablations = ['a1', 'a2', 'a3']

    has_gpu = any('a1_batched_gpu' in results[c] for c in configs)

    cpu_speedups = compute_speedups(results, 'cpu')
    gpu_speedups = compute_speedups(results, 'gpu') if has_gpu else None

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    x = np.arange(len(configs))

    if has_gpu:
        bar_width = 0.13
        group_offsets = [-0.28, 0.0, 0.28]

        for i, ab in enumerate(ablations):
            cpu_x = x + group_offsets[i] - bar_width / 2 - 0.005
            gpu_x = x + group_offsets[i] + bar_width / 2 + 0.005
            cpu_vals = [cpu_speedups[cfg].get(ab, 0) for cfg in configs]
            gpu_vals = [gpu_speedups[cfg].get(ab, 0) for cfg in configs]

            cpu_bars = ax.bar(cpu_x, cpu_vals, bar_width,
                              color=ABLATION_COLORS[ab], hatch='///',
                              edgecolor='white', linewidth=0.5)
            gpu_bars = ax.bar(gpu_x, gpu_vals, bar_width,
                              color=ABLATION_COLORS[ab],
                              edgecolor='white', linewidth=0.5)

            for bars, vals in [(cpu_bars, cpu_vals), (gpu_bars, gpu_vals)]:
                for bar, val in zip(bars, vals):
                    if val > 0:
                        label = f'{val:.1f}x'
                        y_pos = max(bar.get_height(), 0.08)
                        ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.15,
                                label, ha='center', va='bottom',
                                fontsize=18, fontweight='bold')
    else:
        bar_width = 0.25
        offsets = [-bar_width, 0, bar_width]
        for i, ab in enumerate(ablations):
            vals = [cpu_speedups[cfg].get(ab, 0) for cfg in configs]
            bars = ax.bar(x + offsets[i], vals, bar_width,
                          color=ABLATION_COLORS[ab], hatch='///',
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    label = f'{val:.1f}x'
                    y_pos = max(bar.get_height(), 0.08)
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.15,
                            label, ha='center', va='bottom',
                            fontsize=18, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=30)
    ax.set_ylabel('Speedup', fontsize=32)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=31)

    all_vals_flat = []
    for cfg in configs:
        for ab in ablations:
            v_cpu = cpu_speedups[cfg].get(ab, 0)
            if v_cpu > 0:
                all_vals_flat.append(v_cpu)
            if has_gpu:
                v_gpu = gpu_speedups[cfg].get(ab, 0)
                if v_gpu > 0:
                    all_vals_flat.append(v_gpu)
    if all_vals_flat:
        ax.set_ylim(bottom=min(all_vals_flat) * 0.5, top=max(all_vals_flat) * 2)

    method_handles = [
        Patch(facecolor=ABLATION_COLORS['a1'], edgecolor='white', label=ABLATION_LABELS['a1']),
        Patch(facecolor=ABLATION_COLORS['a2'], edgecolor='white', label=ABLATION_LABELS['a2']),
        Patch(facecolor=ABLATION_COLORS['a3'], edgecolor='white', label=ABLATION_LABELS['a3']),
    ]
    extra_artists = []
    if has_gpu:
        device_handles = [
            Patch(facecolor='#666666', edgecolor='white', hatch='///', label='CPU'),
            Patch(facecolor='#666666', edgecolor='white', label='GPU'),
        ]
        methods_legend = ax.legend(handles=method_handles, fontsize=20,
                                   loc='lower center', bbox_to_anchor=(0.5, 1.18),
                                   ncol=3, frameon=False)
        ax.add_artist(methods_legend)
        device_legend = ax.legend(handles=device_handles, fontsize=20,
                                  loc='lower center', bbox_to_anchor=(0.5, 1.02),
                                  ncol=2, frameon=False)
        extra_artists = [methods_legend, device_legend]
    else:
        ax.legend(handles=method_handles, fontsize=20,
                  loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=3, frameon=False)

    plt.tight_layout()
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = os.path.join(heatmap_dir, f'ablation_bars.{ext}')
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    bbox_extra_artists=extra_artists)
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_ablation_times(data, output_dir):
    """Show absolute times: optimized vs ablated for each axis."""
    results = data['results']
    configs = list(results.keys())

    has_gpu = any('a1_batched_gpu' in results[c] for c in configs)
    devices = ['cpu', 'gpu'] if has_gpu else ['cpu']

    pairs = [
        ('a1', 'a1_batched', 'a1_loop', 'Batching'),
        ('a2', 'a2_precompute', 'a2_no_precompute', 'Gram precomputation'),
        ('a3', 'a3_inv_chol', 'a3_std_chol', 'Inverse Cholesky'),
    ]

    fig, axes = plt.subplots(len(devices), len(pairs),
                              figsize=(5 * len(pairs), 4 * len(devices)),
                              squeeze=False)

    for dev_idx, device in enumerate(devices):
        for pair_idx, (ab, key_opt, key_abl, title) in enumerate(pairs):
            ax = axes[dev_idx][pair_idx]
            suffix = f'_{device}'

            opt_times = [results[c].get(f'{key_opt}{suffix}', 0) for c in configs]
            abl_times = [results[c].get(f'{key_abl}{suffix}', 0) for c in configs]

            x = np.arange(len(configs))
            width = 0.35

            bars1 = ax.bar(x - width/2, opt_times, width, label='Optimized',
                          color=ABLATION_COLORS[ab], alpha=0.9)
            bars2 = ax.bar(x + width/2, abl_times, width, label='Without optimization',
                          color='#999999', alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels([c for c in configs], fontsize=8)
            ax.set_ylabel('Time (s)', fontsize=9)
            ax.set_title(f'{device.upper()} — {title}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_yscale('log')
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = os.path.join(heatmap_dir, f'ablation_times.{ext}')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    json_path = None
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if json_path is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        ablation_files = sorted(glob.glob(os.path.join(results_dir, 'ablation_*.json')))
        if not ablation_files:
            print("No ablation JSON files found. Run: python benchmarks/benchmarks.py ablation")
            sys.exit(1)
        json_path = ablation_files[-1]
        print(f"Using most recent ablation: {json_path}")

    data = load_ablation(json_path)
    output_dir = os.path.dirname(json_path)
    plot_ablation_bars(data, output_dir)
    plot_ablation_times(data, output_dir)
