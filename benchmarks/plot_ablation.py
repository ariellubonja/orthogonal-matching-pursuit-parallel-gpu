import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'image_patches': 'Image patches\n(256×1024, S=32, B=5K)',
    'face_recognition': 'Face recog.\n(8064×1207, S=30, B=1.2K)',
    'audio': 'Audio\n(512×2048, S=64, B=5K)',
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

    has_gpu = any(f'a1_batched_gpu' in results[c] for c in configs)
    devices = ['cpu', 'gpu'] if has_gpu else ['cpu']

    fig, axes = plt.subplots(1, len(devices), figsize=(7 * len(devices), 5), squeeze=False)

    for dev_idx, device in enumerate(devices):
        ax = axes[0][dev_idx]
        speedups = compute_speedups(results, device)

        x = np.arange(len(configs))
        width = 0.25
        offsets = [-width, 0, width]

        for i, ab in enumerate(ablations):
            vals = [speedups[cfg].get(ab, 0) for cfg in configs]
            bars = ax.bar(x + offsets[i], vals, width,
                         label=ABLATION_LABELS[ab], color=ABLATION_COLORS[ab],
                         edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    label = f'{val:.1f}x'
                    y_pos = max(bar.get_height(), 0.08)
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos * 1.15,
                            label, ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], fontsize=8)
        ax.set_ylabel('Speedup from optimization (>1 = helps)', fontsize=10)
        ax.set_title(f'{device.upper()} — Ablation: speedup from each optimization',
                     fontsize=11, fontweight='bold')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.set_yscale('log')
        all_vals_flat = [speedups[cfg].get(ab, 0) for cfg in configs for ab in ablations if speedups[cfg].get(ab, 0) > 0]
        if all_vals_flat:
            ax.set_ylim(bottom=min(all_vals_flat) * 0.5, top=max(all_vals_flat) * 2)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = os.path.join(heatmap_dir, f'ablation_bars.{ext}')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
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
