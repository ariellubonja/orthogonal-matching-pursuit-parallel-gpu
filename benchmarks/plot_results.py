import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Data from results/benchmark_20260308_000513.txt ──────────────────────────

# Paper Fig 1: N=8M, S=M/4, B=100
M_values = [16, 32, 64, 128, 256, 512, 1024, 2048]

paper_times = {
    'sklearn':   [0.014, 0.028, 0.092, 0.228, 0.782, 8.542,  63.266, 323.514],
    'naive_cpu': [0.002, 0.056, 0.274, 0.554, 2.376, 14.095, 43.306, 167.872],
    'v0_cpu':    [0.002, 0.017, 0.088, 0.149, 1.573,  1.800, 15.520,  43.205],
    'naive_gpu': [0.004, 0.006, 0.011, 0.021, 0.078,  0.443,  7.805,  67.132],
    'v0_gpu':    [0.002, 0.004, 0.008, 0.015, 0.034,  0.217,  1.640,    None],  # OOM at M=2048
}

# Realistic benchmarks: (label, n_features, n_components, sklearn_sps, naive_cpu_sps, v0_cpu_sps, naive_gpu_sps, v0_gpu_sps)
realistic = [
    ('Image patches\n(256×1024, k=32)', 256,  1024, 277,  1393, 1867,  7966, 21191),
    ('Face recog.\n(8064×1207, k=30)',  8064, 1207, 271,    77,  517,   255,  4134),
    ('Audio\n(512×2048, k=64)',          512,  2048,  99,   230,  429,  1374,  4173),
]

COLORS = {
    'sklearn':   '#888888',
    'naive_cpu': '#2196F3',
    'v0_cpu':    '#FF9800',
    'naive_gpu': '#4CAF50',
    'v0_gpu':    '#E91E63',
}
LABELS = {
    'sklearn':   'sklearn (CPU)',
    'naive_cpu': 'Naive CPU',
    'v0_cpu':    'v0 CPU',
    'naive_gpu': 'Naive GPU',
    'v0_gpu':    'v0 GPU',
}
MARKERS = {
    'sklearn':   'o',
    'naive_cpu': 's',
    'v0_cpu':    '^',
    'naive_gpu': 'D',
    'v0_gpu':    '*',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('OMP Benchmark Results — Intel Core Ultra 9 185H + RTX 4060 Laptop',
             fontsize=13, fontweight='bold', y=1.01)

# ── Panel 1: Absolute time vs M (paper Fig 1 reproduction) ───────────────────
ax = axes[0]
for key in ['sklearn', 'naive_cpu', 'v0_cpu', 'naive_gpu', 'v0_gpu']:
    times = paper_times[key]
    xs = [M_values[i] for i, t in enumerate(times) if t is not None]
    ys = [t for t in times if t is not None]
    ax.plot(xs, ys, color=COLORS[key], marker=MARKERS[key],
            label=LABELS[key], linewidth=1.8, markersize=6)

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('M  (n_features; N=8M, S=M/4, B=100)', fontsize=10)
ax.set_ylabel('Time (seconds)', fontsize=10)
ax.set_title('Paper Fig 1 — Solve time vs problem size', fontsize=11)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.set_xticks(M_values)
ax.legend(fontsize=8.5)
ax.grid(True, which='both', alpha=0.3)

# ── Panel 2: Speedup vs M (paper configs) ────────────────────────────────────
ax = axes[1]
sklearn_times = paper_times['sklearn']
for key in ['naive_cpu', 'v0_cpu', 'naive_gpu', 'v0_gpu']:
    times = paper_times[key]
    xs, speedups = [], []
    for i, t in enumerate(times):
        if t is not None:
            xs.append(M_values[i])
            speedups.append(sklearn_times[i] / t)
    ax.plot(xs, speedups, color=COLORS[key], marker=MARKERS[key],
            label=LABELS[key], linewidth=1.8, markersize=6)

ax.axhline(1.0, color=COLORS['sklearn'], linestyle='--', linewidth=1, label='sklearn (1x)')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('M  (n_features)', fontsize=10)
ax.set_ylabel('Speedup vs sklearn', fontsize=10)
ax.set_title('Speedup vs problem size (paper configs)', fontsize=11)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.set_xticks(M_values)
ax.legend(fontsize=8.5)
ax.grid(True, which='both', alpha=0.3)

# ── Panel 3: Speedup bar chart — realistic benchmarks ────────────────────────
ax = axes[2]
bench_labels = [r[0] for r in realistic]
n = len(realistic)
x = np.arange(n)
bar_w = 0.18

for j, (key, col) in enumerate([
        ('naive_cpu', COLORS['naive_cpu']),
        ('v0_cpu',    COLORS['v0_cpu']),
        ('naive_gpu', COLORS['naive_gpu']),
        ('v0_gpu',    COLORS['v0_gpu']),
]):
    idx = ['naive_cpu', 'v0_cpu', 'naive_gpu', 'v0_gpu'].index(key)
    speedups = [r[4 + idx] / r[3] for r in realistic]  # sps / sklearn_sps
    bars = ax.bar(x + (j - 1.5) * bar_w, speedups,
                  bar_w, label=LABELS[key], color=col, alpha=0.85)
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{sp:.1f}x', ha='center', va='bottom', fontsize=7.5)

ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(bench_labels, fontsize=9)
ax.set_ylabel('Speedup vs sklearn', fontsize=10)
ax.set_title('Speedup — realistic benchmarks', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'results', 'benchmark_plot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
