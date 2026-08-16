import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse

root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(root_dir, 'result')
sns.set_style("darkgrid")

def getdata(experiment_dir):
    # readcsv
    csv_dir = os.path.join(experiment_dir, "table.csv")
    d = pd.read_csv(csv_dir)
    rewards = list(d['best_fit'])
    return rewards

def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        smooth_data = np.convolve(data, y, "same")
        return smooth_data
    else:
        return data


def plot_offspring_attention_ablation(experiment_dir, smooth_window=2,
                                       metric='return'):
    """Plot the fixed-offspring controller ablation in the legacy style."""
    csv_path = os.path.join(experiment_dir, 'learning_curves.csv')
    data = pd.read_csv(csv_path)

    required_columns = {'arm', 'update', metric}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))

    arm_order = [
        'random_init', 'distill_only', 'inherit_only', 'inherit_distill']
    arm_labels = {
        'random_init': 'Random initialization',
        'distill_only': 'Attention distillation only',
        'inherit_only': 'Controller inheritance',
        'inherit_distill': 'Inheritance + attention distillation',
    }
    palette = sns.color_palette()
    fig, ax = plt.subplots(figsize=(7, 4))

    for color, arm in zip(palette, arm_order):
        arm_data = data[data['arm'] == arm].sort_values('update')
        if arm_data.empty:
            # Keep older three-arm result directories plottable.
            continue
        rewards = arm_data[metric].rolling(
            window=smooth_window, min_periods=1, center=True).mean()
        ax.plot(
            arm_data['update'],
            rewards,
            label=arm_labels[arm],
            color=color,
            linewidth=2.0,
        )

    env_name = str(data['env'].iloc[0]) if 'env' in data.columns else 'Walker-v0'
    ylabel = 'Reward' if metric == 'return' else 'Best-so-far reward'
    ax.tick_params(labelsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('PPO updates', fontsize=16)
    ax.set_title(env_name, fontsize=18)
    ax.legend(fontsize=10)
    plt.rcParams.update({'font.size': 9.5})
    fig.tight_layout()

    suffix = 'raw' if metric == 'return' else 'best_so_far'
    output_stem = os.path.join(
        experiment_dir, 'offspring_attention_ablation_' + suffix)
    fig.savefig(output_stem + '.png', dpi=600, pad_inches=0.0)
    fig.savefig(output_stem + '.pdf', dpi=600, pad_inches=0.0)
    plt.close(fig)
    return output_stem + '.png', output_stem + '.pdf'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MACD experiment results.')
    parser.add_argument(
        '--experiment-dir',
        default=os.path.join(
            current_dir, 'offspring_attention_ablation', 'Walker-v0', 'seed_101'),
    )
    parser.add_argument('--smooth-window', type=int, default=2)
    parser.add_argument(
        '--metric', choices=('return', 'best_return'), default='return')
    args = parser.parse_args()

    if args.smooth_window < 1:
        raise ValueError('--smooth-window must be at least 1')
    outputs = plot_offspring_attention_ablation(
        args.experiment_dir,
        smooth_window=args.smooth_window,
        metric=args.metric,
    )
    print('Saved:', outputs[0])
    print('Saved:', outputs[1])
