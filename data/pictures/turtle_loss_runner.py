import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob
from scipy import stats

# 设置matplotlib参数以符合SCI论文要求
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 15,
    'axes.linewidth': 1,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})
plt.rcParams.update({
    # ... 你已有的参数
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.weight': 'bold',  # 会让刻度标签也加粗（注意会影响大部分文字）
})

def load_tensorboard_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_data(json_data):
    steps = [point[1] for point in json_data]
    values = [point[2] for point in json_data]
    return np.array(steps), np.array(values)

def interpolate_data(steps_list, values_list):
    min_step = max([steps.min() for steps in steps_list])
    max_step = min([steps.max() for steps in steps_list])
    common_steps = np.linspace(min_step, max_step, 1000)
    interpolated_values = []
    for steps, values in zip(steps_list, values_list):
        mask = (steps >= min_step) & (steps <= max_step)
        steps_filtered = steps[mask]
        values_filtered = values[mask]
        interpolated = np.interp(common_steps, steps_filtered, values_filtered)
        interpolated_values.append(interpolated)
    return common_steps, np.array(interpolated_values)

def calculate_confidence_interval(values_array, confidence=0.95):
    mean = np.mean(values_array, axis=0)
    sem = stats.sem(values_array, axis=0)
    n = values_array.shape[0]
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * sem
    return mean, mean - ci, mean + ci

def load_multi_seed_data(base_path):
    """
    加载多种子数据
    返回格式: {
        'metric_name': {
            't=3': [(steps1, values1), ...],
            't=4': [...]
        }
    }
    """
    data_dict = {}
    metrics = ['reward', 'train_offline_loss', 'train_online_loss',
               'validation_offline_loss', 'validation_online_loss']
    for metric in metrics:
        metric_path = os.path.join(base_path, metric)
        if not os.path.exists(metric_path):
            print(f"警告: 未找到指标文件夹 {metric_path}")
            continue
        data_dict[metric] = {}
        for t in [3, 4]:
            t_path = os.path.join(metric_path, f't={t}')
            if not os.path.exists(t_path):
                print(f"警告: 未找到文件夹 {t_path}")
                continue
            json_files = glob.glob(os.path.join(t_path, '*.json'))
            if len(json_files) == 0:
                print(f"警告: 在 {t_path} 中未找到JSON文件")
                continue
            seed_data = []
            for json_file in json_files:
                try:
                    json_data = load_tensorboard_json(json_file)
                    steps, values = extract_data(json_data)
                    seed_data.append((steps, values))
                    print(f"成功加载: {json_file}")
                except Exception as e:
                    print(f"加载文件时出错 {json_file}: {e}")
            if seed_data:
                data_dict[metric][f't={t}'] = seed_data
                print(f"  {metric} t={t}: 加载了 {len(seed_data)} 个种子的数据")
    return data_dict

def plot_metrics_with_confidence(data_dict, output_dir='data/pictures/figures'):
    os.makedirs(output_dir, exist_ok=True)
    colors = ['#1f77b4', '#ff7f0e']
    metric_labels = {
        'reward': ('Reward', 'Reward'),
        'train_offline_loss': ('Training Offline Loss', 'Loss'),
        'train_online_loss': ('Training Online Loss', 'Loss'),
        'validation_offline_loss': ('Validation Offline Loss', 'Loss'),
        'validation_online_loss': ('Validation Online Loss', 'Loss')
    }
    for metric_name, metric_data in data_dict.items():
        if metric_name not in metric_labels:
            continue
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        title, ylabel = metric_labels[metric_name]
        curves_info = {}
        for i, (setting, seed_data_list) in enumerate(metric_data.items()):
            if len(seed_data_list) == 0:
                continue
            steps_list = [data[0] for data in seed_data_list]
            values_list = [data[1] for data in seed_data_list]
            common_steps, interpolated_values = interpolate_data(steps_list, values_list)
            mean_values, lower_ci, upper_ci = calculate_confidence_interval(interpolated_values)
            t_value = setting.split('=')[1]
            label = f'{t_value} turtlebots'
            ax.plot(common_steps, mean_values,
                    color=colors[i], linestyle='-', label=label,
                    alpha=0.8, linewidth=2)
            ax.fill_between(common_steps, lower_ci, upper_ci,
                            color=colors[i], alpha=0.2)
            curves_info[setting] = (common_steps, mean_values)
        # 虚线延伸 t=3
        if 't=3' in curves_info and 't=4' in curves_info:
            steps_3, mean_3 = curves_info['t=3']
            steps_4, _ = curves_info['t=4']
            if steps_3.max() < steps_4.max():
                ax.hlines(mean_3[-1], xmin=steps_3.max(), xmax=steps_4.max(),
                          colors=colors[0], linestyles='--', linewidth=2, alpha=0.8)
        ax.set_xlabel('Training Epochs', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_title(title, fontsize=24, fontfamily='Times New Roman')
        ax.legend(loc='best', fontsize=14, prop={'family': 'Times New Roman'})
        for tick in ax.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{metric_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"图表已保存: {filepath}")

def plot_combined_losses_with_confidence(data_dict, output_dir='data/pictures/figures'):
    loss_metrics = ['train_offline_loss', 'train_online_loss',
                    'validation_offline_loss', 'validation_online_loss']
    available_losses = [m for m in loss_metrics if m in data_dict and len(data_dict[m]) > 0]
    if not available_losses:
        return
    n_plots = len(available_losses)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4.5), dpi=150)
    if n_plots == 1:
        axes = [axes]
    colors = ['#1f77b4', '#ff7f0e']
    loss_titles = {
        'train_offline_loss': 'Training Offline Loss',
        'train_online_loss': 'Training Online Loss',
        'validation_offline_loss': 'Validation Offline Loss',
        'validation_online_loss': 'Validation Online Loss'
    }
    for idx, metric_name in enumerate(available_losses):
        ax = axes[idx]
        metric_data = data_dict[metric_name]
        curves_info = {}
        for i, (setting, seed_data_list) in enumerate(metric_data.items()):
            if len(seed_data_list) == 0:
                continue
            steps_list = [data[0] for data in seed_data_list]
            values_list = [data[1] for data in seed_data_list]
            common_steps, interpolated_values = interpolate_data(steps_list, values_list)
            mean_values, lower_ci, upper_ci = calculate_confidence_interval(interpolated_values)
            t_value = setting.split('=')[1]
            label = f'{t_value} turtlebots'
            ax.plot(common_steps, mean_values,
                    color=colors[i], linestyle='-', label=label,
                    alpha=0.8, linewidth=2)
            ax.fill_between(common_steps, lower_ci, upper_ci,
                            color=colors[i], alpha=0.2)
            curves_info[setting] = (common_steps, mean_values)
        # 虚线延伸 t=3
        if 't=3' in curves_info and 't=4' in curves_info:
            steps_3, mean_3 = curves_info['t=3']
            steps_4, _ = curves_info['t=4']
            if steps_3.max() < steps_4.max():
                ax.hlines(mean_3[-1], xmin=steps_3.max(), xmax=steps_4.max(),
                          colors=colors[0], linestyles='--', linewidth=2, alpha=0.8)
        ax.set_xlabel('Training Epochs', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_ylabel('Loss', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_title(loss_titles[metric_name], fontsize=24, fontfamily='Times New Roman')
        ax.legend(loc='best', fontsize=14, prop={'family': 'Times New Roman'})
        for tick in ax.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'combined_losses.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"组合loss图表已保存: {filepath}")

def print_data_summary(data_dict):
    print("\n" + "="*50)
    print("数据摘要:")
    print("="*50)
    for metric_name, metric_data in data_dict.items():
        print(f"\n{metric_name}:")
        for setting, seed_data_list in metric_data.items():
            print(f"  {setting}: {len(seed_data_list)} 个种子")
            if len(seed_data_list) > 0:
                lengths = [len(data[1]) for data in seed_data_list]
                print(f"    数据点数量: {min(lengths)} - {max(lengths)}")

if __name__ == "__main__":
    base_path = 'data/pictures'
    print("开始加载多种子数据...")
    print(f"基础路径: {base_path}")
    data_dict = load_multi_seed_data(base_path)
    if not data_dict:
        print("错误: 没有成功加载任何数据，请检查文件路径结构。")
    else:
        print_data_summary(data_dict)
        print("\n开始生成图表...")
        plot_metrics_with_confidence(data_dict)
        plot_combined_losses_with_confidence(data_dict)
        print("\n所有图表生成完成！")
        print("图表保存在 'pictures/figures' 文件夹中")