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
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 正确的字体名称
    'axes.labelsize': 16,  # x,y轴标签字体加大
    'axes.titlesize': 18,  # 标题字体加大
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,  # 图例字体加大
    'figure.titlesize': 20,
    'axes.linewidth': 1,  # 图片框框加粗
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
    """加载tensorboard导出的JSON文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_data(json_data):
    """从JSON数据中提取步数和值"""
    steps = [point[1] for point in json_data]  # 第二列是步数
    values = [point[2] for point in json_data]  # 第三列是值
    return np.array(steps), np.array(values)

def interpolate_data(steps_list, values_list):
    """
    将不同seed的数据插值到统一的步数网格上
    """
    # 找到所有数据的最小和最大步数
    min_step = max([steps.min() for steps in steps_list])
    max_step = min([steps.max() for steps in steps_list])
    
    # 创建统一的步数网格
    common_steps = np.linspace(min_step, max_step, 1000)
    
    # 对每个seed的数据进行插值
    interpolated_values = []
    for steps, values in zip(steps_list, values_list):
        # 确保数据在范围内
        mask = (steps >= min_step) & (steps <= max_step)
        steps_filtered = steps[mask]
        values_filtered = values[mask]
        
        # 插值
        interpolated = np.interp(common_steps, steps_filtered, values_filtered)
        interpolated_values.append(interpolated)
    
    return common_steps, np.array(interpolated_values)

def calculate_confidence_interval(values_array, confidence=0.95):
    """计算置信区间"""
    mean = np.mean(values_array, axis=0)
    sem = stats.sem(values_array, axis=0)  # 标准误差
    
    # 使用t分布计算置信区间
    n = values_array.shape[0]
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * sem
    
    return mean, mean - ci, mean + ci

def load_multi_seed_data(base_path):
    """
    加载多种子数据
    返回格式: {
        'metric_name': {
            'n=3': [(steps1, values1), (steps2, values2), ...],
            'n=4': [...],
            'n=6': [...]
        }
    }
    """
    data_dict = {}
    
    # 定义指标文件夹名称
    metrics = ['reward', 'train_offline_loss', 'train_online_loss', 
               'validation_offline_loss', 'validation_online_loss']
    
    for metric in metrics:
        metric_path = os.path.join(base_path, metric)
        if not os.path.exists(metric_path):
            print(f"警告: 未找到指标文件夹 {metric_path}")
            continue
            
        data_dict[metric] = {}
        
        # 遍历不同的n值
        for n in [3, 4, 6]:
            n_path = os.path.join(metric_path, f'n={n}')
            if not os.path.exists(n_path):
                print(f"警告: 未找到文件夹 {n_path}")
                continue
            
            # 加载该设置下的所有JSON文件
            json_files = glob.glob(os.path.join(n_path, '*.json'))
            if len(json_files) == 0:
                print(f"警告: 在 {n_path} 中未找到JSON文件")
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
                data_dict[metric][f'n={n}'] = seed_data
                print(f"  {metric} n={n}: 加载了 {len(seed_data)} 个种子的数据")
    
    return data_dict

def plot_metrics_with_confidence(data_dict, output_dir='data/pictures/figures'):
    """
    绘制带置信区间的指标图表
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义颜色，全部使用实线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    # 定义指标的显示名称和y轴标签
    metric_labels = {
        'reward': ('Reward', 'Reward'),
        'train_offline_loss': ('Training Offline Loss', 'Loss'),
        'train_online_loss': ('Training Online Loss', 'Loss'),
        'validation_offline_loss': ('Validation Offline Loss', 'Loss'),
        'validation_online_loss': ('Validation Online Loss', 'Loss')
    }
    
    # 为每个指标创建图表
    for metric_name, metric_data in data_dict.items():
        if metric_name not in metric_labels:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        title, ylabel = metric_labels[metric_name]
        
        # 绘制不同智能体数量的曲线
        for i, (setting, seed_data_list) in enumerate(metric_data.items()):
            if len(seed_data_list) == 0:
                continue
                
            # 提取所有种子的步数和值
            steps_list = [data[0] for data in seed_data_list]
            values_list = [data[1] for data in seed_data_list]
            
            # 插值到统一网格
            common_steps, interpolated_values = interpolate_data(steps_list, values_list)
            
            # 计算均值和置信区间
            mean_values, lower_ci, upper_ci = calculate_confidence_interval(interpolated_values)
            
            # 提取智能体数量并格式化图例
            n_value = setting.split('=')[1]
            label = f'{n_value} drones'
            
            # 绘制均值曲线 - 全部使用实线
            ax.plot(common_steps, mean_values, 
                   color=colors[i], 
                   linestyle='-',  # 全部使用实线
                   label=label,
                   alpha=0.8,
                   linewidth=2)
            
            # 绘制置信区间
            ax.fill_between(common_steps, lower_ci, upper_ci, 
                           color=colors[i], alpha=0.2)
        
        # 设置图表属性 - 标题不加粗
        ax.set_xlabel('Training Epochs', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_title(title, fontsize=18, fontfamily='Times New Roman')  # 去掉fontweight='bold'
        
        # 设置图例 - 加大字体
        legend = ax.legend(loc='best', fontsize=14, prop={'family': 'Times New Roman'})
        
        # # 设置边框加粗
        # for spine in ax.spines.values():
        #     spine.set_linewidth(2)
        
        # 设置刻度字体
        for tick in ax.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
        
        ax.grid(True, alpha=0.3)
        
        # 设置科学记数法（如果需要）
        if len(metric_data) > 0:
            # 获取第一个设置的数据来判断是否需要科学记数法
            first_setting_data = next(iter(metric_data.values()))
            if len(first_setting_data) > 0:
                max_step = max([data[0].max() for data in first_setting_data])
                if max_step > 10000:
                    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        filename = f"{metric_name.replace(' ', '_').lower()}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"图表已保存: {filepath}")
        plt.close()

def plot_combined_losses_with_confidence(data_dict, output_dir='data/pictures/figures'):
    """绘制所有loss指标的组合图（带置信区间）- 排成一排，每个子图保持合理比例"""
    
    loss_metrics = ['train_offline_loss', 'train_online_loss', 
                   'validation_offline_loss', 'validation_online_loss']
    
    # 检查是否有loss数据
    available_losses = [m for m in loss_metrics if m in data_dict and len(data_dict[m]) > 0]
    if not available_losses:
        return
    
    # 排成一排，但每个子图保持合理比例
    n_plots = len(available_losses)
    # 每个子图保持6x4.5的比例，总宽度会相应调整
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4.5), dpi=150)
    
    # 如果只有一个子图，确保axes是列表
    if n_plots == 1:
        axes = [axes]
    
    # 定义颜色，全部使用实线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    loss_titles = {
        'train_offline_loss': 'Training Offline Loss',
        'train_online_loss': 'Training Online Loss',
        'validation_offline_loss': 'Validation Offline Loss',
        'validation_online_loss': 'Validation Online Loss'
    }
    
    for idx, metric_name in enumerate(available_losses):
        ax = axes[idx]
        metric_data = data_dict[metric_name]
        
        for i, (setting, seed_data_list) in enumerate(metric_data.items()):
            if len(seed_data_list) == 0:
                continue
                
            # 提取所有种子的步数和值
            steps_list = [data[0] for data in seed_data_list]
            values_list = [data[1] for data in seed_data_list]
            
            # 插值到统一网格
            common_steps, interpolated_values = interpolate_data(steps_list, values_list)
            
            # 计算均值和置信区间
            mean_values, lower_ci, upper_ci = calculate_confidence_interval(interpolated_values)
            
            # 提取智能体数量并格式化图例
            n_value = setting.split('=')[1]
            label = f'{n_value} drones'
            
            # 绘制均值曲线和置信区间 - 全部使用实线
            ax.plot(common_steps, mean_values, 
                   color=colors[i], 
                   linestyle='-',  # 全部使用实线
                   label=label,
                   alpha=0.8,
                   linewidth=2)
            
            ax.fill_between(common_steps, lower_ci, upper_ci, 
                           color=colors[i], alpha=0.2)
        
        # 设置图表属性 - 标题不加粗
        ax.set_xlabel('Training Epochs', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_ylabel('Loss', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
        ax.set_title(loss_titles[metric_name], fontsize=24, fontfamily='Times New Roman')  # 去掉fontweight='bold'
        
        # 设置图例 - 加大字体
        ax.legend(loc='best', fontsize=14, prop={'family': 'Times New Roman'})
        
        # # 设置边框加粗
        # for spine in ax.spines.values():
        #     spine.set_linewidth(2)
        
        # 设置刻度字体
        for tick in ax.get_xticklabels():
            tick.set_fontfamily('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
        
        ax.grid(True, alpha=0.3)
        
        if len(metric_data) > 0:
            # 获取第一个设置的数据来判断是否需要科学记数法
            first_setting_data = next(iter(metric_data.values()))
            if len(first_setting_data) > 0:
                max_step = max([data[0].max() for data in first_setting_data])
                if max_step > 10000:
                    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'combined_losses.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', 
           facecolor='white', edgecolor='none', 
           frameon=False, pad_inches=0)
    print(f"组合loss图表已保存: {filepath}")
    plt.close()

def print_data_summary(data_dict):
    """打印数据摘要信息"""
    print("\n" + "="*50)
    print("数据摘要:")
    print("="*50)
    
    for metric_name, metric_data in data_dict.items():
        print(f"\n{metric_name}:")
        for setting, seed_data_list in metric_data.items():
            print(f"  {setting}: {len(seed_data_list)} 个种子")
            if len(seed_data_list) > 0:
                # 显示数据长度统计
                lengths = [len(data[1]) for data in seed_data_list]
                print(f"    数据点数量: {min(lengths)} - {max(lengths)}")

# 使用示例
if __name__ == "__main__":
    # 数据基础路径
    base_path = 'data/pictures'
    
    print("开始加载多种子数据...")
    print(f"基础路径: {base_path}")
    
    # 加载数据
    data_dict = load_multi_seed_data(base_path)
    
    if not data_dict:
        print("错误: 没有成功加载任何数据，请检查文件路径结构。")
        print("期望的文件结构:")
        print("pictures/")
        print("├── reward/")
        print("│   ├── n=3/ (包含5个.json文件)")
        print("│   ├── n=4/ (包含5个.json文件)")
        print("│   └── n=6/ (包含5个.json文件)")
        print("├── train_offline_loss/")
        print("│   ├── n=3/")
        print("│   ├── n=4/")
        print("│   └── n=6/")
        print("└── ...")
    else:
        # 打印数据摘要
        print_data_summary(data_dict)
        
        # 生成图表
        print("\n开始生成图表...")
        plot_metrics_with_confidence(data_dict)
        plot_combined_losses_with_confidence(data_dict)
        print("\n所有图表生成完成！")
        print("图表保存在 'pictures/figures' 文件夹中")