import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

def load_mat_data(data_folder, i):
    """加载对应的mat文件数据"""
    states_file = os.path.join(data_folder, f'mpc_visualisation{i}.mat')
    costs_file = os.path.join(data_folder, f'mpc_costs{i}.mat')
    
    try:
        states_data = sio.loadmat(states_file)
        costs_data = sio.loadmat(costs_file)
        
        # 从mat文件中提取实际数据（通常需要去掉文件元信息）
        states_key = [key for key in states_data.keys() if not key.startswith('__')][0]
        costs_key = [key for key in costs_data.keys() if not key.startswith('__')][0]
        
        states = states_data[states_key]
        costs = costs_data[costs_key].flatten()
        
        return states, costs
    except Exception as e:
        print(f"Error loading data for i={i}: {e}")
        return None, None

def get_uav_colors(num_uavs):
    """根据UAV数量生成颜色"""
    if num_uavs <= 3:
        return ['red', 'green', 'blue'][:num_uavs]
    elif num_uavs <= 10:
        # 使用matplotlib的标准颜色循环
        colors = ['red', 'green', 'blue', 'orange', 'purple', 
                 'brown', 'pink', 'gray', 'olive', 'cyan']
        return colors[:num_uavs]
    else:
        # 对于更多UAV，使用颜色映射
        cmap = plt.cm.tab20  # 20种不同颜色
        return [cmap(i / num_uavs) for i in range(num_uavs)]

def get_target_positions(num_uavs):
    """根据UAV数量生成目标位置"""
    if num_uavs == 1:
        return [(0, 0)]
    elif num_uavs == 2:
        return [(-1.5, 0), (1.5, 0)]
    elif num_uavs == 3:
        return [(-2.8, 0), (0, 0), (2.8, 0)]
    elif num_uavs == 4:
        return [[1.25,1.25],[-1.25,1.25],[1.25,-1.25],[-1.25,-1.25]]
    elif num_uavs == 6:
        return [[2.25,-2.25],[-2.25,-2.25],[1.2,0],[-1.2,0],[2.25,2.25],[-2.25,2.25]]
    else:
        # 圆形排列
        positions = []
        radius = 3
        for i in range(num_uavs):
            angle = 2 * np.pi * i / num_uavs
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
        return positions

def detect_num_uavs(states):
    """自动检测UAV数量"""
    # states的第三维应该是 2 * num_uavs (每个UAV有x,y坐标)
    state_dim = states.shape[2]
    num_uavs = state_dim // 2
    print(f"Detected {num_uavs} UAVs (state dimension: {state_dim})")
    return num_uavs

def assign_uav_to_target_colors(states, costs, num_uavs, target_positions):
    """根据无人机最终位置分配与目标点匹配的颜色"""
    colors = get_uav_colors(num_uavs)
    
    # 获取最优轨迹的最终位置
    sorted_indices = np.argsort(costs)
    best_trajectory_idx = sorted_indices[0]
    
    final_positions = []
    for uav_idx in range(num_uavs):
        x_col = uav_idx * 2
        y_col = uav_idx * 2 + 1
        final_x = states[-1, best_trajectory_idx, x_col] * 5
        final_y = states[-1, best_trajectory_idx, y_col] * 5
        final_positions.append((final_x, final_y))
    
    # 计算每个UAV到各个目标点的距离矩阵
    distance_matrix = np.zeros((num_uavs, len(target_positions)))
    for uav_idx, final_pos in enumerate(final_positions):
        for target_idx, target_pos in enumerate(target_positions):
            distance_matrix[uav_idx, target_idx] = np.sqrt(
                (final_pos[0] - target_pos[0])**2 + (final_pos[1] - target_pos[1])**2
            )
    
    # 使用贪心算法进行最优匹配
    uav_to_target_mapping = {}
    used_targets = set()
    
    # 按距离从小到大排序进行分配
    uav_target_pairs = []
    for uav_idx in range(num_uavs):
        for target_idx in range(len(target_positions)):
            uav_target_pairs.append((distance_matrix[uav_idx, target_idx], uav_idx, target_idx))
    
    uav_target_pairs.sort()  # 按距离排序
    
    for _, uav_idx, target_idx in uav_target_pairs:
        if uav_idx not in uav_to_target_mapping and target_idx not in used_targets:
            uav_to_target_mapping[uav_idx] = target_idx
            used_targets.add(target_idx)
    
    # 创建UAV颜色映射（UAV使用其对应目标点的颜色）
    uav_colors = {}
    for uav_idx in range(num_uavs):
        target_idx = uav_to_target_mapping[uav_idx]
        uav_colors[uav_idx] = colors[target_idx]
    
    return uav_colors, uav_to_target_mapping

def plot_single_trajectory(ax, states, costs, i, num_trajectories_to_show=6):
    """在指定的subplot上绘制单个轨迹图"""
    
    # 自动检测UAV数量
    num_uavs = detect_num_uavs(states)
    
    # 获取目标位置和基础颜色
    target_positions = get_target_positions(num_uavs)
    base_colors = get_uav_colors(num_uavs)
    
    # 根据UAV最终位置分配与目标匹配的颜色
    uav_colors, uav_to_target_mapping = assign_uav_to_target_colors(
        states, costs, num_uavs, target_positions
    )
    
    # 根据成本排序
    sorted_indices = np.argsort(costs)
    num_trajectories = min(num_trajectories_to_show, len(sorted_indices))
    
    # 为每个UAV绘制轨迹
    legend_handles = []
    legend_labels = []
    
    for uav_idx in range(num_uavs):
        x_col = uav_idx * 2      # x坐标列
        y_col = uav_idx * 2 + 1  # y坐标列
        color = uav_colors[uav_idx]  # 使用分配的颜色
        
        # 绘制最优轨迹（粗线）
        line = ax.plot(states[:, sorted_indices[0], x_col] * 5, 
                      states[:, sorted_indices[0], y_col] * 5, 
                      linewidth=2.5, color=color, 
                      label=f'UAV{uav_idx+1}')[0]
        legend_handles.append(line)
        legend_labels.append(f'UAV{uav_idx+1}')
        
        # 绘制其他轨迹（细线）
        for j in range(1, num_trajectories):
            ax.plot(states[:, sorted_indices[j], x_col] * 5, 
                    states[:, sorted_indices[j], y_col] * 5, 
                    linewidth=1.2, color='k', alpha=0.6)
        
        # 绘制起始点圆圈
        start_circle = Circle((states[0, 0, x_col] * 5, states[0, 0, y_col] * 5), 
                             0.15, color=color, fill=False, linewidth=2)
        ax.add_patch(start_circle)
    
    # 绘制目标区域（使用基础颜色顺序，确保与UAV颜色匹配）
    for target_idx, (target_x, target_y) in enumerate(target_positions):
        color = base_colors[target_idx]  # 使用基础颜色
        target_circle = Circle((target_x, target_y), 0.5, 
                             color=color, fill=False, 
                             linestyle='--', linewidth=2)
        ax.add_patch(target_circle)
    
    # 设置图形属性
    ax.set_aspect('equal')
    
    # 动态设置坐标轴范围
    all_x = states[:, :, ::2].flatten() * 5  # 所有x坐标
    all_y = states[:, :, 1::2].flatten() * 5  # 所有y坐标
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    # 添加边距
    margin = 1.0
    x_range = max(abs(x_max), abs(x_min)) + margin
    y_range = max(abs(y_max), abs(y_min)) + margin
    plot_range = max(x_range, y_range, 5)  # 最小范围为5
    
    ax.set_xlim([-plot_range, plot_range])
    ax.set_ylim([-plot_range, plot_range])
    ax.set_xlabel("x (m)", fontsize=21, fontweight='bold')
    ax.set_ylabel("y (m)", fontsize=21, fontweight='bold')
    ax.set_title(f"T = {i}", fontsize=25, fontweight='bold')
    
    # 设置图形边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    
    ax.tick_params(labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 只在第一个子图添加图例
    return legend_handles, legend_labels, num_uavs

def main():
    # 设置路径
    data_folder = "data/traj"
    output_folder = "data/traj/output"
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 选择4个时间步
    time_steps = [1, 6, 10, 16]  # 从1到35中选择4个
    
    # 创建包含4个子图的大图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    all_data = []
    uav_counts = []
    
    # 加载所有数据
    for i, time_step in enumerate(time_steps):
        print(f"\nProcessing time step {time_step}...")
        
        # 加载数据
        states, costs = load_mat_data(data_folder, time_step)
        
        if states is not None and costs is not None:
            print(f"States shape: {states.shape}, Costs shape: {costs.shape}")
            all_data.append((states, costs, time_step))
        else:
            print(f"Skipping time step {time_step} due to data loading error")
            all_data.append(None)
    
    # 绘制所有子图
    legend_handles = None
    legend_labels = None
    
    for i, data in enumerate(all_data):
        if data is not None:
            states, costs, time_step = data
            handles, labels, num_uavs = plot_single_trajectory(axes[i], states, costs, time_step)
            if legend_handles is None:  # 只保存第一个图的图例信息
                legend_handles = handles
                legend_labels = labels
            uav_counts.append(num_uavs)
        else:
            # 如果数据加载失败，创建空白子图
            axes[i].text(0.5, 0.5, f'Data not available\nfor t = {time_steps[i]}', 
                        ha='center', va='center', transform=axes[i].transAxes, 
                        fontsize=14, fontweight='bold')
            axes[i].set_xlim([-5, 5])
            axes[i].set_ylim([-5, 5])
            axes[i].set_xlabel("x (m)", fontsize=14, fontweight='bold')
            axes[i].set_ylabel("y (m)", fontsize=14, fontweight='bold')
            axes[i].set_title(f"t = {time_steps[i]}", fontsize=16, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    
    # 设置总标题
    if uav_counts:
        num_uavs = max(set(uav_counts))  # 取最常见的UAV数量
        fig.suptitle(f"Predicted Trajectory (n={num_uavs})", 
                     fontsize=24, fontweight='bold', y=1.0)
    else:
        fig.suptitle("Multi-UAV Trajectory Visualization", 
                     fontsize=24, fontweight='bold', y=1.0)
    
    # 添加全局图例
    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(legend_labels), fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # 保存图片
    output_file = os.path.join(output_folder, 'combined_trajectories.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined plot: {output_file}")
    
    # 总结信息
    if uav_counts:
        unique_counts = list(set(uav_counts))
        print(f"\n=== Summary ===")
        print(f"Processed {len([d for d in all_data if d is not None])} time steps")
        print(f"UAV configurations detected: {unique_counts}")
        print("Combined plot completed!")
    else:
        print("No data processed successfully.")

if __name__ == "__main__":
    main()