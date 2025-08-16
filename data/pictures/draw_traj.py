import numpy as np
import matplotlib.pyplot as plt

# 文件列表
file_list = [
    'data/traj/n=3episode_0.npz',
    'data/traj/n=3episode_1.npz',
    'data/traj/n=3episode_2.npz',
    'data/traj/n=3episode_3.npz',
    'data/traj/n=3episode_4.npz'
]
# file_list = [
#     'data/traj/n=4episode_0.npz',
#     'data/traj/n=4episode_1.npz',
#     'data/traj/n=4episode_2.npz',
#     'data/traj/n=4episode_3.npz',
#     'data/traj/n=4episode_4.npz'
# ]
# file_list = [
#     'data/traj/n=6episode_0.npz',
#     'data/traj/n=6episode_1.npz',
#     'data/traj/n=6episode_2.npz',
#     'data/traj/n=6episode_3.npz',
#     'data/traj/n=6episode_4.npz'
# ]

# 图标题列表
titles = ['Fixed Goal 1', 'Fixed Goal 2', 'Fixed Goal 3', 'Random Goal 1', 'Random Goal 2']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

# 创建子图
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

fig.text(-0.01, 0.5, f'{3} UAVs', rotation=90, va='center', ha='center', fontsize=24, fontweight='bold')
# 用于存储图例信息
legend_handles = []
legend_labels = []

# 目标区域半径
target_radius = 0.4
theta = np.linspace(0, 2*np.pi, 100)

for idx, (file_path, title) in enumerate(zip(file_list, titles)):
    ax = axes[idx]
    
    # 读取数据
    data = np.load(file_path)
    pos_x = data['pos_x']  # shape: [T, n_agents]
    pos_y = data['pos_y']
    
    n_steps, n_agents = pos_x.shape
    
    # 绘制轨迹
    for i in range(n_agents):
        line = ax.plot(pos_x[:, i], pos_y[:, i], color=colors[i], linewidth=2, label=f'UAV{i+1}')
        ax.scatter(pos_x[0, i], pos_y[0, i], color=colors[i], marker='o', s=60, edgecolor='k', zorder=5)  # 起点
        ax.scatter(pos_x[-1, i], pos_y[-1, i], color=colors[i], marker='x', s=60, edgecolor='k', zorder=5) # 终点
        
        # 只在第一个子图中收集图例信息
        if idx == 0:
            legend_handles.append(line[0])
            legend_labels.append(f'UAV{i+1}')
    
    # 在每个UAV的目标点（终点）周围画虚线圆圈
    for i in range(n_agents):
        # 目标点坐标（轨迹的最后一个点）
        target_x = pos_x[-1, i]
        target_y = pos_y[-1, i]
        
        # 计算圆圈上的点
        circle_x = target_x + target_radius * np.cos(theta)
        circle_y = target_y + target_radius * np.sin(theta)
        
        # 画灰色虚线圆圈
        ax.plot(circle_x, circle_y, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # 设置子图属性
    ax.set_xlabel('x (m)', fontsize=24)
    ax.set_ylabel('y (m)', fontsize=24)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=25)

# 调整子图布局
plt.tight_layout()

# 在五个图下方的左侧添加图例
fig.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(0.025, -0.15), 
           ncol=len(legend_labels), fontsize=18)

# 调整布局以为图例留出空间
plt.subplots_adjust(bottom=0.15)

plt.savefig('data/traj/fixed_all_episodes.png', dpi=150, bbox_inches='tight')
plt.show()