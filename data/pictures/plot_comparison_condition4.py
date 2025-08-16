import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.patches as patches

def shaded_error_bar(x, y, yerr, color, linewidth=2.0, alpha=0.2, label=None, linestyle='-'):
    """
    Python equivalent of MATLAB's shadedErrorBar function
    """
    line = plt.plot(x, y, color=color, linewidth=linewidth, label=label, linestyle=linestyle)[0]
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)
    return line

def process_data(data, offset=True, column_index=2):
    """
    Process data similar to MATLAB code for condition 4:
    - Add zero at beginning if offset=True
    - Reshape into groups of 5 and take mean
    - Length is 200 for condition 4
    """
    if offset:
        processed = np.zeros(200)
        processed[1:200] = data[0:199, column_index]  # Use specified column
    else:
        processed = data[:200, column_index]
    
    # Reshape to groups of 5 and take mean
    processed = processed.reshape(40, 5)  # 200/5 = 40 groups
    processed = np.mean(processed, axis=1)
    return processed

def plot_comparison(att_mean4, att_std4, maddpg_mean4, maddpg_std4, mappo_mean4, mappo_std4, 
                   bc_mean4, bc_std4, demo_mean4, demo_std4, column_index=2, title=""):
    """
    Create a comparison plot for condition 4 data
    """
    # Process all data for the specified column
    attmean = process_data(att_mean4, column_index=column_index)
    attstd = process_data(att_std4, column_index=column_index)

    maddpgmean = process_data(maddpg_mean4, column_index=column_index)
    maddpgstd = process_data(maddpg_std4, column_index=column_index)

    mappomean = process_data(mappo_mean4, column_index=column_index)
    mappostd = process_data(mappo_std4, column_index=column_index)

    bcmean = process_data(bc_mean4, column_index=column_index)
    bcstd = process_data(bc_std4, column_index=column_index)

    demomean = process_data(demo_mean4, column_index=column_index)
    demostd = process_data(demo_std4, column_index=column_index)

    # X-axis points (1:5:200 in MATLAB)
    x_points = np.arange(0, 200, 5)

    # Plot all algorithms with shaded error bars and different line styles
    shaded_error_bar(x_points, attmean, attstd, att_color, label='att-MADDPG', linestyle='-')
    shaded_error_bar(x_points, maddpgmean, maddpgstd, maddpg_color, label='MADDPG', linestyle='--')
    shaded_error_bar(x_points, mappomean, mappostd, mappo_color, label='MAPPO', linestyle='-.')
    shaded_error_bar(x_points, bcmean, bcstd, bc_color, label='bc-MADDPG', linestyle=':')
    shaded_error_bar(x_points, demomean, demostd, demo_color, label='demo-MADDPG', linestyle=(0, (3, 1, 1, 1)))

    # Add demonstration baseline line (34.32 for condition 4)
    plt.axhline(y=34.32, color='k', linestyle='--', linewidth=2.2, label='Demonstration')

    plt.xlabel('Episodes(x10)', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.title(title, fontsize=20, fontweight='bold')

    # Set legend to vertical layout on the right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.grid(False)
    plt.tight_layout()

# Load data
try:
    data = loadmat('all_data.mat')
    
    # Extract data arrays for condition 4 (suffix 4)
    att_mean4 = data['att_mean4']
    att_std4 = data['att_std4']
    maddpg_mean4 = data['maddpg_mean4']
    maddpg_std4 = data['maddpg_std4']
    mappo_mean4 = data['mappo_mean4']
    mappo_std4 = data['mappo_std4']
    bc_mean4 = data['bc_mean4']
    bc_std4 = data['bc_std4']
    demo_mean4 = data['demo_mean4']
    demo_std4 = data['demo_std4']
    
except FileNotFoundError:
    print("all_data.mat not found. Creating dummy data for demonstration.")
    # Create dummy data with similar structure (199 episodes for condition 4)
    episodes = 199
    att_mean4 = np.random.randn(episodes, 6) * 2 + 30  # Higher baseline for condition 4
    att_std4 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    maddpg_mean4 = np.random.randn(episodes, 6) * 2 + 28
    maddpg_std4 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    mappo_mean4 = np.random.randn(episodes, 6) * 2 + 29
    mappo_std4 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    bc_mean4 = np.random.randn(episodes, 6) * 2 + 31
    bc_std4 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    demo_mean4 = np.random.randn(episodes, 6) * 2 + 32
    demo_std4 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5

# Set up the plotting style
plt.rcParams.update({
    'font.size': 15,
    'font.weight': 'bold',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0
})

# Colors for different algorithms (consistent with plot_comparison.py)
att_color = [1.0, 0.0, 0.0]  # Red for att-MADDPG
maddpg_color = [0.0, 0.0, 1.0]  # Blue for MADDPG
mappo_color = [0.4660, 0.8740, 0.1880]  # Green for MAPPO
bc_color = [1.0, 0.5, 0.0]  # Orange for bc-MADDPG
demo_color = [0.5, 0.0, 0.5]  # Purple for demo-MADDPG

# ========================= Single Figure with All Algorithms - Condition 4 =========================
plt.figure(figsize=(14, 6))
plot_comparison(att_mean4, att_std4, maddpg_mean4, maddpg_std4, mappo_mean4, mappo_std4, 
               bc_mean4, bc_std4, demo_mean4, demo_std4, column_index=2, 
               title="Four grasping points scenario")

# Show the plot
plt.show() 