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
    Process data similar to MATLAB code for condition 6:
    - Add zero at beginning if offset=True
    - Reshape into groups of 5 and take mean
    - Length is 350 for condition 6
    """
    if offset:
        processed = np.zeros(350)
        processed[1:350] = data[0:349, column_index]  # Use specified column
    else:
        processed = data[:350, column_index]
    
    # Reshape to groups of 5 and take mean
    processed = processed.reshape(70, 5)  # 350/5 = 70 groups
    processed = np.mean(processed, axis=1)
    return processed

def plot_comparison(att_mean6, att_std6, maddpg_mean6, maddpg_std6, mappo_mean6, mappo_std6, 
                   bc_mean6, bc_std6, demo_mean6, demo_std6, column_index=2, title=""):
    """
    Create a comparison plot for condition 6 data
    """
    # Process all data for the specified column
    attmean = process_data(att_mean6, column_index=column_index)
    attstd = process_data(att_std6, column_index=column_index)

    maddpgmean = process_data(maddpg_mean6, column_index=column_index)
    maddpgstd = process_data(maddpg_std6, column_index=column_index)

    mappomean = process_data(mappo_mean6, column_index=column_index)
    mappostd = process_data(mappo_std6, column_index=column_index)

    bcmean = process_data(bc_mean6, column_index=column_index)
    bcstd = process_data(bc_std6, column_index=column_index)

    demomean = process_data(demo_mean6, column_index=column_index)
    demostd = process_data(demo_std6, column_index=column_index)

    # X-axis points (1:5:350 in MATLAB)
    x_points = np.arange(0, 350, 5)

    # Plot all algorithms with shaded error bars and different line styles
    shaded_error_bar(x_points, attmean, attstd, att_color, label='att-MADDPG', linestyle='-')
    shaded_error_bar(x_points, maddpgmean, maddpgstd, maddpg_color, label='MADDPG', linestyle='--')
    shaded_error_bar(x_points, mappomean, mappostd, mappo_color, label='MAPPO', linestyle='-.')
    shaded_error_bar(x_points, bcmean, bcstd, bc_color, label='bc-MADDPG', linestyle=':')
    shaded_error_bar(x_points, demomean, demostd, demo_color, label='demo-MADDPG', linestyle=(0, (3, 1, 1, 1)))

    # Add demonstration baseline line (51.54 for condition 6)
    plt.axhline(y=51.54, color='k', linestyle='--', linewidth=2.2, label='Demonstration')

    # Set x-axis limit to match MATLAB (xlim([0,350]))
    plt.xlim([0, 350])

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
    
    # Extract data arrays for condition 6 (suffix 6)
    att_mean6 = data['att_mean6']
    att_std6 = data['att_std6']
    maddpg_mean6 = data['maddpg_mean6']
    maddpg_std6 = data['maddpg_std6']
    mappo_mean6 = data['mappo_mean6']
    mappo_std6 = data['mappo_std6']
    bc_mean6 = data['bc_mean6']
    bc_std6 = data['bc_std6']
    demo_mean6 = data['demo_mean6']
    demo_std6 = data['demo_std6']
    
except FileNotFoundError:
    print("all_data.mat not found. Creating dummy data for demonstration.")
    # Create dummy data with similar structure (349 episodes for condition 6)
    episodes = 349
    att_mean6 = np.random.randn(episodes, 6) * 2 + 45  # Higher baseline for condition 6
    att_std6 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    maddpg_mean6 = np.random.randn(episodes, 6) * 2 + 43
    maddpg_std6 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    mappo_mean6 = np.random.randn(episodes, 6) * 2 + 44
    mappo_std6 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    bc_mean6 = np.random.randn(episodes, 6) * 2 + 46
    bc_std6 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5
    demo_mean6 = np.random.randn(episodes, 6) * 2 + 47
    demo_std6 = np.abs(np.random.randn(episodes, 6)) * 0.5 + 0.5

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

# ========================= Single Figure with All Algorithms - Condition 6 =========================
plt.figure(figsize=(14, 6))
plot_comparison(att_mean6, att_std6, maddpg_mean6, maddpg_std6, mappo_mean6, mappo_std6, 
               bc_mean6, bc_std6, demo_mean6, demo_std6, column_index=2, 
               title="Six grasping points scenario")

# Show the plot
plt.show() 