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

def process_data(data, offset=True):
    """
    Process data similar to MATLAB code:
    - Add zero at beginning if offset=True
    - Reshape into groups of 5 and take mean
    """
    if offset:
        processed = np.zeros(150)
        processed[1:150] = data[0:149, 2]  # Column 3 in MATLAB (0-indexed as 2)
    else:
        processed = data[:150, 2]
    
    # Reshape to groups of 5 and take mean
    processed = processed.reshape(30, 5)  # 150/5 = 30 groups
    processed = np.mean(processed, axis=1)
    return processed

# Load data
try:
    data = loadmat('all_data.mat')
    
    # Extract data arrays (assuming these variable names exist in the .mat file)
    att_mean3 = data['att_mean3']
    att_std3 = data['att_std3']
    maddpg_mean3 = data['maddpg_mean3']
    maddpg_std3 = data['maddpg_std3']
    mappo_mean3 = data['mappo_mean3']
    mappo_std3 = data['mappo_std3']
    bc_mean3 = data['bc_mean3']
    bc_std3 = data['bc_std3']
    demo_mean3 = data['demo_mean3']
    demo_std3 = data['demo_std3']
    
except FileNotFoundError:
    print("all_data.mat not found. Creating dummy data for demonstration.")
    # Create dummy data with similar structure
    episodes = 149
    att_mean3 = np.random.randn(episodes, 3) * 2 + 20
    att_std3 = np.abs(np.random.randn(episodes, 3)) * 0.5 + 0.5
    maddpg_mean3 = np.random.randn(episodes, 3) * 2 + 18
    maddpg_std3 = np.abs(np.random.randn(episodes, 3)) * 0.5 + 0.5
    mappo_mean3 = np.random.randn(episodes, 3) * 2 + 19
    mappo_std3 = np.abs(np.random.randn(episodes, 3)) * 0.5 + 0.5
    bc_mean3 = np.random.randn(episodes, 3) * 2 + 21
    bc_std3 = np.abs(np.random.randn(episodes, 3)) * 0.5 + 0.5
    demo_mean3 = np.random.randn(episodes, 3) * 2 + 22
    demo_std3 = np.abs(np.random.randn(episodes, 3)) * 0.5 + 0.5

# Set up the plotting style
plt.rcParams.update({
    'font.size': 15,
    'font.weight': 'bold',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0
})

# Colors for different algorithms
att_color = [1.0, 0.0, 0.0]  # Red for att-MADDPG
maddpg_color = [0.0, 0.0, 1.0]  # Blue for MADDPG
mappo_color = [0.4660, 0.8740, 0.1880]  # Green for MAPPO
bc_color = [1.0, 0.5, 0.0]  # Orange for bc-MADDPG
demo_color = [0.5, 0.0, 0.5]  # Purple for demo-MADDPG

# X-axis points (1:5:150 in MATLAB becomes 0-based indexing)
x_points = np.arange(0, 150, 5)

# ========================= Single Figure with All Algorithms =========================
plt.figure(figsize=(14, 6))

# Process all data
attmean3 = process_data(att_mean3)
attstd3 = process_data(att_std3)

maddpgmean3 = process_data(maddpg_mean3)
maddpgstd3 = process_data(maddpg_std3)

mappomean3 = process_data(mappo_mean3)
mappostd3 = process_data(mappo_std3)

bcmean3 = process_data(bc_mean3)
bcstd3 = process_data(bc_std3)

demomean3 = process_data(demo_mean3)
demostd3 = process_data(demo_std3)

# Plot all algorithms with shaded error bars and different line styles
shaded_error_bar(x_points, attmean3, attstd3, att_color, label='att-MADDPG', linestyle='-')
shaded_error_bar(x_points, maddpgmean3, maddpgstd3, maddpg_color, label='MADDPG', linestyle='--')
shaded_error_bar(x_points, mappomean3, mappostd3, mappo_color, label='MAPPO', linestyle='-.')
shaded_error_bar(x_points, bcmean3, bcstd3, bc_color, label='bc-MADDPG', linestyle=':')
shaded_error_bar(x_points, demomean3, demostd3, demo_color, label='demo-MADDPG', linestyle=(0, (3, 1, 1, 1)))

# Add demonstration baseline line
plt.axhline(y=24.95, color='k', linestyle='--', linewidth=2.2, label='Demonstration')

plt.xlabel('Episodes(x10)', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.title('Three grasping points scenario', fontsize=20, fontweight='bold')

# Set legend to vertical layout on the right side
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.grid(False)
plt.tight_layout()

# Show the plot
plt.show() 